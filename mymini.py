"""
MyMini —— 编码器-解码器语言模型（minimind 训练体系兼容）

架构设计（对应 prompt.md 的研究假设）：
    1. 编码器：x -> x' -> x
       瓶颈结构（PCA / word2vec 风格）：
         - 降维矩阵 W 与重构矩阵 W^T 权重转置绑定（PCA 对称重构语义）；
         - 潜在空间经 SiLU 非线性（autoencoder / word2vec 风格）；
         - 重构增益（补偿 SiLU 斜率与转置投影能量损失）使初始重构近似保范数，
           避免多层串联信号衰减；
         - 每个瓶颈块附带重构辅助损失（recon_loss_coef），强迫 x' 保留输入信息；
         - 可堆叠 num_encoder_layers 个瓶颈块。
    2. 解码器：x -> (x + x') 上的注意力模型 -> x+1
         - GQA 分组查询注意力 + RoPE 旋转位置编码；
         - 注意力权重不调用 softmax，采用 log 域裁剪 + log-sum-exp 归一化：
           权重严格落在概率单纯形上，且任意两个 log 权重之差被显式截断，
           避免分值相差过大导致的梯度爆炸；
         - 解码块采用 Pre-LN + RMSNorm + SwiGLU FFN。
    3. 多头 Q/K/V/O 投影按头做正交初始化（每个头的投影行向量构成标准正交基）。
    4. 默认配置训练显存 < 12GB（梯度检查点 + GQA + bf16），内存 < 10GB。

训练入口：trainer/train_pretrain.py（model(input_ids) -> logits）。
兼容 SFT：model(input_ids, labels=labels) -> CausalLMOutput(.loss/.logits/.aux_loss)。
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
@dataclass
class MiniMindConfig:
    """MyMini 配置（字段名与 trainer/train_pretrain.py 的构造参数对齐）"""
    hidden_size: int = 1024          # 隐藏维度
    encoder_hidden_dim: int = 512    # 编码器瓶颈（x'）维度，推荐 hidden_size / 2
    num_attention_heads: int = 8
    num_kv_heads: int = 4            # GQA：KV 头数，需整除 num_attention_heads
    num_encoder_layers: int = 4      # 瓶颈块数量（x->x'->x 堆叠次数）
    num_decoder_layers: int = 12
    vocab_size: int = 6400           # 词表大小，需与 tokenizer 一致
    use_moe: bool = False            # 保留字段：checkpoint 命名/续训逻辑依赖
    dropout: float = 0.0
    intermediate_size: int = 0       # SwiGLU 中间维度，0 表示自动计算
    max_position_embeddings: int = 8192
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    log_clip: float = 4.0            # log 注意力分值截断半径（权重比 ≤ e^(2·clip)）
    recon_loss_coef: float = 0.1     # 编码器重构辅助损失权重（0 = 关闭）
    gradient_checkpointing: bool = True  # 训练时重算激活换显存（12GB 预算的关键开关）
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 "
                f"num_attention_heads({self.num_attention_heads}) 整除"
            )
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads({self.num_attention_heads}) 必须能被 "
                f"num_kv_heads({self.num_kv_heads}) 整除"
            )
        if not (0 < self.encoder_hidden_dim <= self.hidden_size):
            raise ValueError("encoder_hidden_dim 必须满足 0 < encoder_hidden_dim <= hidden_size")
        if self.intermediate_size <= 0:
            # Llama 风格 SwiGLU 比例（约 2.67×hidden，按 64 对齐）
            self.intermediate_size = math.ceil(self.hidden_size * 8 / 3 / 64) * 64


@dataclass
class CausalLMOutput:
    """轻量输出容器（SFT 等需要内部算损失的场景使用）。
    aux_loss 已乘好系数，调用方按 minimind 惯例直接 loss + aux_loss。"""
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    aux_loss: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                  通用工具 / 初始化
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
def init_orthogonal(module: nn.Module, gain: float = 1.0):
    """对 nn.Linear 做正交初始化（保留该公开函数供训练脚本导入/校验）"""
    if isinstance(module, nn.Linear) and module.weight.dim() == 2:
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.zero_()


def _orthogonal_per_heads(weight: Tensor, num_heads: int, head_dim: int):
    """按头正交初始化。

    将 [num_heads*head_dim, in_dim] 的权重按头切片，对每个头的
    [head_dim, in_dim] 子矩阵做正交化：每个头的投影行向量两两标准正交，
    且不同头从独立正交基中采样，保证多头之间初始方向互不相关。
    """
    with torch.no_grad():
        w = weight.view(num_heads, head_dim, -1)
        for i in range(num_heads):
            nn.init.orthogonal_(w[i])


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm.float()).type_as(x)


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                       RoPE
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6) -> Tuple[Tensor, Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float32), freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: Tensor, k: Tensor,
                         cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """q, k: [B, H, L, head_dim]；cos/sin: [L, head_dim]"""
    def rotate_half(x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_out = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_out, k_out


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                编码器：x -> x' -> x
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class BottleneckBlock(nn.Module):
    """单个 PCA/word2vec 风格瓶颈块：
        z  = SiLU(W x)        —— 编码到低维潜在空间 x'
        xr = g·W^T z          —— 用转置权重重构回 x（PCA 对称语义）

    重构增益 g = 2·sqrt(hidden/latent) 同时补偿两个初始衰减因子：
      1. SiLU 在零点附近斜率≈0.5（因子 2）；
      2. 转置投影 W^T W 只保留 latent 维行空间，各向同性输入下每块
         仅保留 latent/hidden 的能量（因子 sqrt(hidden/latent)）。
    两者叠加会使多块串联后重构信号衰减为零（无增益时 4 块堆叠实测
    ||x_rec||/||x||≈0.02，编码器成为死支路）。补偿后初始重构近似保范数；
    重构损失随后驱动 W 的行空间对齐输入的高能量方向（PCA 语义）。
    """

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, latent_dim, bias=False)
        self.recon_gain = 2.0 * math.sqrt(hidden_size / latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = F.silu(self.down(x))                    # [B, L, latent]
        x_rec = self.recon_gain * F.linear(z, self.down.weight.t())  # [B, L, hidden]
        # 重构损失：目标为该块输入（stop-grad），强迫瓶颈保留输入信息
        recon = F.mse_loss(x_rec, x.detach())
        return x_rec, z, recon


class PCAEncoder(nn.Module):
    """堆叠 num_encoder_layers 个瓶颈块。

    返回:
        x_rec : [B, L, hidden_size]  最后一块的重构表示（解码器与之残差融合）
        z     : [B, L, latent_dim]   最后一块的潜在表示 x'（信息瓶颈）
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            BottleneckBlock(config.hidden_size, config.encoder_hidden_dim)
            for _ in range(config.num_encoder_layers)
        ])
        # 降维矩阵按"主成分"方向正交初始化（W W^T = I_latent）
        for block in self.blocks:
            nn.init.orthogonal_(block.down.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        recon_loss = x.new_zeros(())
        h, z = x, None
        for block in self.blocks:
            h, z, r = block(h)
            recon_loss = recon_loss + r          # 各块重构损失求和
        return h, z, recon_loss


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                       解码器注意力：log 域归一化（不使用 softmax）
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class LogAttention(nn.Module):
    """Log-Attention：log 域裁剪 + log-sum-exp 归一化。

    与直接 softmax(qk^T) 的区别：
      1. 不调用 softmax；归一化完全在 log 域手工完成（减最大值后 exp）；
      2. 分值先截断到 [-log_clip, log_clip]：任意两个 log 权重之差
         ≤ 2·log_clip，最大/最小权重比被钉死在 e^(2·log_clip) 以内，
         从机制上保证输出是"彼此差距有界"的概率分布，避免分值发散
         造成的梯度消失/爆炸；
      3. 掩码位置以 -inf 精确置零，归一化计算在 float32 下进行。
    """

    def __init__(self, head_dim: int, log_clip: float = 4.0):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.log_clip = log_clip

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            q, k, v  : [B, H, L, head_dim]
            attn_mask: [1, 1, L, L] bool，True 表示允许注意
        Returns:
            output   : [B, H, L, head_dim]
        """
        # log 域：先缩放，再做有界截断（关键的防梯度爆炸约束）
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.clamp(-self.log_clip, self.log_clip)

        if attn_mask is not None:
            logits = logits.masked_fill(~attn_mask, float('-inf'))

        # log-sum-exp 归一化（float32）。因果掩码下每行至少有自身，max 有限。
        logits_fp = logits.float()
        max_log = logits_fp.max(dim=-1, keepdim=True).values
        weights = torch.exp(logits_fp - max_log)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return torch.matmul(weights, v.float()).to(v.dtype)


def create_causal_mask(seq_len: int, device=None) -> Tensor:
    """因果布尔掩码 [1, 1, L, L]，下三角（含对角）为 True"""
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    return mask.view(1, 1, seq_len, seq_len)


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                   解码块
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class DecoderLayer(nn.Module):
    """Pre-LN 解码块：GQA(LogAttention) + 残差 + RMSNorm + SwiGLU"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = dim // self.num_heads
        kv_dim = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # 多头正交初始化：Q/O 按注意力头、K/V 按 KV 头切分
        _orthogonal_per_heads(self.q_proj.weight, self.num_heads, self.head_dim)
        _orthogonal_per_heads(self.k_proj.weight, self.num_kv_heads, self.head_dim)
        _orthogonal_per_heads(self.v_proj.weight, self.num_kv_heads, self.head_dim)
        _orthogonal_per_heads(self.o_proj.weight, self.num_heads, self.head_dim)

        self.attention = LogAttention(self.head_dim, log_clip=config.log_clip)

        self.norm1 = RMSNorm(dim, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.dropout)

        # SwiGLU FFN: down(SiLU(gate(x)) * up(x))
        self.gate_proj = nn.Linear(dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, dim, bias=False)

        # FFN 小尺度初始化（注意力保持正交，FFN 负责可控的小幅残差更新）
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        bsz, kv_heads, seq, head_dim = x.shape
        if kv_heads == self.num_heads:
            return x
        n_rep = self.num_heads // kv_heads
        return x[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq, head_dim) \
                                  .reshape(bsz, self.num_heads, seq, head_dim)

    def _attn(self, x: Tensor, cos: Tensor, sin: Tensor,
              attn_mask: Optional[Tensor],
              past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
              use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE 作用于新 token 的 q/k（cos/sin 已按绝对位置切片）
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 拼接历史缓存（缓存中存的是 RoPE 之后的 k）
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        k, v = self._repeat_kv(k), self._repeat_kv(v)
        out = self.attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(out)), present

    def _ffn(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                attn_mask: Optional[Tensor] = None,
                past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        attn_out, present = self._attn(self.norm1(x), cos, sin, attn_mask,
                                       past_key_value, use_cache)
        x = x + attn_out
        x = x + self.resid_dropout(self._ffn(self.norm2(x)))
        return x, present


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                   整体模型
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MyMiniModel(nn.Module):
    """编码器-解码器：

        tokens -> Embedding
               -> PCAEncoder: x -> x' -> x_rec
               -> x + x_rec（解码器在"原始信息 + 瓶颈重构信息"上工作）
               -> N × DecoderLayer(GQA + LogAttention + SwiGLU)
               -> RMSNorm -> tied LM Head -> logits
    """

    def __init__(self, config: Optional[MiniMindConfig] = None):
        super().__init__()
        self.config = config if config is not None else MiniMindConfig()

        dim = self.config.hidden_size
        head_dim = dim // self.config.num_attention_heads

        # Token Embedding（padding 行不参与更新）
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, dim, padding_idx=self.config.pad_token_id,
        )

        # 编码器 x -> x' -> x
        self.pca_encoder = PCAEncoder(self.config)

        # 解码器入口：x + x' 的融合归一化
        self.enc_norm = RMSNorm(dim, eps=self.config.rms_norm_eps)
        self.embed_dropout = nn.Dropout(self.config.dropout)

        # 解码层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.config) for _ in range(self.config.num_decoder_layers)
        ])

        # 输出归一化 + LM Head
        self.norm = RMSNorm(dim, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(dim, self.config.vocab_size, bias=False)
        # 权重共享：方向为 lm_head -> embedding（保留 embedding 的 padding_idx 语义）
        self.lm_head.weight = self.embed_tokens.weight
        # 小尺度初始化嵌入矩阵（默认 N(0,1) 会使初始 logits 过大、CE 远超 ln(vocab)）；
        # 共享后 lm_head 同步生效，pad 行保持零
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed_tokens.weight[self.config.pad_token_id].zero_()

        # RoPE 频率（非持久 buffer，不进 checkpoint，由数据类型/设备动态适配）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, self.config.max_position_embeddings, self.config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids: Tensor,
                labels: Optional[Tensor] = None,
                return_dict: bool = False,
                past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
                use_cache: bool = False):
        """
        Args:
            input_ids       : [B, L] long（增量推理时为 [B, L_new]）
            labels          : [B, L] long，可选；给出时内部计算 shift CE 损失
            return_dict     : True 时即使无 labels 也返回 CausalLMOutput
            past_key_values : 各层缓存 (k, v)，k/v 为 [B, H_kv, L_past, head_dim]
                              （存的是 RoPE 之后的值）
            use_cache       : True 时在输出中返回更新后的 past_key_values
        Returns:
            无 labels 且 return_dict=False : logits [B, L, vocab_size]
            否则 : CausalLMOutput(loss/logits/aux_loss/hidden_states/past_key_values)
        """
        seq_len = input_ids.shape[1]
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        total_len = past_len + seq_len
        if total_len > self.config.max_position_embeddings:
            raise ValueError(
                f"总长度 {total_len} 超过最大位置长度 "
                f"{self.config.max_position_embeddings}"
            )

        # 1. Token Embedding（编码器只处理新 token）
        h = self.embed_dropout(self.embed_tokens(input_ids))  # [B, L_new, hidden]

        # 2. 编码器 x -> x' -> x_rec（同时得到各块重构损失之和）
        x_rec, _z, recon_loss = self.pca_encoder(h)

        # 3. 解码器入口：x + x'
        dec_in = self.enc_norm(h + x_rec)

        # 4. RoPE（按绝对位置切片）与因果掩码
        cos = self.freqs_cos[past_len:total_len].to(input_ids.device)
        sin = self.freqs_sin[past_len:total_len].to(input_ids.device)
        if past_len > 0:
            # 增量推理：新 token 对所有历史位置可见
            attn_mask = torch.ones(
                seq_len, total_len, device=input_ids.device, dtype=torch.bool,
            ).tril(diagonal=past_len).view(1, 1, seq_len, total_len)
        else:
            attn_mask = create_causal_mask(seq_len, device=input_ids.device)

        # 5. 解码层（训练时可选梯度检查点，将激活显存从 O(L·N) 降到 O(L)）
        use_ckpt = (
            self.training
            and self.config.gradient_checkpointing
            and input_ids.is_cuda
            and not use_cache
        )
        presents = []
        for i, layer in enumerate(self.decoder_layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_ckpt:
                dec_in, _ = torch.utils.checkpoint.checkpoint(
                    layer, dec_in, cos, sin, attn_mask,
                    use_reentrant=False,
                )
            else:
                dec_in, present = layer(dec_in, cos, sin, attn_mask,
                                        layer_past, use_cache)
                if use_cache:
                    presents.append(present)

        # 6. 输出
        hidden_states = self.norm(dec_in)
        logits = self.lm_head(hidden_states)

        # 编码器重构辅助损失（系数已乘好，调用方直接相加）
        aux_loss = self.config.recon_loss_coef * recon_loss

        if labels is None and not return_dict and not use_cache:
            return logits

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(
            loss=loss, aux_loss=aux_loss, logits=logits,
            hidden_states=hidden_states,
            past_key_values=tuple(presents) if use_cache else None,
        )

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 64,
                 temperature: float = 1.0, top_p: float = 0.9,
                 eos_token_id: Optional[int] = None) -> Tensor:
        """基于 KV cache 的自回归生成（贪心/采样）。

        首个 forward 处理完整 prompt 并建立缓存，之后每步只前向 1 个新 token。
        """
        self.eval()
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        out = self(input_ids, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        generated = [input_ids]
        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    cum_p = torch.cumsum(sorted_p, dim=-1)
                    mask = cum_p - sorted_p > top_p
                    sorted_p[mask] = 0.0
                    sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
                    next_id = sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
                else:
                    next_id = torch.multinomial(probs, 1)
            generated.append(next_id)
            if eos is not None and (next_id == eos).all():
                break
            out = self(next_id, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
        return torch.cat(generated, dim=1)


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                    测试
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
def test_basic():
    """接口 / DDP 安全 / 因果性 / 正交性 / 损失模式 综合测试"""
    config = MiniMindConfig(
        hidden_size=64, encoder_hidden_dim=32,
        num_attention_heads=8, num_kv_heads=4,
        num_encoder_layers=2, num_decoder_layers=2,
        vocab_size=100, max_position_embeddings=512,
        gradient_checkpointing=False,
    )
    model = MyMiniModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created. params: {n_params}")

    x = torch.randint(0, config.vocab_size, (2, 16), dtype=torch.long)

    # --- 1) 预训练契约：forward -> logits ---
    logits = model(x)
    assert tuple(logits.shape) == (2, 16, config.vocab_size)
    assert torch.isfinite(logits).all()
    print(f"Forward OK. logits: {tuple(logits.shape)}")

    # --- 1b) 编码器非死支路：重构尺度必须与输入同量级 ---
    with torch.no_grad():
        emb = model.embed_tokens(x)
        x_rec, _z, recon_sum = model.pca_encoder(emb)
    ratio = (x_rec.norm() / emb.norm()).item()
    assert 0.5 <= ratio <= 2.0, f"重构尺度异常 ||x_rec||/||x||={ratio:.3f}"
    assert recon_sum.item() > 0, "重构损失应非零（瓶颈秩 < hidden）"
    print(f"Encoder scale OK — ||x_rec||/||x||={ratio:.3f} ✓")

    # --- 2) DDP 安全：所有参数都参与损失（CE + 重构损失）---
    out_d = model(x, return_dict=True)
    loss = logits.sum() + out_d.aux_loss
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert not unused, f"DDP will fail on unused params: {unused}"
    print("All params participate in loss — DDP safe ✓")

    # --- 3) SFT 契约：labels -> CausalLMOutput，aux_loss 含重构项 ---
    model.zero_grad(set_to_none=True)
    labels = x.clone()
    out = model(x, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.aux_loss.item() > 0 and torch.isfinite(out.aux_loss).all()
    (out.loss + out.aux_loss).backward()
    print(f"Labels mode OK. loss: {out.loss.item():.4f}, "
          f"recon_aux: {out.aux_loss.item():.5f} ✓")

    # --- 4) 因果性：扰动未来 token 不影响当前位置输出 ---
    model.eval()
    x2 = x.clone()
    x2[0, 10:] = torch.randint(0, config.vocab_size, (6,))
    with torch.no_grad():
        l1 = model(x)
        l2 = model(x2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5), "因果掩码失效"
    print("Causality OK — future tokens do not affect present logits ✓")

    # --- 5) 多头正交初始化：每个头 W_i W_i^T ≈ I ---
    w = model.decoder_layers[0].q_proj.weight.view(8, 8, 64)
    for i in range(8):
        eye = w[i] @ w[i].t()
        assert torch.allclose(eye, torch.eye(8), atol=1e-5), f"head {i} 非正交"
    print("Per-head orthogonal init OK ✓")

    # --- 6) 权重共享：embedding 与 lm_head 是同一块存储 ---
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    print("Tied embeddings OK ✓")

    # --- 7) KV cache 与无缓存推理结果一致 ---
    model.eval()
    with torch.no_grad():
        # 无缓存：完整序列前向
        ref_logits = model(x)  # [2, 16, 100]
        # 有缓存：prompt 前 10 个 + 逐 token 增量
        out = model(x[:, :10], use_cache=True, return_dict=True)
        past = out.past_key_values
        cached_logits = [out.logits]
        for t in range(10, 16):
            out = model(x[:, t:t+1], past_key_values=past,
                        use_cache=True, return_dict=True)
            past = out.past_key_values
            cached_logits.append(out.logits)
        cached_logits = torch.cat(cached_logits, dim=1)
    diff = (ref_logits - cached_logits).abs().max().item()
    assert diff < 1e-4, f"KV cache 不一致，max diff={diff}"
    print(f"KV cache consistency OK — max diff={diff:.2e} ✓")

    # --- 8) generate() 产出形状与 EOS 截断 ---
    with torch.no_grad():
        gen = model.generate(x[:, :4], max_new_tokens=8, temperature=0.0)
    assert gen.shape[1] >= 4 and gen.shape[1] <= 12
    assert (gen[:, :4] == x[:, :4]).all(), "prompt 部分必须原样保留"
    print(f"generate OK — output shape {tuple(gen.shape)} ✓")

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    test_basic()
