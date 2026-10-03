import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
from dataclasses import *


@dataclass
class MiniMindConfig:
    """MiniMind Config with RoPE attention and PCA encoder"""
    hidden_size: int = 1024          # hidden dimension (adjustable)
    encoder_hidden_dim: int = 512    # PCA projection dim after encoding
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 12
    vocab_size: int = 6400           # 词表大小，需与 tokenizer 一致
    use_moe: bool = False            # 是否启用 MoE（保留字段以兼容 lm_checkpoint）
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


def init_orthogonal(module: nn.Module, dim: Optional[int] = None):
    """随机正交基初始化（修复：原 QR 在 (1, N) 上结果退化成标量）"""
    if isinstance(module, nn.Linear) and module.weight.dim() == 2:
        # 直接使用 PyTorch 内置正交初始化，避免手写 QR 的形状 bug
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()


class PCAEncoder(nn.Module):
    """基于线性投影的 PCA 编码器：hidden -> encoder_hidden_dim -> head_dim

    返回:
        combined  : [B, L, head_dim]     —— 低/高路合并的解码输出
        x_pca     : [B, L, encoder_hidden_dim] —— PCA 瓶颈表示，供解码层融合
    """

    def __init__(self, hidden_size, encoder_hidden_dim, num_attention_heads):
        super().__init__()
        self.encoder_latent_x_pca = nn.Linear(hidden_size, encoder_hidden_dim)
        head_dim = hidden_size // num_attention_heads
        self.dec_proj_l = nn.Linear(encoder_hidden_dim, head_dim)
        self.dec_proj_h = nn.Linear(encoder_hidden_dim, head_dim)
        init_orthogonal(self.dec_proj_l)
        init_orthogonal(self.dec_proj_h)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_pca = self.encoder_latent_x_pca(x)                # [B, L, encoder_hidden_dim]
        x_dec_l = self.dec_proj_l(x_pca)                    # [B, L, head_dim]
        x_dec_h = self.dec_proj_h(x_pca)                    # [B, L, head_dim]
        return x_dec_l + x_dec_h, x_pca


class LogAttention(nn.Module):
    """Log-Attention：用 log-sum-exp 替代 softmax，避免数值溢出"""

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
               mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            q, k, v : [B, H, L, head_dim]
            mask    : [1, 1, L, L] 因果掩码（可选）
        Returns:
            output   : [B, H, L, head_dim]
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        if mask is not None:
            scores = scores + mask

        # 稳定的 log-sum-exp → 归一化权重
        max_log_w = torch.max(scores, dim=-1, keepdim=True).values
        log_w = scores - max_log_w
        weights = torch.exp(log_w)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # SiLU 截断门控（保留原设计的 SiLU 激活思想）
        z = scores / self.head_dim
        gate = torch.sigmoid(z - 3.0)
        weights = weights * gate
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.matmul(weights, v)  # [B, H, L, head_dim]
        return output


class MixedAttentionLayer(nn.Module):
    """解码层：自注意力 (LogAttention) + 残差门控 + LayerNorm

    修复:
      - 原代码 residual(64) + encoder_fusion(32) 维度不匹配
      - 原代码未真正使用 attention，只是 nn.functional.layer_norm
      - 现在把 encoder_reprs_x_pca 投影回 hidden_size 再相加
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        head_dim = dim // self.num_heads

        # Q/K/V 投影（用 GQA：num_kv_heads < num_heads 时共享）
        self.num_kv_heads = min(config.num_kv_heads, self.num_heads)
        kv_dim = head_dim * self.num_kv_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.attention = LogAttention(head_dim)

        # 把 encoder 的 PCA 表示投影回 hidden_size 用于残差融合
        self.enc_proj = nn.Linear(config.encoder_hidden_dim, dim)

        # FFN + 残差门控
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """GQA: 把 kv 头复制到与 q 头数一致"""
        bsz, kv_heads, seq, head_dim = x.shape
        if kv_heads == self.num_heads:
            return x
        n_rep = self.num_heads // kv_heads
        return x[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq, head_dim) \
                                  .reshape(bsz, self.num_heads, seq, head_dim)

    def forward(self, x: Tensor,
                encoder_reprs_x_pca: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x                     : [B, L, hidden_size]
            encoder_reprs_x_pca   : [B, L, encoder_hidden_dim]  (可选)
            attn_mask             : [1, 1, L, L]                 (可选)
        """
        bsz, seq_len, dim = x.shape
        head_dim = dim // self.num_heads

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_out = self.attention(q, k, v, mask=attn_mask)  # [B, H, L, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        attn_out = self.o_proj(attn_out)

        # 残差 + LayerNorm (自注意力子层)
        out = self.norm1(x + attn_out)

        # 融合 encoder PCA 表示（投影到 hidden_size 后相加）
        if encoder_reprs_x_pca is not None:
            enc = self.enc_proj(encoder_reprs_x_pca)
            out = out + enc

        # FFN + 残差 + LayerNorm
        out = self.norm2(out + self.res_gate(out))
        return out


def create_attention_mask(seq_len: int, device=None) -> Tensor:
    """生成因果掩码 [1, 1, L, L]，上三角为 -inf"""
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


class MyMiniModel(nn.Module):
    """Encoder-Decoder：PCAEncoder + N×MixedAttentionLayer + LM Head

    架构:
        1. Token + Position Embedding
        2. PCAEncoder       —— 降维瓶颈，提取主成分
        3. N × MixedAttentionLayer —— 带 LogAttention 的解码层
        4. LayerNorm + LM Head   —— 输出词表 logits
    """

    def __init__(self, config: Optional[MiniMindConfig] = None):
        super().__init__()
        self.config = MiniMindConfig() if config is None else config
        self.hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        # 1. Token + Position Embedding
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.hidden_size,
            padding_idx=self.config.pad_token_id,
        )
        max_positions = 16384 if self.config.num_encoder_layers > 1 else 2048
        self.pe = nn.Embedding(max_positions, self.hidden_size)

        # 2. PCA Encoder
        self.pca_encoder = PCAEncoder(
            self.hidden_size, self.config.encoder_hidden_dim, num_heads,
        )

        # 3. 把 PCA 的 head_dim 输出投影回 hidden_size 给解码层
        self.enc_to_hidden = nn.Linear(head_dim, self.hidden_size)
        self.enc_norm = nn.LayerNorm(self.hidden_size)

        # 4. N 层 MixedAttentionLayer 解码层
        self.decoder_layers = nn.ModuleList([
            MixedAttentionLayer(self.config)
            for _ in range(self.config.num_decoder_layers)
        ])

        # 5. 输出 LayerNorm + LM Head（与 embed_tokens 权重共享）
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: [B, L] long
        Returns:
            logits: [B, L, vocab_size]
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.pe.num_embeddings:
            raise ValueError(
                f"seq_len {seq_len} 超过 position embedding 容量 {self.pe.num_embeddings}"
            )

        # 1. Embedding + Position
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        h = self.embed_tokens(input_ids) + self.pe(positions)  # [B, L, hidden]

        # 2. PCA Encoder: h -> combined(head_dim) + x_pca(encoder_hidden_dim)
        combined, x_pca = self.pca_encoder(h)

        # 3. 投影回 hidden_size
        dec_in = self.enc_norm(self.enc_to_hidden(combined) + h)

        # 4. 因果掩码
        attn_mask = create_attention_mask(seq_len, device=input_ids.device)

        # 5. 解码层
        for layer in self.decoder_layers:
            dec_in = layer(dec_in, encoder_reprs_x_pca=x_pca, attn_mask=attn_mask)

        out = self.layer_norm(dec_in)
        logits = self.lm_head(out)  # [B, L, vocab_size]
        return logits


def test_basic():
    """基本测试：验证模型可创建、前向可跑、所有参数都参与损失（DDP 安全）"""
    config = MiniMindConfig(
        hidden_size=64, encoder_hidden_dim=32,
        num_encoder_layers=2, num_decoder_layers=2,
        num_attention_heads=8, vocab_size=100,
    )
    model = MyMiniModel(config)
    print("Model created. params:", sum(p.numel() for p in model.parameters()))

    x = torch.randint(0, config.vocab_size, (2, 16), dtype=torch.long)
    logits = model(x)
    print(f"Forward OK. logits: {tuple(logits.shape)}")

    # 检查是否有未参与损失的参数（DDP 兼容性）
    loss = logits.sum()
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"Unused params (DDP-unsafe): {unused}")
    assert not unused, f"DDP will fail: {unused}"
    print("All params participate in loss — DDP safe ✓")


if __name__ == "__main__":
    test_basic()
