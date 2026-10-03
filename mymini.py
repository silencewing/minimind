import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List


@dataclass
class MyMiniConfig:
    """模型配置类"""
    hidden_size: int = 1024          # 隐藏层维度（可调整）
    encoder_hidden_dim: int = 512    # 编码器降维后维度（PCA 投影）
    num_attention_heads: int = 8     # 注意力头数
    num_kv_heads: int = 4            # KV 头数（分组查询注意力）
    num_encoder_layers: int = 4      # 编码器层数
    num_decoder_layers: int = 12     # 解码器层数


# ==================== 正交初始化 ====================

def init_orthogonal(module: nn.Module, dim: Optional[int] = None):
    """随机正交基初始化（Gram-Schmidt 近似）"""
    if isinstance(module, nn.Linear) and module.weight.dim() == 2:
        out_dim = dim or module.out_features
        
        # QR 分解生成 Q 矩阵
        x_init = torch.randn(1, out_dim, device=module.weight.device)
        _, Q = torch.linalg.qr(x_init.float())
        weight_update = Q[:, :out_dim].mm(Q.t()[:, :out_dim])  
        module.weight.data.copy_(weight_update)

        # 零偏置
        module.bias.data.zero_()


class PCAEncoder(nn.Module):
    """基于自注意力矩阵的线性投影编码器（PCA）"""
    
    def __init__(self, hidden_size: int, encoder_hidden_dim: int, num_attention_heads: int):
        super().__init__()
        
        self.encoder_latent_x_pca = nn.Linear(hidden_size, encoder_hidden_dim)
        self.dec_proj_l = nn.Linear(encoder_hidden_dim, hidden_size // num_attention_heads)
        self.dec_proj_h = nn.Linear(encoder_hidden_dim, hidden_size // num_attention_heads)
        
        init_orthogonal(self.dec_proj_l)
        init_orthogonal(self.dec_proj_h)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """自注意力 PCA 降维 -> [B, L, D/num_heads]"""
        bsz, seq_len, _ = x.shape
        
        # PCA 降维到潜空间 x'（identity residual path）
        x_pca = self.encoder_latent_x_pca(x)           # [B, L, K=512]
        
        # Low/High latent decoder projections (PCA decode back)
        x_dec_l = self.dec_proj_l(x_pca)               # Low dimension
        x_dec_h = self.dec_proj_h(x_pca)               # High dimension
        
        return x_dec_l + x_dec_h, x_pca


class LogAttention(nn.Module):
    """
    Log-Attention 模块：不使用 softmax，而是采用 log-sum-exp + sigmoid 截断机制
    """
    
    def __init__(self, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, scores: Tensor, mask: Optional[Tensor] = None, 
                inputs: Optional[Tensor] = None):
        """
        Args:
            scores: [B, L_Q, L_K] Q·K^T 原始分数
        
        Returns:
            attn_weights: 注意力权重
            outputs: 加权求和结果
        """
        if mask is not None:
            logits = scores + mask
            
            # Stability: subtract max to avoid underflow
            max_log_w = torch.logsumexp(logits, dim=-2, keepdim=True)
            weights = torch.exp(logits - max_log_w)
        
        else:
            # Log-Attention: sigmoid truncation + log-sum-exp stability
            # z ∈ [-10, +10] ⇒ sigmoid(z/5) ≈ [0.3, 0.7]
            scores_scaled = scores / self.head_dim
            truncated_scores = torch.clamp(scores_scaled, min=-6.0, max=6.0)
            
            # weights = exp(max(-z,-6)) * sigmoid((z-3)/2) - Gaussian tail approximation
            negative_exp = torch.exp(torch.minimum(-truncated_scores, -6.0))
            gaussian_sigmoid = torch.sigmoid((truncated_scores - 3.0) / 2.0)
            
            weights = negative_exp * gaussian_sigmoid
        
        if inputs is not None:
            output = (weights.unsqueeze(1) if weights.dim() > 1 else weights.view(-1, 1)) @ inputs.transpose(-2, -1)
        
        return weights, output


class MixedAttentionLayer(nn.Module):
    """
    Decoder 层：混合注意力 x -> x + x'
    
    类型：Self-Attention (原始序列) + Cross-Encoder Attention (编码表示 x')
    """
    
    def __init__(self, config: MyMiniConfig):
        super().__init__()
        
        self.config = config
        
        dim = config.hidden_size
        head_dim = dim // config.num_attention_heads
        
        # Residual gate with GELU (dropout regularization)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # LayerNorm for residual connection
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: Tensor, encoder_reprs_x_pca: Optional[Tensor] = None):
        """混合注意力"""
        bsz, seq_len, _ = x.shape
        
        # 1. Residual gate path (identity residual: x -> res_gate(x))
        residual = self.res_gate(x) + x
        
        # 2. Encoder representation fusion (if available) or identity fallback
        if encoder_reprs_x_pca is not None:
            encoder_fusion = F.layer_norm(encoder_reprs_x_pca, (encoder_reprs_x_pca.shape[-1],))
        else:
            encoder_fusion = x
        
        # 3. Residual connection + LayerNorm
        out = self.layer_norm(residual) + encoder_fusion
        
        return out


def create_attention_mask(bsz: int, device=None) -> Optional[Tensor]:
    """创建三角掩码用于 decoder (causal mask)"""
    if bsz == 0:
        return None
    
    # Create upper triangular mask: 1 where row > col (valid positions for decoder)
    max_seq_len = 2048
    attn_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.float32), diagonal=1).view(max_seq_len, -1) < 0
    return attn_mask.view(1, 1, -1, max_seq_len).to(device)


class MyMiniModel(nn.Module):
    """基于 PCAEncoder+LogAttention 的 Encoder-Decoder"""
    
    def __init__(self, config: Optional[MyMiniConfig] = None):
        super().__init__()
        
        self.config = config or MyMiniConfig()
        self.hidden_size = self.config.hidden_size
        
        # RoPE embeddings for position encoding
        if self.config.num_encoder_layers > 1:
            max_positions = 16384
        else:
            max_positions = 2048
        
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        scale = self.config.hidden_size ** -0.5
        
        # PE embedding for rotary embeddings
        inv_freq_exponent = torch.arange(head_dim, dtype=torch.float32) * scale
        self.inv_freq = torch.exp(torch.log(inv_freq_exponent + 1.0))
        
        self.pe = nn.Embedding(max_positions, head_dim, dtype=torch.float32)

    def apply_rope(self, x: Tensor, positions: Optional[Tensor] = None):
        """RoPE 位置编码：sin/cos rotation"""
        if positions is None:
            max_positions = 16384
            device = next(p.device for p in self.parameters())
            
            # inv_freq: 1 / ((max_positions ** (k * scale)) for k in [0, ..., head_dim-1]
            inv_freq = torch.exp(-torch.arange(head_dim, dtype=torch.float32, device=device) * math.log(max_positions) * scale).to(device)
            
            batch_size, seq_len, _ = x.shape
            
            positions = torch.arange(seq_len, device=device).unsqueeze(-1)  # [seq_len]
            freqs = positions @ inv_freq.unsqueeze(-1)  # [seq_len, head_dim]
            
            # Split into even/odd half dimensions for rotary embedding
            angle_half = head_dim // 2
            cos_cached = torch.cos(freqs.to(x.dtype)).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
            sin_cached = torch.sin(freqs.to(x.dtype)).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
            
            # Apply rotation: x' = cos(x) - rot(theta) * sin(x), where rot(theta) is cross-half rotation matrix
            x_even = x[:, :, :angle_half]
            x_odd = x[:, :, angle_half:]
            
            return (x_even * cos_cached + x_odd * sin_cached).to(dtype=x.dtype)

    @property
    def n_layers(self) -> int:
        return self.config.num_decoder_layers
    
    def forward(
        self, 
        x: Tensor, 
        encoder_hidden_states: Optional[Tensor] = None,
    ):
        """
        Args:
            x: [B, L, D=hidden_size] input sequence
            encoder_hidden_states: PCAEncoder 输出 x'（降维表示，可选）
        
        Returns:
            outputs: [B, L, D] 解码结果 + encoder_reprs_x_pca 潜空间表示
        """
        bsz, seq_len, dim = x.shape
        
        # ==================== RoPE Position Encoding ====================
        if self.config.num_encoder_layers > 1:
            max_positions = 16384
            pe_dim = self.pe.embedding_dim
        else:
            max_positions = 2048
            pe_dim = self.pe.embedding_dim
        
        # Create positional embeddings dynamically
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        # ==================== PCAEncoder Latent Representation ====================
        encoder_latent_x_pca = self.encoder_latent_x_pca(x)
        encoder_reprs_x_pca = F.layer_norm(encoder_latent_x_pca, [encoder_latent_x_pca.shape[-1]])
        
        if encoder_hidden_states is not None:
            # External encoder states override internal calculation
            encoder_reprs_x_pca = encoder_hidden_states
        
        # ==================== Decoder Layers ====================
        decoder_layers = nn.ModuleList() if self.config.num_decoder_layers > 0 else None
        
        output = x
        if decoder_layers is not None:
            for layer in decoder_layers:
                output, past_key_value = layer(
                    output,
                    encoder_reprs_x_pca=encoder_reprs_x_pca if not hasattr(self, '_skip_encoder_layer') else None
                )
        
        return output


def validate_orthogonality(module: nn.Module) -> float:
    """检查正交性：Q·W^T ≈ I"""
    w = module.weight.data.reshape(-1, module.out_features).t()
    ortho_metric = torch.trace(w @ w.t())  # Trace(W·W^T) should ≈ dim
    
    return ortho_metric.item()


def test_model_memory(config: MyMiniConfig):
    """测试模型显存占用"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyMiniModel(MyMiniConfig(hidden_size=config.hidden_size))
    
    # Simple forward pass memory analysis
    x_sample = torch.randn(1, 32, config.hidden_size)
    _ = model(x_sample)
    
    return f"Model loaded on {device}"


if __name__ == "__main__":
    # Simple test
    config = MyMiniConfig()
    
    print("=" * 60)
    print(f"MyMini Config: ")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Encoder hidden dim: {config.encoder_hidden_dim}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Head dim: {config.hidden_size // config.num_attention_heads}")
    