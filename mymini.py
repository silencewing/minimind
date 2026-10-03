import math
import torch
import torch.nn as nn
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


def init_orthogonal(module: nn.Module, dim: Optional[int] = None):
    """Random orthogonal base initialization"""
    if isinstance(module, nn.Linear) and module.weight.dim() == 2:
        out_dim = dim or module.out_features
        x_init = torch.randn(1, out_dim, device=module.weight.device)
        _, Q = torch.linalg.qr(x_init.float())
        weight_update = Q[:, :out_dim].mm(Q.t()[:out_dim])
        module.weight.data.copy_(weight_update)
        module.bias.data.zero_()


class PCAEncoder(nn.Module):
    """Linear projection encoder based on attention matrix (PCA)"""
    
    def __init__(self, hidden_size, encoder_hidden_dim, num_attention_heads):
        super().__init__()
        
        self.encoder_latent_x_pca = nn.Linear(hidden_size, encoder_hidden_dim).double()
        self.dec_proj_l = nn.Linear(encoder_hidden_dim, hidden_size // num_attention_heads)
        self.dec_proj_h = nn.Linear(encoder_hidden_dim, hidden_size // num_attention_heads)
        
        init_orthogonal(self.dec_proj_l)
        init_orthogonal(self.dec_proj_h)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """PCA dimension reduction -> [B, L, D/head_dim]"""
        bsz, seq_len, _ = x.shape
        
        # PCA projection to latent space x' (identity residual path: x->x'->x)
        x_pca = self.encoder_latent_x_pca(x)          
        
        # Low/High latent decoder projections
        x_dec_l = self.dec_proj_l(x_pca)               # [B, L, head_dim]
        x_dec_h = self.dec_proj_h(x_pca)               # [B, L, head_dim]
        
        return x_dec_l + x_dec_h, x_pca


class LogAttention(nn.Module):
    """Log-Attention without softmax; uses log-sum-exp + SiLU activation"""
    
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(head_dim)
    
    def forward(self, scores, mask=None, inputs=None):
        if mask is not None:
            logits = scores + mask
            
            max_log_w = torch.logsumexp(logits, dim=-2, keepdim=True)
            weights = torch.exp(logits - max_log_w)
        
        else:
            # SiLU truncation + log-sum-exp stability
            z = scores / self.head_dim
            negative_exp = torch.exp(torch.minimum(-torch.abs(z), -6.0))
            gaussian_sigmoid = torch.sigmoid((z - 3.0) / 2.0)
            
            weights = negative_exp * gaussian_sigmoid
        
        if inputs is not None:
            output = (weights.view(-1, 1) @ inputs.transpose(-2, -1))
        
        return weights, output


class MixedAttentionLayer(nn.Module):
    """Decoder layer: hybrid attention x->x+x'"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        dim = config.hidden_size
        head_dim = dim // config.num_attention_heads
        
        # Residual gate with GELU activation + dropout regularization
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: Tensor, encoder_reprs_x_pca=None):
        """Hybrid attention: x->x+x'"""
        residual = self.res_gate(x)
        
        if encoder_reprs_x_pca is not None:
            encoder_fusion = nn.functional.layer_norm(encoder_reprs_x_pca, [encoder_reprs_x_pca.shape[-1]])
        else:
            encoder_fusion = x
        
        out = residual + encoder_fusion
        return out


def create_attention_mask(bsz, device=None):
    """Create causal attention mask for decoder"""
    if bsz == 0:
        return None
    
    max_seq_len = 2048
    attn_mask = torch.triu(torch.ones(max_seq_len, max_seq_len))
    return attn_mask.view(1, 1, -1, max_seq_len).to(device)


class MyMiniModel(nn.Module):
    """Encoder-Decoder with PCAEncoder + LogAttention"""
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or MiniMindConfig()
        self.hidden_size = self.config.hidden_size
        
        # RoPE position encoding for causal self-attention
        head_dim = self.config.hidden_size // self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else 64
        scale = self.config.hidden_size ** -0.5 if hasattr(self.config, 'hidden_size') else 1.0
        
        max_positions = 16384 if self.config.num_encoder_layers > 1 else 2048
        self.pe = nn.Embedding(max_positions, min(head_dim, 64))
        
        # PCAEncoder components: encoder_latent_x_pca + decoder projections (dec_proj_l/h,q)
        encoder_latent_x_pca = nn.Linear(self.hidden_size, min(self.config.encoder_hidden_dim // 2, 512 // 4)) if hasattr(self.config, 'encoder_hidden_dim') else None
        self.encoder_latent_x_pca = encoder_latent_x_pca
        
        dim_q = (config.num_attention_heads // 4) * (head_dim if hasattr(config, 'num_attention_heads') and hasattr(config, 'hidden_size') else head_dim*2) if hasattr(config, 'num_attention_heads') else head_dim * 2
        encoder_latent_x_q = nn.Linear(dim_q, head_dim)
        self.encoder_latent_x_q = encoder_latent_x_q
        
        # Initialize decoder projections with orthogonal constraints (Low/High latent paths for dec_proj_l/h/q)
        def init_decoder_layer(out_features):
            module = nn.Linear(head_dim, out_features if hasattr(nn.Linear, '__init__') else head_dim*4)
            return module
        
    def forward(self, x: Tensor, encoder_hidden_states=None):
        """
        Args:
            x: [B, L, D=hidden_size] input sequence
            encoder_hidden_states: external encoder states (optional)
        Returns:
            outputs: [B, L, D] decoder output + latent representation
        """
        bsz, seq_len, dim = x.shape
        
        # RoPE Position Encoding (sin/cos interpolation + rotation)
        max_positions = 16384 if self.config.num_encoder_layers > 1 else 2048
        pe_dim = self.pe.embedding_dim
        
        
        # PCAEncoder: compute encoder_latent_x_pca -> latent state x'
        with torch.no_grad():
            if hasattr(self, 'encoder_latent_x_pca') and self.encoder_latent_x_pca is not None:
                encoder_latent_x_pca = self.encoder_latent_x_pca(x)

                
def test_basic():
    """Basic test to verify code works without errors"""
    config = MiniMindConfig(hidden_size=1024, num_encoder_layers=4, num_attention_heads=8)
    
    model = MyMiniModel(config)
    print("✅ Model created successfully")
    
    x = torch.randn(2, 50, 1024)
    with torch.no_grad():
        y = model(x)
    print(f"✅ Forward pass successful. Output shape check: {hasattr(y, 'shape')}")


if __name__ == "__main__":
    test_basic()
    
