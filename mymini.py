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
    vocab_size: int = 6400           # 词表大小，需与 tokenizer 一致
    use_moe: bool = False            # 是否启用 MoE（保留字段以兼容 lm_checkpoint）
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


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

        self.config = MiniMindConfig() if config is None else config
        self.hidden_size = self.config.hidden_size
        encoder_hidden_dim = getattr(self.config, 'encoder_hidden_dim', 512)

        # Token embedding: [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.hidden_size,
                                         padding_idx=getattr(self.config, 'pad_token_id', 0))

        # Position embedding (kept for compatibility with trainer's pe.embedding_dim reference)
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        max_positions = 16384 if self.config.num_encoder_layers > 1 else 2048
        self.pe = nn.Embedding(max_positions, head_dim)

        # PCAEncoder: latent projection for encoder state x -> x'
        self.encoder_latent_x_pca = nn.Linear(self.hidden_size, encoder_hidden_dim)
        # Low / High latent decoder projections: x' -> head_dim
        self.dec_proj_l = nn.Linear(encoder_hidden_dim, head_dim)
        self.dec_proj_h = nn.Linear(encoder_hidden_dim, head_dim)
        # Project combined head_dim back to hidden_size for the LM head
        self.dec_proj_out = nn.Linear(head_dim, self.hidden_size)

        # Output normalization + LM head (logits over vocab)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.config.vocab_size, bias=False)

        # Tie weights to reduce parameter count and stabilize training
        self.embed_tokens.weight = self.lm_head.weight

        # Initialize decoder projections with orthogonal constraints
        init_orthogonal(self.dec_proj_l)
        init_orthogonal(self.dec_proj_h)

    def forward(self, input_ids: Tensor):
        """
        Args:
            input_ids: [B, L] long tensor of token ids
        Returns:
            logits: [B, L, vocab_size] tensor of vocabulary logits
        """
        bsz, seq_len = input_ids.shape

        # 1. Token embedding -> hidden states
        h = self.embed_tokens(input_ids)  # [B, L, hidden_size]

        # 2. PCA projection to latent state x'
        x_pca = self.encoder_latent_x_pca(h)  # [B, L, encoder_hidden_dim]

        # 3. Low / High latent decoder projections
        dec_l = self.dec_proj_l(x_pca)  # [B, L, head_dim]
        dec_h = self.dec_proj_h(x_pca)  # [B, L, head_dim]
        combined = dec_l + dec_h  # [B, L, head_dim]

        # 4. Project back to hidden_size + residual + norm
        out = self.layer_norm(self.dec_proj_out(combined) + h)  # [B, L, hidden_size]

        # 5. LM head -> logits over vocab
        logits = self.lm_head(out)  # [B, L, vocab_size]
        return logits


def test_basic():
    """Basic test to verify code works without errors"""
    config = MiniMindConfig(hidden_size=1024, num_encoder_layers=4, num_attention_heads=8)

    model = MyMiniModel(config)
    print("Model created successfully")

    x = torch.randint(0, config.vocab_size, (2, 50), dtype=torch.long)
    with torch.no_grad():
        y = model(x)
    print(f"Forward pass successful. Output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    test_basic()
    