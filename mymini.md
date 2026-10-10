# MyMini 模型架构

> 与 [mymini.py](mymini.py) 同步更新。每次修改架构时同步本文档。

## 总体结构

```mermaid
flowchart TD
    IDS["input_ids<br/>[B, L]"] --> EMB["Token Embedding<br/>nn.Embedding(6400 → 1024)<br/>pad_idx=0 · 与 lm_head 共享权重"]
    EMB --> DROP["Dropout"] --> H["h [B, L, 1024]"]

    subgraph ENC["🔷 编码器 PCAEncoder：x → x′ → x（×4 个 BottleneckBlock 串联）"]
        direction TB
        B1["z = SiLU(W·x)<br/>1024 → 512 瓶颈降维<br/>W 正交初始化 W·Wᵀ=I"]
        B2["x_rec = g·Wᵀ·z<br/>512 → 1024 转置重构<br/>g = 2·√(hidden/latent) 重构增益"]
        RL["recon_loss += MSE(x_rec, stopgrad(x))<br/>强迫瓶颈保留输入信息"]
        B1 --> B2
        B2 -.->|"逐块累加"| RL
        B2 -->|"输出喂给下一块"| B1
    end

    H --> ENC
    ENC -->|"x_rec"| FUSE["解码器入口<br/>enc_norm = RMSNorm(h + x_rec)"]
    H -->|"h"| FUSE

    subgraph DEC["🔶 解码器 DecoderLayer × 12（Pre-LN 结构）"]
        direction TB
        N1["RMSNorm"] --> ATTN
        subgraph ATTN["GQA LogAttention（不使用 softmax）"]
            direction TB
            QKV["Q: 1024→1024（8 头）<br/>K/V: 1024→512（4 头 GQA）<br/>Q/K/V/O 全部按头正交初始化"]
            CACHE["KV Cache（推理）<br/>缓存 RoPE 之后的 K/V<br/>past_key_values 逐层传递"]
            ROPE["RoPE 旋转位置编码 θ=1e6<br/>按绝对位置切片 cos/sin<br/>repeat_kv ×2 扩展 KV 头"]
            LOG["logits = QKᵀ/√128<br/>① clamp ±4（log 域截断，权重比 ≤ e⁸）<br/>② 因果掩码置 -inf<br/>③ log-sum-exp 归一化（fp32）"]
            QKV --> ROPE --> CACHE --> LOG
        end
        ATTN --> R1{{"+ 残差"}} --> N2["RMSNorm"]
        N2 --> FFN["SwiGLU FFN<br/>down( SiLU(gate(x)) ⊙ up(x) )<br/>1024 → 2752 → 1024<br/>std=0.02 小尺度初始化"]
        FFN --> R2{{"+ 残差"}}
    end

    FUSE --> DEC --> NORM["最终 RMSNorm"]
    NORM --> HEAD["lm_head（与 embedding 共享权重）<br/>1024 → 6400"]
    HEAD --> LOGITS["logits [B, L, 6400]"]

    LOGITS --> LOSS["📉 总损失 = CE(shift 预测) + 0.1 × Σ recon_loss"]
    RL -.->|"aux_loss"| LOSS

    style ENC fill:#e8f4fd,stroke:#2196F3
    style DEC fill:#fff3e0,stroke:#FF9800
    style LOSS fill:#ffebee,stroke:#f44336
```

## 推理路径（KV cache + generate）

```mermaid
sequenceDiagram
    participant U as 调用方
    participant M as MyMiniModel
    participant E as PCAEncoder
    participant D as DecoderLayer ×12

    Note over U,D: 首步：完整 prompt（建缓存）
    U->>M: forward(prompt_ids, use_cache=True)
    M->>E: 编码 prompt 全部 token
    M->>D: 逐层前向，缓存 RoPE 后的 K/V
    D-->>M: past_key_values (12 层)
    M-->>U: logits[:, -1] + past

    Note over U,D: 增量步：每步仅 1 个新 token
    loop max_new_tokens
        U->>M: 采样/贪心得 next_id
        U->>M: forward(next_id, past_key_values=past)
        M->>E: 只编码新 token
        M->>D: K/V 拼接缓存后注意力<br/>RoPE 按绝对位置 past_len 切片<br/>掩码 tril(diagonal=past_len)
        D-->>M: 更新后的 past
        M-->>U: logits[:, -1]
    end
```

## 训练配套

| 项目 | 配置 |
|---|---|
| 总损失 | `CE(shift) + recon_loss_coef(0.1) × Σ各块重构MSE` |
| 优化器 | AdamW 三参数组：embedding 组 lr×0.3 无衰减 / 矩阵组 wd=0.01 / 1D 参数无衰减 |
| 显存控制 | 梯度检查点默认开启（use_cache 时自动禁用）、bf16 autocast |
| 监控 | 日志含 `emb_g`（共享 embedding 梯度范数） |
| 显存实测 | RTX 3060 12GB：bs=16×accum2, seq=340 约 4.6GB |

## 关键设计约束

1. **编码器非死支路**：重构增益 `g = 2·√(hidden/latent)` 补偿 SiLU 斜率(0.5)与转置投影能量损失(latent/hidden)，初始 ‖x_rec‖/‖x‖ ≈ 1
2. **无 softmax**：log 域 clamp ±4 → 权重比 ≤ e⁸，log-sum-exp 归一化在 fp32 下进行
3. **多头正交**：Q/K/V/O 按头切片正交初始化（测试断言 WᵢWᵢᵀ ≈ I）
4. **KV cache 一致性**：缓存存 RoPE 之后的 K/V，数值与无缓存前向 max diff ≈ 3e-7
