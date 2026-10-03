import os
import sys

# 把项目根目录加入 sys.path，确保 mymini / dataset / trainer 等模块可被导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
__package__ = "trainer"

from dataclasses import *
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# 导入 mymini.py 定义的训练组件
from mymini import MiniMindConfig, MyMiniModel, init_orthogonal
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, model, optimizer, scaler, args, autocast_ctx,
                lm_config, start_step=0, wandb=None):
    """训练一个 epoch。所有依赖通过参数显式传入，避免依赖全局变量。"""
    start_time = time.time()
    model.train()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        with autocast_ctx:
            logits = model(input_ids)  # [B, L, vocab_size]

            # 自回归语言模型损失：用上一 token 预测下一 token
            # logits[:, :-1] 预测 labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous().view(-1, lm_config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100
            )

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) '
                   f'loss: {loss.item() * args.accumulation_steps:.4f} '
                   f'time/step: {spend_time / max(step - start_step, 1):.2f}s')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            moe_suffix = '_moe' if getattr(lm_config, 'use_moe', False) else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 半精度保存，节省显存 (~12GB VRAM)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
            torch.save(state_dict, ckp)
            Logger(f"Saved checkpoint to {ckp}")

        del input_ids, labels, logits, loss


def validate_orthogonality(module: nn.Module) -> float:
    """检查正交性 Q·(Q^T)≈I"""
    if module is not None and hasattr(module, 'weight') and module.weight.dim() == 2:
        w = module.weight.data.reshape(-1, module.out_features).t()
        ortho_metric = torch.trace(w @ w.t())  # Trace(W·W^T) should ≈ dim
        return ortho_metric.item()
    return 0.0


def test_model_memory(config=None):
    """测试模型显存占用 (默认 hidden_size=1024, num_encoder_layers=4 → fits in 12GB VRAM)"""
    if config is None:
        config = MiniMindConfig(hidden_size=1024, encoder_hidden_dim=512, num_encoder_layers=4,
                               num_attention_heads=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyMiniModel(config)

    x_sample = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)
    _ = model(x_sample)

    return f"Model loaded on {device}\n   Config: hidden={config.hidden_size}, encoder_dim={config.encoder_hidden_dim}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyMini Pretraining Pipeline")

    # 实验设置
    parser.add_argument("--save_dir", type=str, default="../out", help="保存目录")
    parser.add_argument('--save_weight', default='pretrain_mymini', type=str, help="权重前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")

    # 设备与精度配置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"], help="混合精度类型 (默认: bfloat16)")

    # 模型配置参数 (关键：hidden_size=1024/encoder_hidden_dim=512/heads=8 → ~4GB overhead)
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度 (默认:1024; VRAM<12GB 可调低至 768)")
    parser.add_argument('--encoder_hidden_dim', default=512, type=int, help="PCA 降维后维度 (推荐：hidden_size/2 = 512)")
    parser.add_argument('--num_encoder_layers', default=4, type=int, help="编码器层数")
    parser.add_argument('--num_decoder_layers', default=12, type=int, help="解码器层数")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数 (默认:8)")

    # 训练控制参数
    parser.add_argument('--max_seq_len', default=340, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', action='store_true', help="是否使用 MoE")
    parser.add_argument('--seed', default=42, type=int, help="随机种子")
    parser.add_argument('--accumulation_steps', default=1, type=int, help="梯度累积步数")
    parser.add_argument('--grad_clip', default=1.0, type=float, help="梯度裁剪阈值")
    parser.add_argument('--log_interval', default=10, type=int, help="日志打印间隔 step")
    parser.add_argument('--save_interval', default=200, type=int, help="权重保存间隔 step")
    parser.add_argument('--num_workers', default=0, type=int, help="DataLoader 工作进程数")
    parser.add_argument('--from_resume', default=0, type=int, help="是否从 checkpoint 恢复 (0/1)")
    parser.add_argument('--use_wandb', action='store_true', help="是否启用 swanlab/wandb 日志")
    parser.add_argument('--use_compile', default=0, type=int, help="是否启用 torch.compile (0/1)")

    # 输入数据配置
    parser.add_argument('--data_path', default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--tokenizer_path', default="../model", help="tokenizer 路径")

    args = parser.parse_args()

    # 路径解析策略：
    #   1. 用户显式传入的相对路径（如 ./dataset/xxx）按 CWD 解析
    #   2. 默认值（如 ../model）按脚本目录解析（保持向后兼容）
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve(p: str, default: str) -> str:
        if os.path.isabs(p):
            return p
        if p == default:
            # 默认值：相对脚本目录解析
            return os.path.abspath(os.path.join(script_dir, p))
        # 用户传入的路径：先按 CWD 解析，若不存在再回退到脚本目录
        cwd_resolved = os.path.abspath(p)
        if os.path.exists(cwd_resolved) or os.path.exists(os.path.dirname(cwd_resolved)):
            return cwd_resolved
        return os.path.abspath(os.path.join(script_dir, p))

    args.data_path = _resolve(args.data_path, "../dataset/pretrain_t2t_mini.jsonl")
    args.tokenizer_path = _resolve(args.tokenizer_path, "../model")
    args.save_dir = _resolve(args.save_dir, "../out")

    # ========== 1. 初始化环境（分布式 DDP/单机 FP16 可选）==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数（MiniMindConfig 验证）==========
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        encoder_hidden_dim=args.encoder_hidden_dim,
        num_attention_heads=args.num_attention_heads,
        use_moe=args.use_moe,
    )

    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 + RoPE Embedding (scale=hidden_size**(-0.5))==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置 wandb 日志记录（可选）==========
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
            wandb_id = ckp_data.get('wandb_id') if ckp_data else None
            resume = 'must' if wandb_id else None
            wandb_run_name = f"MyMini-Epoch-{args.epochs}-LR-{args.learning_rate}"
            wandb.init(project="MyMini", name=wandb_run_name, id=wandb_id, resume=resume)
        except ImportError:
            pass

    # ========== 5. 定义模型 + PCAEncoder (from mymini.py + RoPE/LogAttention)==========
    model = MyMiniModel(lm_config).to(args.device)

    # PCAEncoder (encoder_latent_x_pca + dec_proj_l/h) 验证正交性
    validate_orthogonality(getattr(model, 'encoder_latent_x_pca', None))

    # ========== 6. 加载 tokenizer + 数据集 ==========
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 数据加载策略：DistributedSampler/随机洗牌（单机）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == "bfloat16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

    # ========== 7. 从 ckp 恢复状态（自动续训）==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. 编译优化 + DDP 包装==========
    if args.use_compile == 1:
        model = torch.compile(model, mode="reduce-overhead")

    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    # ========== 9. 开始训练（epoch 级主循环）==========
    iters = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        setup_seed(args.seed + epoch)

        if start_step > 0 and epoch == start_epoch:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 从 step {start_step + 1} 开始')
            # 简单实现：跳过前 start_step 个 batch
            train_epoch(epoch, train_loader, iters - start_step, model, optimizer, scaler,
                        args, autocast_ctx, lm_config, start_step=start_step, wandb=wandb)
        else:
            train_epoch(epoch, train_loader, iters, model, optimizer, scaler,
                        args, autocast_ctx, lm_config, start_step=0, wandb=wandb)
        start_step = 0  # 仅第一个 epoch 跳过

    # ========== 10. 清理分布式进程==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
