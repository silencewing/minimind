import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from mymini import MyMiniConfig as MiniMindConfig, init_orthogonal, validate_orthogonality
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练轮次"""
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids)
            loss = res.loss if hasattr(res, 'loss') else torch.nn.functional.cross_entropy(
                res.float().transpose(-1, -2).contiguous(), labels.reshape(-1)
            )
            aux_loss = res.aux_loss if hasattr(res, 'aux_loss') else 0.0
            loss = loss / args.accumulation_steps + aux_loss / 4

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = (loss.item() * args.accumulation_steps if hasattr(loss, 'item') else 0)
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {lr:.8f}')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if hasattr(lm_config, 'use_moe') and lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            
            # 保存权重（使用 half precision）
            state_dict = {k: v.half().cpu() for k, v in dict(raw_model.named_parameters()).items()}
            torch.save(state_dict, ckp)
            Logger(f"Saved checkpoint to {ckp}")

            model.train()

        del input_ids, labels, res, loss


def validate_orthogonality(module: nn.Module) -> float:
    """检查正交性"""
    w = module.weight.data.reshape(-1, module.out_features).t() if hasattr(module, 'weight') and module.weight.dim() == 2 else torch.eye(6)
    ortho_metric = torch.trace(w @ w.t())
    return ortho_metric.item()


def test_model_memory(config: MiniMindConfig):
    """测试模型显存占用"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyMiniModel(MyMiniConfig(hidden_size=config.hidden_size))
    x_sample = torch.randn(1, 32, config.hidden_size)
    _ = model(x_sample)
    return f"Model loaded on {device}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyMini Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")

    # 模型配置参数
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度 (默认:1024)")
    parser.add_argument('--num_hidden_layers', default=4, type=int, help="编码器层数")
    parser.add_argument('--encoder_hidden_dim', default=512, type=int, help="PCA 降维后维度")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数")
    
    # 训练控制参数
    parser.add_argument('--max_seq_len', default=340, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE")
    parser.add_argument('--from_weight', default='none', type=str, help="预训练权重路径")
    parser.add_argument('--data_path', type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="数据路径")

    args = parser.parse_args()

    # 初始化环境
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # 配置目录、模型参数
    os.makedirs(args.save_dir, exist_ok=True)
    
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_hidden_layers,
        encoder_hidden_dim=args.encoder_hidden_dim,
        num_attention_heads=args.num_attention_heads
    )

    # 初始化 MyMiniModel (RoPE + PCAEncoder)
    head_dim = args.hidden_size // args.num_attention_heads
    scale = args.hidden_size ** -0.5
    max_seq_len = 16384

    model = torch.nn.Module()
    model.eval()