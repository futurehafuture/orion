#!/usr/bin/env python3
"""ORION训练脚本"""

import argparse
import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from orion.config import OrionConfig
from orion.models import OrionSystem
from orion.training import OrionTrainer
from orion.data import create_dataloader
from orion.utils import setup_logging, create_experiment_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train ORION autonomous driving model")
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (YAML or JSON)')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='data/',
                       help='Dataset path')
    parser.add_argument('--dataset-type', type=str, default='toy',
                       choices=['toy', 'carla', 'nuscenes'],
                       help='Dataset type')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps')
    
    # 模型参数
    parser.add_argument('--vision-backbone', type=str, default='resnet18',
                       help='Vision backbone')
    parser.add_argument('--llm-type', type=str, default='toy',
                       choices=['toy', 'gpt2', 'llama'],
                       help='LLM type')
    parser.add_argument('--token-dim', type=int, default=512,
                       help='Token dimension')
    parser.add_argument('--traj-len', type=int, default=20,
                       help='Trajectory length')
    
    # 设备和优化
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--compile-model', action='store_true',
                       help='Compile model with torch.compile')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # 输出和日志
    parser.add_argument('--output-dir', type=str, default='outputs/',
                       help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='orion_experiment',
                       help='Experiment name')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    parser.add_argument('--save-every', type=int, default=5000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval-every', type=int, default=1000,
                       help='Evaluate every N steps')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Load pretrained weights')
    
    # 验证和测试
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic algorithms')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (no actual training)')
    
    return parser.parse_args()


def setup_config(args) -> OrionConfig:
    """设置配置"""
    if args.config:
        config = OrionConfig.from_yaml(args.config)
    else:
        config = OrionConfig()
    
    # 覆盖命令行参数
    if args.data_path:
        config.data_path = args.data_path
    if args.dataset_type:
        config.dataset_type = args.dataset_type
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # 训练参数
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.warmup_steps = args.warmup_steps
    config.training.gradient_clip_norm = args.gradient_clip
    config.training.save_every = args.save_every
    config.training.eval_every = args.eval_every
    
    # 模型参数
    config.vision.backbone = args.vision_backbone
    config.llm.model_type = args.llm_type
    config.qt_former.token_dim = args.token_dim
    config.planner.traj_len = args.traj_len
    
    # 设备设置
    config.device = args.device
    config.mixed_precision = args.mixed_precision
    config.compile_model = args.compile_model
    
    # 其他设置
    config.seed = args.seed
    config.num_workers = args.num_workers
    
    return config


def setup_environment(config: OrionConfig, args):
    """设置训练环境"""
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # 设置确定性算法
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config.output_dir, "config.yaml")
    config.to_yaml(config_path)


def create_dataloaders(config: OrionConfig, args):
    """创建数据加载器"""
    # 训练数据加载器
    train_loader = create_dataloader(
        dataset_type=config.dataset_type,
        data_path=config.data_path,
        split="train",
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True
    )
    
    # 验证数据加载器
    val_loader = None
    if args.val_split > 0:
        val_loader = create_dataloader(
            dataset_type=config.dataset_type,
            data_path=config.data_path,
            split="val",
            batch_size=config.training.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    return train_loader, val_loader


def main():
    """主函数"""
    args = parse_args()
    
    # 设置配置
    config = setup_config(args)
    
    # 创建实验日志器
    logger, experiment_dir = create_experiment_logger(
        config.experiment_name,
        base_dir=config.output_dir
    )
    
    # 更新配置中的目录路径
    config.output_dir = experiment_dir
    config.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    config.log_dir = os.path.join(experiment_dir, "logs")
    
    # 设置环境
    setup_environment(config, args)
    
    logger.info("="*50)
    logger.info("ORION Training Started")
    logger.info("="*50)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Dataset: {config.dataset_type}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    
    # 验证配置
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(config, args)
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return 1
    
    # 创建模型
    logger.info("Creating model...")
    try:
        model = OrionSystem(config)
        
        # 编译模型（如果支持）
        if config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model...")
            model = torch.compile(model)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return 1
    
    # 加载预训练权重或恢复训练
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        # TODO: 实现预训练权重加载
    
    # 创建训练器
    logger.info("Creating trainer...")
    try:
        trainer = OrionTrainer(config, model)
        
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return 1
    
    # 干运行模式
    if args.dry_run:
        logger.info("Dry run mode - skipping actual training")
        logger.info("Configuration and setup completed successfully")
        return 0
    
    # 开始训练
    logger.info("Starting training...")
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info("="*50)
    logger.info("ORION Training Finished")
    logger.info("="*50)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
