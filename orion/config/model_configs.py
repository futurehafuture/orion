"""模型配置类"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class VisionConfig:
    """视觉编码器配置"""
    backbone: str = "resnet18"  # resnet18, resnet50, efficientnet, etc.
    pretrained: bool = True
    output_dim: int = 512
    frozen_layers: int = 0  # 冻结的层数
    dropout: float = 0.1


@dataclass 
class QTFormerConfig:
    """QT-Former配置"""
    vision_dim: int = 512
    token_dim: int = 512
    num_history_queries: int = 8
    memory_size: int = 64
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    memory_update_rate: float = 0.1


@dataclass
class LLMConfig:
    """LLM配置"""
    model_type: str = "toy"  # toy, gpt2, llama, etc.
    model_name: Optional[str] = None  # HuggingFace模型名
    token_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    hidden_dim: int = 2048
    vqa_classes: int = 16
    max_sequence_length: int = 256
    dropout: float = 0.1
    use_cache: bool = True


@dataclass
class PlannerConfig:
    """生成式规划器配置"""
    token_dim: int = 512
    traj_len: int = 20
    hidden_dim: int = 512
    latent_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    # 损失权重
    beta_kl: float = 0.001
    align_weight: float = 0.1
    diversity_weight: float = 0.05


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # 学习率调度
    scheduler: str = "cosine"  # cosine, linear, exponential
    min_lr: float = 1e-6
    
    # 优化器
    optimizer: str = "adamw"  # adamw, adam, sgd
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # 梯度处理
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # 损失权重
    vqa_weight: float = 0.1
    trajectory_weight: float = 1.0
    temporal_consistency_weight: float = 0.05
    
    # 验证和保存
    eval_every: int = 1000
    save_every: int = 5000
    keep_checkpoints: int = 5
    
    # 早停
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
