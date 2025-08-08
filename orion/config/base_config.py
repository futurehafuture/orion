"""基础配置类"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
import yaml
import os

from .model_configs import (
    VisionConfig,
    QTFormerConfig, 
    LLMConfig,
    PlannerConfig,
    TrainingConfig,
)


@dataclass
class OrionConfig:
    """ORION系统主配置类"""
    
    # 模型配置
    vision: VisionConfig = field(default_factory=VisionConfig)
    qt_former: QTFormerConfig = field(default_factory=QTFormerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 数据配置
    data_path: str = "data/"
    dataset_type: str = "toy"  # toy, carla, nuscenes
    image_size: tuple = (224, 224)
    sequence_length: int = 8
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    compile_model: bool = False
    
    # 实验配置
    experiment_name: str = "orion_experiment"
    output_dir: str = "outputs/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    # 其他配置
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "OrionConfig":
        """从YAML文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # 使用OmegaConf处理嵌套配置
        omega_conf = OmegaConf.create(config_dict)
        return cls(**omega_conf)
    
    def to_yaml(self, save_path: str) -> None:
        """保存配置到YAML文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        omega_conf = OmegaConf.structured(self)
        with open(save_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(omega_conf, f)
    
    def update(self, **kwargs) -> "OrionConfig":
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def validate(self) -> None:
        """验证配置的有效性"""
        # 检查关键参数
        assert self.vision.output_dim == self.qt_former.vision_dim, \
            "Vision output dim must match QT-Former vision dim"
        assert self.qt_former.token_dim == self.llm.token_dim, \
            "QT-Former token dim must match LLM token dim"
        assert self.llm.token_dim == self.planner.token_dim, \
            "LLM token dim must match Planner token dim"
        
        # 检查路径
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
