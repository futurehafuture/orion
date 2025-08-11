"""视觉骨干网络实现"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple

from ...config.model_configs import VisionConfig


class VisionBackbone(nn.Module):
    """
    可配置的视觉骨干网络
    支持多种预训练模型：ResNet, EfficientNet等
    """
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # 选择骨干网络
        if config.backbone.startswith('resnet'):
            self.backbone = self._build_resnet(config)
        elif config.backbone.startswith('efficientnet'):
            self.backbone = self._build_efficientnet(config)
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim)
        )
        
        # 冻结部分层
        self._freeze_layers(config.frozen_layers)
    
    def _build_resnet(self, config: VisionConfig) -> nn.Module:
        """构建ResNet骨干"""
        if config.backbone == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if config.pretrained else None)
            self.backbone_dim = 512
        elif config.backbone == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if config.pretrained else None)
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {config.backbone}")
        
        # 移除分类头
        modules = list(model.children())[:-1]
        return nn.Sequential(*modules)
    
    def _build_efficientnet(self, config: VisionConfig) -> nn.Module:
        """构建EfficientNet骨干"""
        if config.backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if config.pretrained else None)
            self.backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {config.backbone}")
        
        # 移除分类头
        model.classifier = nn.Identity()
        return model
    
    def _freeze_layers(self, num_layers: int):
        """冻结指定数量的层"""
        if num_layers <= 0:
            return
            
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor, return_patches: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: (B, C, H, W) 输入图像
            
        Returns:
            features: (B, D) 视觉特征
        """
        # 特征提取
        x = self.backbone(images)  # (B, backbone_dim, H', W') or (B, backbone_dim)
        
        if return_patches and x.dim() == 4:
            # 返回patch token：(B, N, C)
            B, C, H, W = x.shape
            patches = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
            patches = self.feature_proj(patches)
            # 同时返回全局向量
            pooled = torch.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            global_feat = self.feature_proj(pooled)
            return global_feat, patches
        else:
            # 全局平均池化
            if x.dim() > 2:
                x = torch.adaptive_avg_pool2d(x, (1, 1))
                x = x.flatten(1)
            features = self.feature_proj(x)
            return features
    
    def get_feature_dim(self) -> int:
        """获取输出特征维度"""
        return self.config.output_dim
    
    def get_layer_features(self, images: torch.Tensor, layer_name: str) -> torch.Tensor:
        """获取指定层的特征（用于可视化分析）"""
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # 注册钩子
        handles = []
        for name, module in self.backbone.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # 前向传播
        _ = self.backbone(images)
        
        # 移除钩子
        for handle in handles:
            handle.remove()
        
        return features.get(layer_name, None)
