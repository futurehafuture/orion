"""ORION系统主模型"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..config import OrionConfig
from .vision import VisionBackbone
from .temporal import QTFormer
from .reasoning import LLMInterface, ToyLLM, HuggingFaceLLM, VQAHead
from .planning import ConditionalTrajectoryVAE


class OrionSystem(nn.Module):
    """
    ORION完整系统
    
    架构流程：
    1. 视觉编码器：提取当前帧特征
    2. QT-Former：融合历史记忆，生成时序感知的token
    3. LLM推理：基于视觉token进行场景理解和推理
    4. 生成式规划器：将推理结果转化为具体的轨迹规划
    
    关键创新：
    - 统一推理空间和动作空间：通过规划token实现对齐
    - 时序记忆机制：QT-Former维护长期历史信息
    - 多任务学习：同时优化VQA和轨迹规划任务
    """
    
    def __init__(self, config: OrionConfig):
        super().__init__()
        self.config = config
        
        # 验证配置
        config.validate()
        
        # 视觉骨干网络
        self.vision_backbone = VisionBackbone(config.vision)
        
        # QT-Former时序模块
        self.qt_former = QTFormer(config.qt_former)
        
        # LLM推理模块
        if config.llm.model_type == "toy":
            self.llm = ToyLLM(config.llm)
        elif config.llm.model_type in ["gpt2", "llama", "mistral"]:
            self.llm = HuggingFaceLLM(config.llm)
        else:
            raise ValueError(f"Unsupported LLM type: {config.llm.model_type}")
        
        # VQA头部
        self.vqa_head = VQAHead(
            input_dim=config.llm.token_dim,
            num_classes=config.llm.vqa_classes,
            task_types=["scene_classification", "weather_detection", "traffic_light", "driving_intent"]
        )
        
        # 生成式规划器
        self.trajectory_planner = ConditionalTrajectoryVAE(config.planner)
        
        # 时序一致性模块
        self.temporal_consistency = nn.Sequential(
            nn.Linear(config.planner.token_dim * 2, config.planner.token_dim),
            nn.ReLU(),
            nn.Linear(config.planner.token_dim, 1),
            nn.Sigmoid()
        )
        
        # 注册历史规划token缓存
        self.register_buffer(
            "prev_planning_token",
            torch.zeros(1, config.planner.token_dim),
            persistent=False
        )
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def reset(self):
        """重置系统状态（新episode开始时调用）"""
        self.qt_former.reset_memory()
        self.prev_planning_token.zero_()
        self.logger.info("ORION system state reset")
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 包含以下键的字典
                - image: (B, C, H, W) 当前帧图像
                - trajectory: (B, T, 2) 目标轨迹（训练时）
                - text_prompt: 可选的文本提示
            training: 是否为训练模式
            
        Returns:
            包含所有输出的字典
        """
        device = next(self.parameters()).device
        batch_size = batch["image"].size(0)
        
        # 确保prev_planning_token的批次维度正确
        if self.prev_planning_token.size(0) != batch_size:
            self.prev_planning_token = self.prev_planning_token.expand(batch_size, -1).contiguous()
        
        # Step 1: 视觉特征提取（提供patch token以提升细粒度对齐）
        vision_features, patch_tokens = self.vision_backbone(batch["image"], return_patches=True)  # (B, D), (B, N, D)
        
        # Step 2: QT-Former时序建模（返回历史/场景LLM标记与规划token）
        x_h, x_s, _ = self.qt_former(vision_features, patch_tokens)  # qt_planning_token is no longer used here
        # 组装给LLM的视觉token序列（场景在前，历史在后）
        llm_tokens = torch.cat([x_s, x_h], dim=1)
        
        # Step 3: LLM推理
        text_input = batch.get("text_prompt", None)
        llm_outputs = self.llm(llm_tokens, text_input)
        planning_token = llm_outputs["planning_token"]  # (B, D_planner) - This is the primary planning token now
        
        # Step 4: VQA预测
        vqa_outputs = self.vqa_head(llm_outputs.get("vqa_logits"))
        
        # Step 5: (Removed) 规划token融合 is no longer needed
        
        # Step 6: 时序一致性检查
        if hasattr(self, 'prev_planning_token') and self.prev_planning_token.numel() > 0:
            consistency_input = torch.cat([
                planning_token, self.prev_planning_token
            ], dim=-1)
            temporal_consistency = self.temporal_consistency(consistency_input)
        else:
            temporal_consistency = torch.ones(batch_size, 1, device=device)
        
        # Step 7: 轨迹规划
        if training and "trajectory" in batch:
            # 训练模式：使用目标轨迹
            planning_outputs = self.trajectory_planner(
                batch["trajectory"], planning_token, training=True
            )
        else:
            # 推理模式：从先验采样
            planning_outputs = {}
            sampled_trajectories = self.trajectory_planner.sample(
                planning_token, num_samples=5
            )  # (B, 5, T, 2)
            planning_outputs["sampled_trajectories"] = sampled_trajectories
            planning_outputs["best_trajectory"] = sampled_trajectories[:, 0]  # 选择第一个
        
        # 更新历史规划token
        self.prev_planning_token = planning_token.detach()
        
        # 整合所有输出
        outputs = {
            # 特征表示
            "vision_features": vision_features,
            "llm_tokens": llm_tokens,
            "planning_token": planning_token,
            
            # VQA输出
            **{f"vqa_{k}": v for k, v in vqa_outputs.items()},
            
            # 规划输出
            **{f"planning_{k}": v for k, v in planning_outputs.items()},
            
            # 其他
            "temporal_consistency": temporal_consistency,
            "training": training
        }

        # 附加QT-Former辅助输出
        if hasattr(self.qt_former, '_aux_outputs'):
            for k, v in self.qt_former._aux_outputs.items():
                outputs[f"qt_{k}"] = v
        
        # 添加LLM的其他输出
        for key, value in llm_outputs.items():
            if key not in ["planning_token"]:
                outputs[f"llm_{key}"] = value
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出
            batch: 输入批次数据
            
        Returns:
            包含各项损失的字典
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 轨迹规划损失
        if "trajectory" in batch and "planning_recon_trajectory" in outputs:
            planning_loss_dict = self.trajectory_planner.compute_loss(
                {k.replace("planning_", ""): v for k, v in outputs.items() 
                 if k.startswith("planning_")},
                batch["trajectory"]
            )
            for key, value in planning_loss_dict.items():
                losses[f"planning_{key}"] = value
            total_loss += losses["planning_total_loss"] * self.config.training.trajectory_weight
        
        # 2. VQA损失
        if "vqa_labels" in batch:
            vqa_loss = 0.0
            vqa_count = 0
            for key, logits in outputs.items():
                if key.startswith("vqa_") and key.endswith("_logits"):
                    task_name = key[4:-7]  # 移除"vqa_"和"_logits"
                    if f"{task_name}_labels" in batch:
                        task_loss = nn.functional.cross_entropy(
                            logits, batch[f"{task_name}_labels"]
                        )
                        losses[f"vqa_{task_name}_loss"] = task_loss
                        vqa_loss += task_loss
                        vqa_count += 1
            
            if vqa_count > 0:
                vqa_loss = vqa_loss / vqa_count
                losses["vqa_total_loss"] = vqa_loss
                total_loss += vqa_loss * self.config.training.vqa_weight
        
        # 3. 时序一致性损失
        if "temporal_consistency" in outputs:
            # 鼓励时序一致性接近1（一致）
            temporal_loss = nn.functional.mse_loss(
                outputs["temporal_consistency"],
                torch.ones_like(outputs["temporal_consistency"])
            )
            losses["temporal_consistency_loss"] = temporal_loss
            total_loss += temporal_loss * self.config.training.temporal_consistency_weight
        
        # 4. 规划token对齐损失（隐式通过VAE的对齐损失处理）
        
        losses["total_loss"] = total_loss
        return losses
    
    def predict(self, image: torch.Tensor, 
                text_prompt: Optional[str] = None,
                num_trajectories: int = 5) -> Dict[str, Any]:
        """
        推理预测
        
        Args:
            image: (B, C, H, W) 输入图像
            text_prompt: 可选文本提示
            num_trajectories: 生成的轨迹数量
            
        Returns:
            预测结果字典
        """
        self.eval()
        with torch.no_grad():
            batch = {"image": image}
            if text_prompt:
                batch["text_prompt"] = text_prompt
            
            outputs = self.forward(batch, training=False)
            
            # 生成多个轨迹
            planning_token = outputs["planning_token"]
            trajectories = self.trajectory_planner.sample(
                planning_token, num_samples=num_trajectories
            )  # (B, num_trajectories, T, 2)
            
            # VQA预测
            vqa_predictions = {}
            for key, value in outputs.items():
                if key.startswith("vqa_") and key.endswith("_logits"):
                    task_name = key[4:-7]
                    probs = torch.softmax(value, dim=-1)
                    pred_classes = torch.argmax(probs, dim=-1)
                    vqa_predictions[task_name] = {
                        "predictions": pred_classes.cpu().numpy(),
                        "probabilities": probs.cpu().numpy()
                    }
            
            # 轨迹特征分析
            trajectory_features = []
            for i in range(num_trajectories):
                features = self.trajectory_planner.get_trajectory_features(
                    trajectories[:, i]
                )
                trajectory_features.append(features)
            
            return {
                "trajectories": trajectories.cpu().numpy(),
                "vqa_predictions": vqa_predictions,
                "trajectory_features": trajectory_features,
                "planning_token": planning_token.cpu().numpy(),
                "temporal_consistency": outputs["temporal_consistency"].cpu().numpy()
            }
    
    def explain_decision(self, image: torch.Tensor,
                        text_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        决策解释
        
        Args:
            image: (B, C, H, W) 输入图像
            text_prompt: 可选文本提示
            
        Returns:
            决策解释字典
        """
        self.eval()
        with torch.no_grad():
            batch = {"image": image}
            if text_prompt:
                batch["text_prompt"] = text_prompt
            
            outputs = self.forward(batch, training=False)
            
            # VQA解释
            vqa_explanations = {}
            for task_type in ["scene_classification", "weather_detection", "traffic_light", "driving_intent"]:
                if f"vqa_{task_type}_logits" in outputs:
                    explanation = self.vqa_head.explain_prediction(
                        outputs["llm_vqa_logits"], task_type
                    )
                    vqa_explanations[task_type] = explanation
            
            # 注意力可视化
            attention_maps = self.qt_former.get_attention_maps(outputs["vision_features"])
            
            # 特征重要性
            feature_importance = {
                "vision_features_norm": torch.norm(outputs["vision_features"], dim=-1).cpu().numpy(),
                "planning_token_importance": torch.norm(outputs["planning_token"], dim=-1).cpu().numpy(),
            }
            
            return {
                "vqa_explanations": vqa_explanations,
                "attention_maps": attention_maps.cpu().numpy(),
                "feature_importance": feature_importance,
                "memory_usage": self.qt_former.get_memory_usage()
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # 假设float32
            "modules": {
                "vision_backbone": sum(p.numel() for p in self.vision_backbone.parameters()),
                "qt_former": sum(p.numel() for p in self.qt_former.parameters()),
                "llm": sum(p.numel() for p in self.llm.parameters()),
                "vqa_head": sum(p.numel() for p in self.vqa_head.parameters()),
                "trajectory_planner": sum(p.numel() for p in self.trajectory_planner.parameters()),
            },
            "config": self.config
        }
