"""ORION损失函数"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..config.model_configs import TrainingConfig


class OrionLoss(nn.Module):
    """
    ORION统一损失函数
    
    包含多个损失项：
    1. 轨迹重建损失
    2. KL散度损失
    3. 对齐损失
    4. VQA分类损失
    5. 时序一致性损失
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # 损失权重
        self.trajectory_weight = config.trajectory_weight
        self.vqa_weight = config.vqa_weight
        self.temporal_weight = config.temporal_consistency_weight
        
        # 轨迹损失函数
        self.trajectory_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        # VQA损失函数
        self.vqa_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        # 时序一致性损失
        self.temporal_loss_fn = nn.MSELoss(reduction='mean')
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出
            targets: 目标数据
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 轨迹规划损失
        if "planning_recon_trajectory" in outputs and "trajectory" in targets:
            trajectory_losses = self._compute_trajectory_losses(outputs, targets)
            losses.update(trajectory_losses)
            total_loss += trajectory_losses["trajectory_total"] * self.trajectory_weight
        
        # 2. VQA损失
        vqa_losses = self._compute_vqa_losses(outputs, targets)
        if vqa_losses:
            losses.update(vqa_losses)
            total_loss += vqa_losses["vqa_total"] * self.vqa_weight
        
        # 3. 时序一致性损失
        if "temporal_consistency" in outputs:
            temporal_loss = self._compute_temporal_loss(outputs)
            losses["temporal_consistency"] = temporal_loss
            total_loss += temporal_loss * self.temporal_weight
        
        # 4. 规划token对齐损失（通过VAE内部处理）
        
        losses["total_loss"] = total_loss
        return losses
    
    def _compute_trajectory_losses(self, outputs: Dict[str, torch.Tensor],
                                 targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算轨迹相关损失"""
        losses = {}
        
        # 基础重建损失
        if "planning_recon_trajectory" in outputs:
            recon_loss = self.trajectory_loss_fn(
                outputs["planning_recon_trajectory"],
                targets["trajectory"]
            )
            losses["trajectory_recon"] = recon_loss
        
        # KL散度损失
        if "planning_mu_posterior" in outputs and "planning_logvar_posterior" in outputs:
            mu_post = outputs["planning_mu_posterior"]
            logvar_post = outputs["planning_logvar_posterior"]
            
            if "planning_mu_prior" in outputs and "planning_logvar_prior" in outputs:
                # KL(posterior || prior)
                mu_prior = outputs["planning_mu_prior"]
                logvar_prior = outputs["planning_logvar_prior"]
                
                kl_loss = 0.5 * torch.mean(
                    logvar_prior - logvar_post - 1 +
                    (logvar_post.exp() + (mu_post - mu_prior).pow(2)) / logvar_prior.exp()
                )
            else:
                # KL(posterior || N(0,1))
                kl_loss = -0.5 * torch.mean(
                    1 + logvar_post - mu_post.pow(2) - logvar_post.exp()
                )
            
            losses["trajectory_kl"] = kl_loss
        
        # 对齐损失
        if "planning_align_target" in outputs and "planning_mu_posterior" in outputs:
            align_loss = F.mse_loss(
                outputs["planning_mu_posterior"],
                outputs["planning_align_target"]
            )
            losses["trajectory_align"] = align_loss
        
        # 平滑性损失
        if "planning_recon_trajectory" in outputs:
            smooth_loss = self._compute_smoothness_loss(outputs["planning_recon_trajectory"])
            losses["trajectory_smooth"] = smooth_loss
        
        # 质量一致性损失
        if "planning_quality_score" in outputs:
            # 鼓励高质量轨迹
            quality_target = torch.ones_like(outputs["planning_quality_score"])
            quality_loss = F.mse_loss(outputs["planning_quality_score"], quality_target)
            losses["trajectory_quality"] = quality_loss
        
        # 轨迹总损失
        trajectory_total = sum(losses.values())
        losses["trajectory_total"] = trajectory_total
        
        return losses
    
    def _compute_vqa_losses(self, outputs: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算VQA损失"""
        losses = {}
        vqa_loss_total = 0.0
        vqa_count = 0
        
        # 遍历所有VQA任务
        vqa_tasks = ["scene_classification", "weather_detection", "traffic_light", "driving_intent"]
        
        for task in vqa_tasks:
            logits_key = f"vqa_{task}_logits"
            labels_key = f"{task}_labels"
            
            if logits_key in outputs and labels_key in targets:
                task_loss = self.vqa_loss_fn(outputs[logits_key], targets[labels_key])
                losses[f"vqa_{task}"] = task_loss
                vqa_loss_total += task_loss
                vqa_count += 1
        
        # 通用VQA损失
        if "vqa_logits" in outputs and "vqa_labels" in targets:
            vqa_loss = self.vqa_loss_fn(outputs["vqa_logits"], targets["vqa_labels"])
            losses["vqa_general"] = vqa_loss
            vqa_loss_total += vqa_loss
            vqa_count += 1
        
        # VQA置信度损失
        if "vqa_confidence" in outputs:
            # 鼓励高置信度预测
            confidence_target = torch.ones_like(outputs["vqa_confidence"])
            confidence_loss = F.mse_loss(outputs["vqa_confidence"], confidence_target)
            losses["vqa_confidence"] = confidence_loss
            vqa_loss_total += 0.1 * confidence_loss
        
        if vqa_count > 0:
            losses["vqa_total"] = vqa_loss_total / vqa_count
        
        return losses
    
    def _compute_temporal_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算时序一致性损失"""
        # 鼓励时序一致性接近1
        consistency_target = torch.ones_like(outputs["temporal_consistency"])
        temporal_loss = self.temporal_loss_fn(
            outputs["temporal_consistency"], 
            consistency_target
        )
        return temporal_loss
    
    def _compute_smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """计算轨迹平滑性损失"""
        # 计算速度（一阶差分）
        velocity = trajectory[:, 1:] - trajectory[:, :-1]  # (B, T-1, 2)
        
        # 计算加速度（二阶差分）
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # (B, T-2, 2)
        
        # L2正则化加速度以鼓励平滑运动
        smooth_loss = torch.mean(torch.norm(acceleration, dim=-1))
        
        return smooth_loss


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """对比学习损失"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 特征向量
            labels: (B,) 标签
        """
        # 计算相似度矩阵
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线（自身相似度）
        mask = mask.fill_diagonal_(0)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # 正样本概率
        pos_sim = exp_sim * mask
        pos_prob = pos_sim / sum_exp_sim
        
        # 计算损失
        loss = -torch.log(pos_prob + 1e-8) * mask
        loss = torch.sum(loss) / torch.sum(mask)
        
        return loss


class MultiTaskLoss(nn.Module):
    """多任务损失自适应权重"""
    
    def __init__(self, num_tasks: int, method: str = 'uncertainty'):
        super().__init__()
        self.num_tasks = num_tasks
        self.method = method
        
        if method == 'uncertainty':
            # 基于不确定性的权重
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        elif method == 'gradnorm':
            # GradNorm方法的初始权重
            self.task_weights = nn.Parameter(torch.ones(num_tasks))
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            losses: (num_tasks,) 各任务损失
        """
        if self.method == 'uncertainty':
            # 基于不确定性加权：1/(2*sigma^2) * loss + log(sigma)
            precision = torch.exp(-self.log_vars)
            weighted_losses = precision * losses + self.log_vars
            return torch.mean(weighted_losses)
        
        elif self.method == 'gradnorm':
            # GradNorm动态权重
            weighted_losses = self.task_weights * losses
            return torch.sum(weighted_losses)
        
        else:
            # 均等权重
            return torch.mean(losses)
