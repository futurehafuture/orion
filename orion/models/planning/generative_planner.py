"""生成式轨迹规划器实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math

from ...config.model_configs import PlannerConfig


class TrajectoryDecoder(nn.Module):
    """轨迹解码器 - 从潜在空间生成轨迹"""
    
    def __init__(self, latent_dim: int, condition_dim: int, 
                 traj_len: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.traj_len = traj_len
        self.latent_dim = latent_dim
        
        # 输入投影
        self.input_proj = nn.Linear(latent_dim + condition_dim, hidden_dim)
        
        # RNN解码器（用于时序建模）
        self.rnn = nn.GRU(
            input_size=2,  # x, y坐标
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 2)  # x, y坐标
            ) for _ in range(traj_len)
        ])
        
        # 初始化隐藏状态的投影
        self.hidden_proj = nn.Linear(latent_dim + condition_dim, num_layers * hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) 潜在变量
            condition: (B, condition_dim) 条件
            
        Returns:
            trajectory: (B, traj_len, 2) 生成的轨迹
        """
        B = z.size(0)
        device = z.device
        
        # 融合潜在变量和条件
        combined = torch.cat([z, condition], dim=1)  # (B, latent_dim + condition_dim)
        
        # 初始化隐藏状态
        h_init = self.hidden_proj(combined)  # (B, num_layers * hidden_dim)
        h_init = h_init.view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        
        # 自回归生成轨迹
        trajectory = []
        current_pos = torch.zeros(B, 1, 2, device=device)  # 起始位置
        hidden = h_init
        
        for step in range(self.traj_len):
            # RNN前向传播
            output, hidden = self.rnn(current_pos, hidden)
            
            # 预测下一个位置的偏移
            delta = self.output_layers[step](output.squeeze(1))  # (B, 2)
            
            # 累积位置
            if step == 0:
                next_pos = delta.unsqueeze(1)
            else:
                next_pos = trajectory[-1] + delta.unsqueeze(1)
            
            trajectory.append(next_pos)
            current_pos = next_pos
        
        # 拼接轨迹
        trajectory = torch.cat(trajectory, dim=1)  # (B, traj_len, 2)
        
        return trajectory


class ConditionalTrajectoryVAE(nn.Module):
    """
    条件轨迹变分自编码器
    - 编码器：q(z|x,c) 给定轨迹和条件预测潜在变量
    - 解码器：p(x|z,c) 给定潜在变量和条件生成轨迹
    - 对齐机制：将规划token映射到潜在空间，实现推理空间和动作空间的统一
    """
    
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        self.traj_len = config.traj_len
        self.latent_dim = config.latent_dim
        self.token_dim = config.token_dim
        
        # 编码器 q(z|x,c)
        encoder_input_dim = config.traj_len * 2 + config.token_dim  # 轨迹展平 + 条件
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # 均值和方差预测
        self.mu_layer = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_layer = nn.Linear(config.hidden_dim, config.latent_dim)
        
        # 解码器 p(x|z,c)
        self.decoder = TrajectoryDecoder(
            latent_dim=config.latent_dim,
            condition_dim=config.token_dim,
            traj_len=config.traj_len,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )
        
        # 对齐头：将规划token映射到潜在空间
        self.alignment_head = nn.Sequential(
            nn.Linear(config.token_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        
        # 先验网络：p(z|c) 仅基于条件预测潜在变量
        self.prior_net = nn.Sequential(
            nn.Linear(config.token_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim * 2)  # mu + logvar
        )
        
        # 轨迹质量评估器
        self.quality_evaluator = nn.Sequential(
            nn.Linear(config.traj_len * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def encode(self, trajectory: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码轨迹和条件到潜在空间
        
        Args:
            trajectory: (B, T, 2) 轨迹
            condition: (B, D) 条件（规划token）
            
        Returns:
            包含mu, logvar的字典
        """
        B, T, _ = trajectory.shape
        
        # 展平轨迹
        traj_flat = trajectory.view(B, -1)  # (B, T*2)
        
        # 拼接轨迹和条件
        x = torch.cat([traj_flat, condition], dim=1)
        
        # 编码
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return {"mu": mu, "logvar": logvar}
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        从潜在变量解码轨迹
        
        Args:
            z: (B, latent_dim) 潜在变量
            condition: (B, token_dim) 条件
            
        Returns:
            trajectory: (B, T, 2) 重构的轨迹
        """
        return self.decoder(z, condition)
    
    def get_prior(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取先验分布参数"""
        prior_params = self.prior_net(condition)  # (B, latent_dim * 2)
        mu_prior = prior_params[:, :self.latent_dim]
        logvar_prior = prior_params[:, self.latent_dim:]
        return {"mu_prior": mu_prior, "logvar_prior": logvar_prior}
    
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor, 
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            trajectory: (B, T, 2) 目标轨迹
            condition: (B, D) 规划token
            training: 是否为训练模式
            
        Returns:
            包含重构轨迹、潜在变量、损失项等的字典
        """
        # 编码
        posterior = self.encode(trajectory, condition)
        mu_post, logvar_post = posterior["mu"], posterior["logvar"]
        
        # 获取先验
        prior = self.get_prior(condition)
        mu_prior, logvar_prior = prior["mu_prior"], prior["logvar_prior"]
        
        # 重参数化
        z = self.reparameterize(mu_post, logvar_post)
        
        # 解码
        recon_trajectory = self.decode(z, condition)
        
        # 对齐目标：规划token的潜在表示
        align_target = self.alignment_head(condition)
        
        # 轨迹质量评分
        quality_score = self.quality_evaluator(trajectory.view(trajectory.size(0), -1))
        
        # 如果是推理模式，也从先验采样
        if not training:
            z_prior = self.reparameterize(mu_prior, logvar_prior)
            prior_trajectory = self.decode(z_prior, condition)
        else:
            prior_trajectory = None
        
        return {
            "recon_trajectory": recon_trajectory,
            "prior_trajectory": prior_trajectory,
            "mu_posterior": mu_post,
            "logvar_posterior": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "z": z,
            "align_target": align_target,
            "quality_score": quality_score
        }
    
    def sample(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        从先验分布采样生成轨迹
        
        Args:
            condition: (B, D) 条件
            num_samples: 每个条件生成的轨迹数量
            
        Returns:
            trajectories: (B, num_samples, T, 2) 生成的轨迹
        """
        B = condition.size(0)
        device = condition.device
        
        # 获取先验参数
        prior = self.get_prior(condition)
        mu_prior, logvar_prior = prior["mu_prior"], prior["logvar_prior"]
        
        # 采样多个轨迹
        trajectories = []
        for _ in range(num_samples):
            z = self.reparameterize(mu_prior, logvar_prior)
            traj = self.decode(z, condition)
            trajectories.append(traj.unsqueeze(1))
        
        trajectories = torch.cat(trajectories, dim=1)  # (B, num_samples, T, 2)
        
        return trajectories
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    target_trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            outputs: 模型输出
            target_trajectory: (B, T, 2) 目标轨迹
            
        Returns:
            包含各项损失的字典
        """
        # 重构损失
        recon_loss = F.smooth_l1_loss(
            outputs["recon_trajectory"], 
            target_trajectory,
            reduction='mean'
        )
        
        # KL散度损失（后验与先验）
        mu_post = outputs["mu_posterior"]
        logvar_post = outputs["logvar_posterior"]
        mu_prior = outputs["mu_prior"]
        logvar_prior = outputs["logvar_prior"]
        
        # KL(q(z|x,c) || p(z|c))
        kl_loss = 0.5 * torch.mean(
            logvar_prior - logvar_post - 1 + 
            (logvar_post.exp() + (mu_post - mu_prior).pow(2)) / logvar_prior.exp()
        )
        
        # 对齐损失：鼓励后验均值接近规划token的对齐目标
        align_loss = F.mse_loss(mu_post, outputs["align_target"])
        
        # 轨迹平滑损失
        smooth_loss = self._compute_smoothness_loss(outputs["recon_trajectory"])
        
        # 质量一致性损失
        target_quality = self.quality_evaluator(target_trajectory.view(target_trajectory.size(0), -1))
        pred_quality = outputs["quality_score"]
        quality_loss = F.mse_loss(pred_quality, target_quality.detach())
        
        # 总损失
        total_loss = (
            recon_loss + 
            self.config.beta_kl * kl_loss + 
            self.config.align_weight * align_loss +
            0.01 * smooth_loss +
            0.05 * quality_loss
        )
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "align_loss": align_loss,
            "smooth_loss": smooth_loss,
            "quality_loss": quality_loss
        }
    
    def _compute_smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """计算轨迹平滑损失"""
        # 计算相邻点的加速度
        velocity = trajectory[:, 1:] - trajectory[:, :-1]  # (B, T-1, 2)
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # (B, T-2, 2)
        
        # L2正则化加速度
        smooth_loss = torch.mean(torch.norm(acceleration, dim=-1))
        
        return smooth_loss
    
    def get_trajectory_features(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取轨迹特征（用于分析）"""
        B, T, _ = trajectory.shape
        
        # 速度
        velocity = torch.norm(trajectory[:, 1:] - trajectory[:, :-1], dim=-1)  # (B, T-1)
        avg_speed = torch.mean(velocity, dim=1)  # (B,)
        max_speed = torch.max(velocity, dim=1)[0]  # (B,)
        
        # 加速度
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # (B, T-2)
        avg_accel = torch.mean(torch.abs(acceleration), dim=1)  # (B,)
        
        # 曲率（转向角度变化）
        directions = trajectory[:, 1:] - trajectory[:, :-1]  # (B, T-1, 2)
        angles = torch.atan2(directions[:, :, 1], directions[:, :, 0])  # (B, T-1)
        angle_changes = torch.abs(angles[:, 1:] - angles[:, :-1])  # (B, T-2)
        avg_curvature = torch.mean(angle_changes, dim=1)  # (B,)
        
        # 总距离
        distances = torch.norm(directions, dim=-1)  # (B, T-1)
        total_distance = torch.sum(distances, dim=1)  # (B,)
        
        return {
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "avg_acceleration": avg_accel,
            "avg_curvature": avg_curvature,
            "total_distance": total_distance,
            "trajectory_length": torch.full((B,), T, device=trajectory.device)
        }
