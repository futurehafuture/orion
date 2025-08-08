"""可视化工具"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import os


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_single_trajectory(self, 
                             trajectory: np.ndarray,
                             title: str = "Trajectory",
                             save_path: Optional[str] = None,
                             show_velocity: bool = True) -> plt.Figure:
        """
        绘制单条轨迹
        
        Args:
            trajectory: (T, 2) 轨迹数据
            title: 图标题
            save_path: 保存路径
            show_velocity: 是否显示速度信息
            
        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(1, 2 if show_velocity else 1, figsize=self.figsize)
        if not show_velocity:
            axes = [axes]
        
        # 轨迹图
        ax1 = axes[0]
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
        
        # 添加箭头显示方向
        for i in range(0, len(trajectory)-1, max(1, len(trajectory)//10)):
            dx = trajectory[i+1, 0] - trajectory[i, 0]
            dy = trajectory[i+1, 1] - trajectory[i, 1]
            ax1.arrow(trajectory[i, 0], trajectory[i, 1], dx*0.5, dy*0.5, 
                     head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'{title} - Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 速度图
        if show_velocity:
            ax2 = axes[1]
            velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            times = np.arange(len(velocities))
            
            ax2.plot(times, velocities, 'r-', linewidth=2, label='Speed')
            ax2.fill_between(times, velocities, alpha=0.3)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Speed (m/s)')
            ax2.set_title(f'{title} - Speed Profile')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_trajectories(self,
                                 trajectories: List[np.ndarray],
                                 labels: Optional[List[str]] = None,
                                 title: str = "Multiple Trajectories",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制多条轨迹"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for i, traj in enumerate(trajectories):
            label = labels[i] if labels and i < len(labels) else f'Trajectory {i+1}'
            color = colors[i]
            
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, label=label)
            ax.scatter(traj[0, 0], traj[0, 1], c=color, s=80, marker='o', alpha=0.8)
            ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=80, marker='s', alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_comparison(self,
                                 ground_truth: np.ndarray,
                                 predictions: List[np.ndarray],
                                 prediction_labels: Optional[List[str]] = None,
                                 title: str = "Prediction vs Ground Truth",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制预测与真实轨迹对比"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 真实轨迹
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'k-', linewidth=3, 
               label='Ground Truth', zorder=10)
        ax.scatter(ground_truth[0, 0], ground_truth[0, 1], c='green', s=150, 
                  marker='o', label='Start', zorder=15)
        ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c='red', s=150, 
                  marker='s', label='End', zorder=15)
        
        # 预测轨迹
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        for i, pred in enumerate(predictions):
            label = prediction_labels[i] if prediction_labels and i < len(prediction_labels) else f'Prediction {i+1}'
            ax.plot(pred[:, 0], pred[:, 1], '--', color=colors[i], linewidth=2, 
                   label=label, alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trajectory_heatmap(self,
                              trajectories: List[np.ndarray],
                              title: str = "Trajectory Density",
                              save_path: Optional[str] = None,
                              bins: int = 50) -> plt.Figure:
        """绘制轨迹密度热力图"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 合并所有轨迹点
        all_points = np.vstack(trajectories)
        
        # 创建2D直方图
        h, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=bins)
        
        # 绘制热力图
        im = ax.imshow(h.T, origin='lower', aspect='auto', cmap='hot', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        # 覆盖几条示例轨迹
        for i, traj in enumerate(trajectories[:5]):  # 只显示前5条
            ax.plot(traj[:, 0], traj[:, 1], 'cyan', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class AttentionVisualizer:
    """注意力可视化器"""
    
    def __init__(self):
        pass
    
    def plot_attention_weights(self,
                             attention_weights: np.ndarray,
                             input_labels: Optional[List[str]] = None,
                             output_labels: Optional[List[str]] = None,
                             title: str = "Attention Weights",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制注意力权重矩阵
        
        Args:
            attention_weights: (num_heads, seq_len, seq_len) 或 (seq_len, seq_len)
            input_labels: 输入序列标签
            output_labels: 输出序列标签
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        if attention_weights.ndim == 3:
            num_heads = attention_weights.shape[0]
            fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
            if num_heads == 1:
                axes = [axes]
        else:
            num_heads = 1
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
            attention_weights = attention_weights[np.newaxis, ...]
        
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            weights = attention_weights[head_idx]
            
            # 绘制热力图
            im = ax.imshow(weights, cmap='Blues', aspect='auto')
            
            # 设置标签
            if input_labels:
                ax.set_xticks(range(len(input_labels)))
                ax.set_xticklabels(input_labels, rotation=45)
            if output_labels:
                ax.set_yticks(range(len(output_labels)))
                ax.set_yticklabels(output_labels)
            
            ax.set_title(f'{title} - Head {head_idx + 1}' if num_heads > 1 else title)
            ax.set_xlabel('Input Position')
            ax.set_ylabel('Output Position')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 添加数值标注（如果矩阵较小）
            if weights.shape[0] <= 10 and weights.shape[1] <= 10:
                for i in range(weights.shape[0]):
                    for j in range(weights.shape[1]):
                        ax.text(j, i, f'{weights[i, j]:.2f}', 
                               ha='center', va='center', color='red', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_over_image(self,
                                image: np.ndarray,
                                attention_map: np.ndarray,
                                title: str = "Attention over Image",
                                save_path: Optional[str] = None,
                                alpha: float = 0.6) -> plt.Figure:
        """在图像上叠加注意力图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 注意力图
        axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        # 叠加图
        axes[2].imshow(image)
        axes[2].imshow(attention_map, cmap='hot', alpha=alpha)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MetricsVisualizer:
    """指标可视化器"""
    
    def __init__(self):
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_training_curves(self,
                           history: Dict[str, List[float]],
                           title: str = "Training Curves",
                           save_path: Optional[str] = None,
                           smooth: bool = True,
                           window_size: int = 10) -> plt.Figure:
        """绘制训练曲线"""
        # 分离训练和验证指标
        train_metrics = {k: v for k, v in history.items() if k.startswith('train_')}
        val_metrics = {k: v for k, v in history.items() if k.startswith('val_')}
        
        # 创建子图
        num_metrics = len(set(k.replace('train_', '').replace('val_', '') 
                            for k in list(train_metrics.keys()) + list(val_metrics.keys())))
        
        fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if num_metrics > 1 else [axes]
        
        metric_names = list(set(k.replace('train_', '').replace('val_', '') 
                              for k in list(train_metrics.keys()) + list(val_metrics.keys())))
        
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            
            # 训练曲线
            train_key = f'train_{metric_name}'
            if train_key in train_metrics:
                values = train_metrics[train_key]
                if smooth and len(values) > window_size:
                    values = self._smooth_curve(values, window_size)
                ax.plot(values, label=f'Train {metric_name.title()}', 
                       color=self.colors[0], linewidth=2)
            
            # 验证曲线
            val_key = f'val_{metric_name}'
            if val_key in val_metrics:
                values = val_metrics[val_key]
                if smooth and len(values) > window_size:
                    values = self._smooth_curve(values, window_size)
                ax.plot(values, label=f'Val {metric_name.title()}', 
                       color=self.colors[1], linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.title())
            ax.set_title(f'{metric_name.title()} Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self,
                              metrics_dict: Dict[str, Dict[str, float]],
                              title: str = "Metrics Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
        """绘制指标对比图"""
        metric_names = list(list(metrics_dict.values())[0].keys())
        model_names = list(metrics_dict.keys())
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name][metric] for metric in metric_names]
            ax.bar(x + i * width, values, width, label=model_name, 
                  color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _smooth_curve(self, values: List[float], window_size: int) -> List[float]:
        """平滑曲线"""
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start_idx:end_idx]))
        return smoothed


def save_visualization_grid(visualizations: List[plt.Figure],
                          save_path: str,
                          grid_shape: Optional[Tuple[int, int]] = None) -> None:
    """保存可视化网格"""
    if grid_shape is None:
        n = len(visualizations)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid_shape = (rows, cols)
    
    fig, axes = plt.subplots(*grid_shape, figsize=(15, 10))
    if grid_shape[0] * grid_shape[1] == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, vis_fig in enumerate(visualizations):
        if i < len(axes):
            # 这里需要将子图内容复制到网格中
            # 实际实现可能需要更复杂的逻辑
            axes[i].text(0.5, 0.5, f'Visualization {i+1}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Plot {i+1}')
    
    # 隐藏多余的子图
    for i in range(len(visualizations), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_demo_video(images: List[np.ndarray],
                     trajectories: List[np.ndarray],
                     predictions: List[np.ndarray],
                     save_path: str,
                     fps: int = 10) -> None:
    """创建演示视频"""
    if not images or not trajectories:
        raise ValueError("Images and trajectories cannot be empty")
    
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width * 2, height))
    
    try:
        for i, (img, traj, pred) in enumerate(zip(images, trajectories, predictions)):
            # 左侧：原始图像
            left_frame = img.copy()
            
            # 右侧：轨迹可视化
            right_frame = np.ones_like(img) * 255
            
            # 绘制轨迹
            traj_scaled = ((traj + 10) * width / 20).astype(int)  # 简单缩放
            pred_scaled = ((pred + 10) * width / 20).astype(int)
            
            # 真实轨迹（绿色）
            for j in range(len(traj_scaled) - 1):
                cv2.line(right_frame, tuple(traj_scaled[j]), tuple(traj_scaled[j+1]), (0, 255, 0), 2)
            
            # 预测轨迹（红色）
            for j in range(len(pred_scaled) - 1):
                cv2.line(right_frame, tuple(pred_scaled[j]), tuple(pred_scaled[j+1]), (0, 0, 255), 2)
            
            # 合并帧
            combined_frame = np.hstack([left_frame, right_frame])
            
            # 添加文本
            cv2.putText(combined_frame, f'Frame {i+1}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, 'Original', (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, 'Trajectory', (width + 10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(combined_frame)
    
    finally:
        writer.release()


class PerformanceDashboard:
    """性能仪表板"""
    
    def __init__(self, figsize: Tuple[int, int] = (20, 12)):
        self.figsize = figsize
    
    def create_dashboard(self,
                        metrics_history: Dict[str, List[float]],
                        current_metrics: Dict[str, float],
                        model_info: Dict[str, Any],
                        save_path: Optional[str] = None) -> plt.Figure:
        """创建性能仪表板"""
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 损失曲线
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(ax1, metrics_history)
        
        # 2. 指标曲线
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_metric_curves(ax2, metrics_history)
        
        # 3. 当前指标
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_current_metrics(ax3, current_metrics)
        
        # 4. 模型信息
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_model_info(ax4, model_info)
        
        # 5. 系统状态
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_system_status(ax5)
        
        plt.suptitle('ORION Training Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_loss_curves(self, ax, metrics_history):
        """绘制损失曲线"""
        loss_keys = [k for k in metrics_history.keys() if 'loss' in k.lower()]
        for key in loss_keys:
            ax.plot(metrics_history[key], label=key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metric_curves(self, ax, metrics_history):
        """绘制指标曲线"""
        metric_keys = [k for k in metrics_history.keys() if 'accuracy' in k.lower() or 'f1' in k.lower()]
        for key in metric_keys:
            ax.plot(metrics_history[key], label=key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_current_metrics(self, ax, current_metrics):
        """绘制当前指标"""
        metrics = list(current_metrics.keys())
        values = list(current_metrics.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
        
        bars = ax.bar(metrics, values, color=colors)
        ax.set_title('Current Performance')
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_model_info(self, ax, model_info):
        """绘制模型信息"""
        ax.axis('off')
        info_text = []
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    info_text.append(f"{key}: {value:.2f}")
                else:
                    info_text.append(f"{key}: {value:,}")
            else:
                info_text.append(f"{key}: {value}")
        
        ax.text(0.1, 0.9, "Model Information:", fontweight='bold', fontsize=12,
                transform=ax.transAxes)
        
        for i, line in enumerate(info_text):
            ax.text(0.1, 0.8 - i*0.1, line, fontsize=10, transform=ax.transAxes)
    
    def _plot_system_status(self, ax):
        """绘制系统状态"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU使用率（如果可用）
            gpu_percent = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100
            except ImportError:
                pass
            
            # 绘制使用率
            categories = ['CPU', 'Memory', 'GPU']
            values = [cpu_percent, memory_percent, gpu_percent]
            colors = ['skyblue', 'lightgreen', 'orange']
            
            bars = ax.bar(categories, values, color=colors)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Usage (%)')
            ax.set_title('System Resource Usage')
            
            # 添加百分比标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
        
        except ImportError:
            ax.text(0.5, 0.5, 'psutil not available\nfor system monitoring',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('System Status - Unavailable')
