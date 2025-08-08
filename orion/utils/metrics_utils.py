"""指标计算工具"""

import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings


def compute_metrics(predictions: Union[torch.Tensor, np.ndarray],
                   targets: Union[torch.Tensor, np.ndarray],
                   metric_types: List[str] = None) -> Dict[str, float]:
    """
    计算各种评估指标
    
    Args:
        predictions: 预测值
        targets: 真实值
        metric_types: 指标类型列表
        
    Returns:
        指标字典
    """
    if metric_types is None:
        metric_types = ['accuracy', 'precision', 'recall', 'f1']
    
    # 转换为numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # 分类指标
    if 'accuracy' in metric_types:
        metrics['accuracy'] = accuracy_score(targets, predictions)
    
    if any(metric in metric_types for metric in ['precision', 'recall', 'f1']):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        if 'precision' in metric_types:
            metrics['precision'] = precision
        if 'recall' in metric_types:
            metrics['recall'] = recall
        if 'f1' in metric_types:
            metrics['f1'] = f1
    
    return metrics


def compute_trajectory_metrics(pred_trajectories: Union[torch.Tensor, np.ndarray],
                             gt_trajectories: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    计算轨迹预测指标
    
    Args:
        pred_trajectories: 预测轨迹 (B, T, 2)
        gt_trajectories: 真实轨迹 (B, T, 2)
        
    Returns:
        轨迹指标字典
    """
    # 转换为numpy数组
    if isinstance(pred_trajectories, torch.Tensor):
        pred_trajectories = pred_trajectories.detach().cpu().numpy()
    if isinstance(gt_trajectories, torch.Tensor):
        gt_trajectories = gt_trajectories.detach().cpu().numpy()
    
    metrics = {}
    
    # Average Displacement Error (ADE)
    displacement_errors = np.linalg.norm(pred_trajectories - gt_trajectories, axis=2)
    ade = np.mean(displacement_errors)
    metrics['ade'] = ade
    
    # Final Displacement Error (FDE)
    final_errors = np.linalg.norm(pred_trajectories[:, -1] - gt_trajectories[:, -1], axis=1)
    fde = np.mean(final_errors)
    metrics['fde'] = fde
    
    # Miss Rate (预测终点距离真实终点超过2米的比例)
    miss_rate = np.mean(final_errors > 2.0)
    metrics['miss_rate'] = miss_rate
    
    # 速度一致性
    pred_velocities = np.diff(pred_trajectories, axis=1)
    gt_velocities = np.diff(gt_trajectories, axis=1)
    velocity_errors = np.linalg.norm(pred_velocities - gt_velocities, axis=2)
    metrics['velocity_consistency'] = np.mean(velocity_errors)
    
    # 方向一致性
    pred_directions = pred_velocities / (np.linalg.norm(pred_velocities, axis=2, keepdims=True) + 1e-8)
    gt_directions = gt_velocities / (np.linalg.norm(gt_velocities, axis=2, keepdims=True) + 1e-8)
    direction_similarity = np.sum(pred_directions * gt_directions, axis=2)
    metrics['direction_consistency'] = np.mean(direction_similarity)
    
    return metrics


def compute_collision_metrics(trajectories: Union[torch.Tensor, np.ndarray],
                            obstacles: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算碰撞检测指标
    
    Args:
        trajectories: 轨迹 (B, T, 2)
        obstacles: 障碍物列表，每个障碍物包含位置和大小信息
        
    Returns:
        碰撞指标字典
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    
    metrics = {}
    total_collisions = 0
    total_trajectories = trajectories.shape[0]
    
    for traj in trajectories:
        has_collision = False
        for point in traj:
            for obstacle in obstacles:
                # 简单的圆形碰撞检测
                distance = np.linalg.norm(point - obstacle['center'])
                if distance < obstacle['radius']:
                    has_collision = True
                    break
            if has_collision:
                break
        
        if has_collision:
            total_collisions += 1
    
    metrics['collision_rate'] = total_collisions / total_trajectories
    metrics['safety_rate'] = 1.0 - metrics['collision_rate']
    
    return metrics


def compute_comfort_metrics(trajectories: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    计算舒适度指标
    
    Args:
        trajectories: 轨迹 (B, T, 2)
        
    Returns:
        舒适度指标字典
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    
    metrics = {}
    
    # 加速度指标
    velocities = np.diff(trajectories, axis=1)  # (B, T-1, 2)
    accelerations = np.diff(velocities, axis=1)  # (B, T-2, 2)
    
    # 平均加速度幅值
    accel_magnitudes = np.linalg.norm(accelerations, axis=2)
    metrics['avg_acceleration'] = np.mean(accel_magnitudes)
    metrics['max_acceleration'] = np.max(accel_magnitudes)
    
    # 加速度变化率（Jerk）
    jerks = np.diff(accelerations, axis=1)  # (B, T-3, 2)
    jerk_magnitudes = np.linalg.norm(jerks, axis=2)
    metrics['avg_jerk'] = np.mean(jerk_magnitudes)
    metrics['max_jerk'] = np.max(jerk_magnitudes)
    
    # 路径平滑度
    curvatures = []
    for traj in trajectories:
        traj_curvature = compute_path_curvature(traj)
        curvatures.extend(traj_curvature)
    
    metrics['avg_curvature'] = np.mean(curvatures)
    metrics['max_curvature'] = np.max(curvatures)
    
    return metrics


def compute_path_curvature(trajectory: np.ndarray) -> List[float]:
    """计算路径曲率"""
    if len(trajectory) < 3:
        return [0.0]
    
    curvatures = []
    for i in range(1, len(trajectory) - 1):
        p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
        
        # 计算三点构成的曲率
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 避免除零
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0.0)
            continue
        
        # 叉积计算曲率
        cross_product = np.cross(v1, v2)
        curvature = abs(cross_product) / (v1_norm * v2_norm * np.linalg.norm(p3 - p1))
        curvatures.append(curvature)
    
    return curvatures


def compute_efficiency_metrics(trajectories: Union[torch.Tensor, np.ndarray],
                             target_points: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    计算效率指标
    
    Args:
        trajectories: 轨迹 (B, T, 2)
        target_points: 目标点 (B, 2)
        
    Returns:
        效率指标字典
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.detach().cpu().numpy()
    
    metrics = {}
    
    # 路径长度
    path_lengths = []
    direct_distances = []
    
    for traj, target in zip(trajectories, target_points):
        # 实际路径长度
        distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        path_length = np.sum(distances)
        path_lengths.append(path_length)
        
        # 直线距离
        direct_distance = np.linalg.norm(traj[-1] - traj[0])
        direct_distances.append(direct_distance)
    
    # 路径效率（直线距离 / 实际路径长度）
    path_efficiencies = []
    for path_len, direct_dist in zip(path_lengths, direct_distances):
        if path_len > 1e-8:
            efficiency = direct_dist / path_len
        else:
            efficiency = 1.0
        path_efficiencies.append(efficiency)
    
    metrics['avg_path_length'] = np.mean(path_lengths)
    metrics['avg_path_efficiency'] = np.mean(path_efficiencies)
    metrics['path_efficiency_std'] = np.std(path_efficiencies)
    
    return metrics


def compute_temporal_consistency_metrics(predictions: List[Union[torch.Tensor, np.ndarray]]) -> Dict[str, float]:
    """
    计算时序一致性指标
    
    Args:
        predictions: 连续帧的预测结果列表
        
    Returns:
        时序一致性指标字典
    """
    if len(predictions) < 2:
        return {'temporal_consistency': 1.0}
    
    metrics = {}
    consistency_scores = []
    
    for i in range(len(predictions) - 1):
        pred1 = predictions[i]
        pred2 = predictions[i + 1]
        
        if isinstance(pred1, torch.Tensor):
            pred1 = pred1.detach().cpu().numpy()
        if isinstance(pred2, torch.Tensor):
            pred2 = pred2.detach().cpu().numpy()
        
        # 计算相邻预测的相似度
        if pred1.shape == pred2.shape:
            # 使用余弦相似度
            similarity = compute_cosine_similarity(pred1.flatten(), pred2.flatten())
            consistency_scores.append(similarity)
    
    metrics['temporal_consistency'] = np.mean(consistency_scores)
    metrics['temporal_consistency_std'] = np.std(consistency_scores)
    
    return metrics


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_diversity_metrics(trajectories: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    计算轨迹多样性指标
    
    Args:
        trajectories: 多个预测轨迹 (B, N, T, 2)，其中N是每个样本的轨迹数量
        
    Returns:
        多样性指标字典
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    
    metrics = {}
    
    if trajectories.ndim != 4:
        warnings.warn("Expected 4D tensor (B, N, T, 2) for diversity computation")
        return metrics
    
    diversity_scores = []
    
    for batch_trajs in trajectories:  # 遍历每个batch
        # 计算轨迹间的平均距离
        n_trajs = batch_trajs.shape[0]
        distances = []
        
        for i in range(n_trajs):
            for j in range(i + 1, n_trajs):
                # 计算两条轨迹的平均距离
                traj_distance = np.mean(np.linalg.norm(batch_trajs[i] - batch_trajs[j], axis=1))
                distances.append(traj_distance)
        
        if distances:
            diversity_scores.append(np.mean(distances))
    
    metrics['trajectory_diversity'] = np.mean(diversity_scores)
    metrics['diversity_std'] = np.std(diversity_scores)
    
    return metrics


def format_metrics(metrics: Dict[str, float], 
                  precision: int = 4,
                  percentage_keys: List[str] = None) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
        precision: 小数精度
        percentage_keys: 需要显示为百分比的键
        
    Returns:
        格式化的字符串
    """
    if percentage_keys is None:
        percentage_keys = ['accuracy', 'precision', 'recall', 'f1', 'miss_rate', 
                          'collision_rate', 'safety_rate', 'temporal_consistency']
    
    formatted_lines = []
    
    for key, value in sorted(metrics.items()):
        if key in percentage_keys:
            formatted_value = f"{value * 100:.{precision-2}f}%"
        else:
            formatted_value = f"{value:.{precision}f}"
        
        formatted_lines.append(f"{key}: {formatted_value}")
    
    return " | ".join(formatted_lines)


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    聚合多个指标字典
    
    Args:
        metrics_list: 指标字典列表
        
    Returns:
        聚合后的指标字典，包含mean, std, min, max
    """
    if not metrics_list:
        return {}
    
    # 收集所有键
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in metrics_list if key in metrics]
        
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    return aggregated


def compute_statistical_significance(metrics1: List[float], 
                                   metrics2: List[float],
                                   test_type: str = 'ttest') -> Dict[str, float]:
    """
    计算统计显著性
    
    Args:
        metrics1: 第一组指标
        metrics2: 第二组指标
        test_type: 统计检验类型 ('ttest', 'wilcoxon')
        
    Returns:
        统计检验结果
    """
    try:
        from scipy import stats
        
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(metrics1, metrics2)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(metrics1, metrics2)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    except ImportError:
        warnings.warn("scipy not available for statistical tests")
        return {'error': 'scipy not available'}


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics_history = {}
        self.best_metrics = {}
        self.best_epoch = {}
    
    def update(self, metrics: Dict[str, float], epoch: int = None):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            self.metrics_history[key].append(value)
            
            # 更新最佳指标（假设越大越好，对于loss等指标需要特殊处理）
            if key not in self.best_metrics:
                self.best_metrics[key] = value
                self.best_epoch[key] = epoch
            else:
                # 对于包含'loss'或'error'的指标，越小越好
                if 'loss' in key.lower() or 'error' in key.lower():
                    if value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self.best_epoch[key] = epoch
                else:
                    if value > self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self.best_epoch[key] = epoch
    
    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取最佳指标"""
        result = {}
        for key in self.best_metrics:
            result[key] = {
                'value': self.best_metrics[key],
                'epoch': self.best_epoch[key]
            }
        return result
    
    def get_recent_average(self, window_size: int = 10) -> Dict[str, float]:
        """获取最近几个epoch的平均指标"""
        recent_metrics = {}
        for key, values in self.metrics_history.items():
            if len(values) >= window_size:
                recent_metrics[key] = np.mean(values[-window_size:])
            else:
                recent_metrics[key] = np.mean(values)
        return recent_metrics
    
    def get_improvement_rate(self, window_size: int = 10) -> Dict[str, float]:
        """计算指标改善率"""
        improvement_rates = {}
        
        for key, values in self.metrics_history.items():
            if len(values) < window_size * 2:
                improvement_rates[key] = 0.0
                continue
            
            old_avg = np.mean(values[-window_size*2:-window_size])
            new_avg = np.mean(values[-window_size:])
            
            if old_avg != 0:
                improvement_rate = (new_avg - old_avg) / abs(old_avg)
            else:
                improvement_rate = 0.0
            
            improvement_rates[key] = improvement_rate
        
        return improvement_rates
