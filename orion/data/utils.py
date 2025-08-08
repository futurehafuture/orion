"""数据工具函数"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random

from .datasets import DrivingDataset, ToyDrivingDataset, CarlaDataset
from .transforms import DrivingTransforms, MultiModalDataCollator


def create_dataloader(dataset_type: str,
                     data_path: str,
                     split: str = "train",
                     batch_size: int = 8,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     shuffle: Optional[bool] = None,
                     **dataset_kwargs) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset_type: 数据集类型 ('toy', 'carla', 'nuscenes')
        data_path: 数据路径
        split: 数据划分
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        shuffle: 是否打乱数据
        **dataset_kwargs: 数据集额外参数
        
    Returns:
        DataLoader对象
    """
    # 默认设置
    if shuffle is None:
        shuffle = (split == "train")
    
    # 创建数据变换
    is_training = (split == "train")
    transform = DrivingTransforms(is_training=is_training)
    
    # 创建数据集
    if dataset_type.lower() == "toy":
        dataset = ToyDrivingDataset(
            transform=transform,
            **dataset_kwargs
        )
    elif dataset_type.lower() == "carla":
        dataset = CarlaDataset(
            data_path=data_path,
            split=split,
            transform=transform,
            **dataset_kwargs
        )
    elif dataset_type.lower() == "nuscenes":
        from .datasets import NuScenesDataset
        dataset = NuScenesDataset(
            data_path=data_path,
            split=split,
            transform=transform,
            **dataset_kwargs
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # 创建数据整理器
    collate_fn = MultiModalDataCollator()
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=is_training  # 训练时丢弃最后一个不完整批次
    )
    
    return dataloader


def split_dataset(dataset: DrivingDataset, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 seed: int = 42) -> Tuple[DrivingDataset, DrivingDataset, DrivingDataset]:
    """
    划分数据集
    
    Args:
        dataset: 原始数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # 设置随机种子
    torch.manual_seed(seed)
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    自定义的批次整理函数
    
    Args:
        batch: 样本列表
        
    Returns:
        整理后的批次数据
    """
    collator = MultiModalDataCollator()
    return collator(batch)


def compute_dataset_statistics(dataloader: DataLoader) -> Dict[str, Any]:
    """
    计算数据集统计信息
    
    Args:
        dataloader: 数据加载器
        
    Returns:
        统计信息字典
    """
    stats = {
        'num_samples': 0,
        'image_stats': {'mean': None, 'std': None},
        'trajectory_stats': {'mean': None, 'std': None, 'lengths': []},
        'label_distributions': {}
    }
    
    # 收集统计信息
    all_images = []
    all_trajectories = []
    label_counts = {}
    
    for batch in dataloader:
        stats['num_samples'] += batch['image'].size(0)
        
        # 图像统计
        images = batch['image']  # (B, C, H, W)
        all_images.append(images.view(images.size(0), images.size(1), -1))
        
        # 轨迹统计
        if 'trajectory' in batch:
            trajectories = batch['trajectory']  # (B, T, 2)
            all_trajectories.append(trajectories.view(trajectories.size(0), -1))
            stats['trajectory_stats']['lengths'].extend([trajectories.size(1)] * trajectories.size(0))
        
        # 标签分布
        for key, value in batch.items():
            if 'labels' in key and isinstance(value, torch.Tensor):
                if key not in label_counts:
                    label_counts[key] = {}
                
                unique, counts = torch.unique(value, return_counts=True)
                for label, count in zip(unique.tolist(), counts.tolist()):
                    label_counts[key][label] = label_counts[key].get(label, 0) + count
    
    # 计算图像统计
    if all_images:
        all_images = torch.cat(all_images, dim=0)  # (N, C, H*W)
        stats['image_stats']['mean'] = all_images.mean(dim=[0, 2]).tolist()
        stats['image_stats']['std'] = all_images.std(dim=[0, 2]).tolist()
    
    # 计算轨迹统计
    if all_trajectories:
        all_trajectories = torch.cat(all_trajectories, dim=0)  # (N, T*2)
        stats['trajectory_stats']['mean'] = all_trajectories.mean(dim=0).tolist()
        stats['trajectory_stats']['std'] = all_trajectories.std(dim=0).tolist()
    
    # 标签分布
    stats['label_distributions'] = label_counts
    
    return stats


def visualize_batch(batch: Dict[str, torch.Tensor], 
                   save_path: str = "batch_visualization.png",
                   num_samples: int = 4):
    """
    可视化批次数据
    
    Args:
        batch: 批次数据
        save_path: 保存路径
        num_samples: 可视化的样本数量
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    num_samples = min(num_samples, batch['image'].size(0))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # 图像可视化
        image = batch['image'][i]  # (C, H, W)
        if image.max() <= 1.0:
            # 反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
        
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # 轨迹可视化
        if 'trajectory' in batch:
            trajectory = batch['trajectory'][i].numpy()  # (T, 2)
            axes[1, i].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
            axes[1, i].scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
            axes[1, i].scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
            axes[1, i].set_title(f'Trajectory {i+1}')
            axes[1, i].legend()
            axes[1, i].grid(True)
            axes[1, i].axis('equal')
        else:
            axes[1, i].text(0.5, 0.5, 'No Trajectory', 
                          ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'No Trajectory {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_weighted_sampler(dataset: DrivingDataset, 
                          label_key: str = 'scene_classification_labels') -> torch.utils.data.WeightedRandomSampler:
    """
    创建加权采样器以平衡类别
    
    Args:
        dataset: 数据集
        label_key: 用于平衡的标签键
        
    Returns:
        WeightedRandomSampler对象
    """
    # 收集所有标签
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if label_key in sample:
            labels.append(sample[label_key].item())
    
    # 计算类别权重
    unique_labels = list(set(labels))
    label_counts = {label: labels.count(label) for label in unique_labels}
    total_samples = len(labels)
    
    # 计算每个样本的权重
    weights = []
    for label in labels:
        weight = total_samples / (len(unique_labels) * label_counts[label])
        weights.append(weight)
    
    # 创建加权采样器
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def setup_reproducible_dataloader(dataloader: DataLoader, seed: int = 42):
    """
    设置可重现的数据加载器
    
    Args:
        dataloader: 数据加载器
        seed: 随机种子
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # 设置worker初始化函数
    dataloader.worker_init_fn = worker_init_fn
    
    # 设置全局种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def analyze_trajectory_patterns(dataloader: DataLoader) -> Dict[str, Any]:
    """
    分析轨迹模式
    
    Args:
        dataloader: 数据加载器
        
    Returns:
        轨迹分析结果
    """
    trajectory_stats = {
        'total_length_distribution': [],
        'speed_distribution': [],
        'curvature_distribution': [],
        'trajectory_types': {'straight': 0, 'left_turn': 0, 'right_turn': 0, 'complex': 0}
    }
    
    for batch in dataloader:
        if 'trajectory' not in batch:
            continue
        
        trajectories = batch['trajectory'].numpy()  # (B, T, 2)
        
        for traj in trajectories:
            # 总长度
            distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            total_length = np.sum(distances)
            trajectory_stats['total_length_distribution'].append(total_length)
            
            # 平均速度
            avg_speed = total_length / len(traj)
            trajectory_stats['speed_distribution'].append(avg_speed)
            
            # 曲率分析
            if len(traj) > 2:
                # 计算转向角
                vectors = np.diff(traj, axis=0)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                angle_changes = np.abs(np.diff(angles))
                # 处理角度跳跃
                angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
                avg_curvature = np.mean(angle_changes)
                trajectory_stats['curvature_distribution'].append(avg_curvature)
                
                # 轨迹类型分类
                total_angle_change = np.sum(angle_changes)
                if total_angle_change < 0.2:
                    trajectory_stats['trajectory_types']['straight'] += 1
                elif np.sum(np.diff(angles) > 0) > np.sum(np.diff(angles) < 0):
                    trajectory_stats['trajectory_types']['left_turn'] += 1
                elif np.sum(np.diff(angles) < 0) > np.sum(np.diff(angles) > 0):
                    trajectory_stats['trajectory_types']['right_turn'] += 1
                else:
                    trajectory_stats['trajectory_types']['complex'] += 1
    
    # 计算统计量
    for key in ['total_length_distribution', 'speed_distribution', 'curvature_distribution']:
        if trajectory_stats[key]:
            values = np.array(trajectory_stats[key])
            trajectory_stats[f'{key}_stats'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return trajectory_stats
