"""数据变换和增强"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional, Any
import cv2
import random


class DrivingTransforms:
    """
    自动驾驶数据变换类
    
    包含图像变换、轨迹变换和数据增强
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 is_training: bool = True,
                 augment_probability: float = 0.5):
        self.image_size = image_size
        self.is_training = is_training
        self.augment_prob = augment_probability
        
        # 图像预处理
        self.image_transform = self._build_image_transform()
        
        # 图像增强
        if is_training:
            self.image_augment = self._build_image_augment()
        else:
            self.image_augment = None
    
    def _build_image_transform(self) -> T.Compose:
        """构建图像变换流水线"""
        transforms = [
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return T.Compose(transforms)
    
    def _build_image_augment(self) -> T.Compose:
        """构建图像增强流水线"""
        transforms = [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
        return T.Compose(transforms)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """应用变换"""
        # 图像变换
        if 'image' in sample:
            if isinstance(sample['image'], Image.Image):
                # 增强（训练时）
                if self.is_training and self.image_augment and random.random() < self.augment_prob:
                    sample['image'] = self.image_augment(sample['image'])
                
                # 基础变换
                sample['image'] = self.image_transform(sample['image'])
            elif isinstance(sample['image'], torch.Tensor):
                # 已经是tensor，可能需要重新归一化
                if sample['image'].max() > 1.0:
                    sample['image'] = sample['image'] / 255.0
        
        # 轨迹变换
        if 'trajectory' in sample and self.is_training:
            sample['trajectory'] = self.transform_trajectory(sample['trajectory'])
        
        return sample
    
    def transform_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """轨迹变换和增强"""
        if not self.is_training:
            return trajectory
        
        # 转换为numpy进行处理
        if isinstance(trajectory, torch.Tensor):
            traj_np = trajectory.numpy()
        else:
            traj_np = trajectory
        
        # 随机应用变换
        if random.random() < self.augment_prob:
            traj_np = self._augment_trajectory(traj_np)
        
        # 转换回tensor
        return torch.from_numpy(traj_np.astype(np.float32))
    
    def _augment_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """轨迹增强"""
        augmented = trajectory.copy()
        
        # 1. 随机旋转
        if random.random() < 0.3:
            angle = random.uniform(-np.pi/12, np.pi/12)  # ±15度
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            center = augmented.mean(axis=0)
            augmented = (augmented - center) @ rotation_matrix.T + center
        
        # 2. 随机缩放
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            center = augmented.mean(axis=0)
            augmented = (augmented - center) * scale + center
        
        # 3. 随机平移
        if random.random() < 0.3:
            translation = np.random.normal(0, 0.5, size=2)
            augmented += translation
        
        # 4. 添加噪声
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.1, size=augmented.shape)
            augmented += noise
        
        # 5. 时间采样变化
        if random.random() < 0.2:
            augmented = self._resample_trajectory(augmented)
        
        return augmented
    
    def _resample_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """重新采样轨迹（改变时间分辨率）"""
        T, D = trajectory.shape
        
        # 随机选择新的时间步数（±20%）
        new_T = int(T * random.uniform(0.8, 1.2))
        new_T = max(5, min(new_T, T * 2))  # 限制范围
        
        # 线性插值重采样
        old_indices = np.linspace(0, T-1, new_T)
        new_trajectory = np.zeros((T, D))
        
        for d in range(D):
            new_trajectory[:, d] = np.interp(
                np.arange(T), 
                old_indices * (T-1) / (new_T-1),
                np.interp(old_indices, np.arange(T), trajectory[:, d])
            )
        
        return new_trajectory


class WeatherAugmentation:
    """天气条件增强"""
    
    def __init__(self):
        self.weather_effects = {
            'rain': self._apply_rain_effect,
            'fog': self._apply_fog_effect,
            'snow': self._apply_snow_effect,
            'sun_glare': self._apply_sun_glare,
            'night': self._apply_night_effect
        }
    
    def __call__(self, image: torch.Tensor, weather_type: str = None) -> torch.Tensor:
        """应用天气效果"""
        if weather_type is None:
            weather_type = random.choice(list(self.weather_effects.keys()))
        
        if weather_type in self.weather_effects:
            return self.weather_effects[weather_type](image)
        else:
            return image
    
    def _apply_rain_effect(self, image: torch.Tensor) -> torch.Tensor:
        """雨天效果"""
        # 降低亮度和对比度
        image = image * 0.7 + 0.1
        
        # 添加雨滴噪声
        if random.random() < 0.5:
            rain_noise = torch.randn_like(image) * 0.05
            rain_mask = (torch.rand_like(image) < 0.01).float()
            image = image + rain_noise * rain_mask
        
        return torch.clamp(image, 0, 1)
    
    def _apply_fog_effect(self, image: torch.Tensor) -> torch.Tensor:
        """雾天效果"""
        # 添加白色雾气
        fog_intensity = random.uniform(0.2, 0.6)
        fog = torch.ones_like(image) * fog_intensity
        image = image * (1 - fog_intensity) + fog
        
        return torch.clamp(image, 0, 1)
    
    def _apply_snow_effect(self, image: torch.Tensor) -> torch.Tensor:
        """雪天效果"""
        # 增加亮度，添加雪花
        image = image * 0.8 + 0.2
        
        # 添加雪花点
        if random.random() < 0.7:
            snow_mask = (torch.rand_like(image[0:1]) < 0.005).float()
            snow_mask = snow_mask.expand_as(image)
            image = torch.where(snow_mask > 0, torch.ones_like(image), image)
        
        return torch.clamp(image, 0, 1)
    
    def _apply_sun_glare(self, image: torch.Tensor) -> torch.Tensor:
        """阳光眩光效果"""
        C, H, W = image.shape
        
        # 创建径向眩光
        center_x, center_y = random.randint(W//4, 3*W//4), random.randint(H//4, H//2)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        distance = torch.sqrt((x - center_x).float()**2 + (y - center_y).float()**2)
        
        # 眩光强度随距离衰减
        glare_radius = random.uniform(30, 80)
        glare_intensity = torch.exp(-distance / glare_radius) * random.uniform(0.3, 0.8)
        glare_mask = glare_intensity.unsqueeze(0).expand(C, -1, -1)
        
        image = image + glare_mask
        return torch.clamp(image, 0, 1)
    
    def _apply_night_effect(self, image: torch.Tensor) -> torch.Tensor:
        """夜晚效果"""
        # 大幅降低亮度
        image = image * 0.3
        
        # 添加蓝色色调
        image[2] = torch.min(image[2] + 0.1, torch.ones_like(image[2]))
        
        # 模拟车灯（在图像下方添加亮点）
        if random.random() < 0.6:
            C, H, W = image.shape
            num_lights = random.randint(1, 3)
            
            for _ in range(num_lights):
                light_x = random.randint(W//4, 3*W//4)
                light_y = random.randint(2*H//3, H-10)
                light_size = random.randint(5, 15)
                
                # 创建光晕效果
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                distance = torch.sqrt((x - light_x).float()**2 + (y - light_y).float()**2)
                light_mask = torch.exp(-distance / light_size) * 0.8
                
                # 添加黄色车灯
                image[0] += light_mask  # R
                image[1] += light_mask  # G
                image[2] += light_mask * 0.3  # B (less blue for yellow light)
        
        return torch.clamp(image, 0, 1)


class TrajectoryNormalization:
    """轨迹归一化"""
    
    def __init__(self, method: str = 'relative'):
        """
        Args:
            method: 归一化方法
                - 'relative': 相对于起始点的位移
                - 'absolute': 绝对坐标归一化
                - 'velocity': 速度表示
        """
        self.method = method
    
    def normalize(self, trajectory: np.ndarray) -> np.ndarray:
        """归一化轨迹"""
        if self.method == 'relative':
            return self._relative_normalization(trajectory)
        elif self.method == 'absolute':
            return self._absolute_normalization(trajectory)
        elif self.method == 'velocity':
            return self._velocity_normalization(trajectory)
        else:
            return trajectory
    
    def denormalize(self, trajectory: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """反归一化轨迹"""
        if self.method == 'relative' and reference is not None:
            return trajectory + reference[0:1]  # 加上起始点
        elif self.method == 'velocity' and reference is not None:
            return self._integrate_velocity(trajectory, reference[0])
        else:
            return trajectory
    
    def _relative_normalization(self, trajectory: np.ndarray) -> np.ndarray:
        """相对归一化：减去起始点"""
        return trajectory - trajectory[0:1]
    
    def _absolute_normalization(self, trajectory: np.ndarray) -> np.ndarray:
        """绝对归一化：缩放到[-1, 1]"""
        min_vals = trajectory.min(axis=0)
        max_vals = trajectory.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # 避免除零
        
        normalized = 2 * (trajectory - min_vals) / range_vals - 1
        return normalized
    
    def _velocity_normalization(self, trajectory: np.ndarray) -> np.ndarray:
        """速度表示：计算相邻点的速度"""
        velocities = np.diff(trajectory, axis=0)
        # 第一个点的速度设为0
        velocities = np.vstack([np.zeros((1, trajectory.shape[1])), velocities])
        return velocities
    
    def _integrate_velocity(self, velocities: np.ndarray, start_point: np.ndarray) -> np.ndarray:
        """从速度积分得到轨迹"""
        trajectory = np.cumsum(velocities, axis=0) + start_point
        return trajectory


class MultiModalDataCollator:
    """多模态数据整理器"""
    
    def __init__(self, pad_trajectory: bool = True):
        self.pad_trajectory = pad_trajectory
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        collated = {}
        
        # 收集所有键
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())
        
        for key in all_keys:
            if key == 'metadata':
                # metadata保持为列表
                collated[key] = [sample.get(key, {}) for sample in batch]
            else:
                # 尝试堆叠tensor
                values = [sample[key] for sample in batch if key in sample]
                if values and isinstance(values[0], torch.Tensor):
                    if key == 'trajectory' and self.pad_trajectory:
                        # 轨迹可能有不同长度，需要填充
                        collated[key] = self._pad_trajectories(values)
                    else:
                        try:
                            collated[key] = torch.stack(values)
                        except RuntimeError:
                            # 如果无法堆叠，保持为列表
                            collated[key] = values
                else:
                    collated[key] = values
        
        return collated
    
    def _pad_trajectories(self, trajectories: List[torch.Tensor]) -> torch.Tensor:
        """填充轨迹到相同长度"""
        max_length = max(traj.size(0) for traj in trajectories)
        padded = []
        
        for traj in trajectories:
            if traj.size(0) < max_length:
                # 用最后一个点填充
                padding = traj[-1:].repeat(max_length - traj.size(0), 1)
                padded_traj = torch.cat([traj, padding], dim=0)
            else:
                padded_traj = traj[:max_length]
            padded.append(padded_traj)
        
        return torch.stack(padded)
