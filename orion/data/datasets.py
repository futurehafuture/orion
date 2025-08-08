"""数据集实现"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import json
import h5py
from PIL import Image
import random

from .transforms import DrivingTransforms


class DrivingDataset(Dataset):
    """
    通用自动驾驶数据集基类
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = "train",
                 sequence_length: int = 8,
                 transform: Optional[DrivingTransforms] = None,
                 load_vqa: bool = True,
                 load_trajectory: bool = True):
        self.data_path = data_path
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.load_vqa = load_vqa
        self.load_trajectory = load_trajectory
        
        # 加载数据索引
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载数据样本索引"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        raise NotImplementedError


class ToyDrivingDataset(DrivingDataset):
    """
    合成玩具数据集，用于快速测试和调试
    """
    
    def __init__(self, 
                 length: int = 10000,
                 image_size: Tuple[int, int, int] = (3, 224, 224),
                 trajectory_length: int = 20,
                 vqa_classes: int = 16,
                 seed: int = 42,
                 **kwargs):
        self.length = length
        self.image_size = image_size
        self.trajectory_length = trajectory_length
        self.vqa_classes = vqa_classes
        
        # 设置随机种子
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        
        # 预生成一些基础模式
        self._generate_base_patterns()
        
        # 不调用父类__init__，因为我们不需要加载真实数据
        
    def _generate_base_patterns(self):
        """生成基础的轨迹和场景模式"""
        # 轨迹模式：直线、左转、右转、U转等
        self.trajectory_patterns = {
            'straight': lambda t: np.stack([t * 10, np.zeros_like(t)], axis=1),
            'left_turn': lambda t: np.stack([
                5 * np.sin(t * np.pi), 
                5 * (1 - np.cos(t * np.pi))
            ], axis=1),
            'right_turn': lambda t: np.stack([
                5 * np.sin(t * np.pi), 
                -5 * (1 - np.cos(t * np.pi))
            ], axis=1),
            'curve': lambda t: np.stack([
                t * 8, 
                2 * np.sin(t * 2 * np.pi)
            ], axis=1),
            'stop': lambda t: np.stack([
                t * 5 * np.exp(-t * 3), 
                np.zeros_like(t)
            ], axis=1)
        }
        
        # 场景类型
        self.scene_types = [
            'urban', 'highway', 'rural', 'parking', 
            'intersection', 'tunnel', 'bridge', 'residential'
        ]
        
        # 天气类型
        self.weather_types = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy']
        
        # 交通灯状态
        self.traffic_light_states = ['red', 'yellow', 'green', 'none']
        
        # 驾驶意图
        self.driving_intents = [
            'go_straight', 'turn_left', 'turn_right', 
            'stop', 'park', 'change_lane'
        ]
    
    def _generate_trajectory(self, pattern_name: str) -> np.ndarray:
        """生成轨迹"""
        t = np.linspace(0, 1, self.trajectory_length)
        
        # 添加噪声
        noise_scale = 0.1
        noise = self.rng.normal(0, noise_scale, (self.trajectory_length, 2))
        
        # 生成基础轨迹
        base_trajectory = self.trajectory_patterns[pattern_name](t)
        
        # 添加随机变换
        angle = self.rng.uniform(-np.pi/6, np.pi/6)  # ±30度旋转
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 应用旋转和平移
        trajectory = base_trajectory @ rotation_matrix.T
        trajectory += self.rng.uniform(-2, 2, (1, 2))  # 随机平移
        trajectory += noise
        
        return trajectory.astype(np.float32)
    
    def _generate_image(self, scene_type: str, weather: str) -> np.ndarray:
        """生成合成图像"""
        C, H, W = self.image_size
        
        # 基础背景颜色（根据天气和场景）
        if weather == 'sunny':
            base_color = [0.7, 0.8, 0.9]
        elif weather == 'cloudy':
            base_color = [0.5, 0.6, 0.7]
        elif weather == 'rainy':
            base_color = [0.3, 0.4, 0.5]
        elif weather == 'snowy':
            base_color = [0.8, 0.8, 0.9]
        else:  # foggy
            base_color = [0.6, 0.6, 0.6]
        
        # 生成基础图像
        image = np.ones((H, W, C), dtype=np.float32)
        for i in range(C):
            image[:, :, i] = base_color[i]
        
        # 添加纹理和噪声
        noise = self.rng.normal(0, 0.1, (H, W, C))
        image += noise
        
        # 添加简单的几何形状（模拟道路、车辆等）
        if scene_type in ['urban', 'intersection']:
            # 添加道路线条
            cv2.line(image, (0, H//2), (W, H//2), (0.2, 0.2, 0.2), 5)
            cv2.line(image, (W//2, 0), (W//2, H), (0.2, 0.2, 0.2), 3)
        
        elif scene_type == 'highway':
            # 添加高速公路标线
            for y in [H//3, 2*H//3]:
                cv2.line(image, (0, y), (W, y), (0.9, 0.9, 0.9), 2)
        
        # 添加随机对象（模拟车辆、行人等）
        num_objects = self.rng.randint(0, 5)
        for _ in range(num_objects):
            x, y = self.rng.randint(10, W-10), self.rng.randint(10, H-10)
            size = self.rng.randint(5, 20)
            color = self.rng.uniform(0, 1, 3)
            cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
        
        # 裁剪到[0, 1]范围
        image = np.clip(image, 0, 1)
        
        # 转换为CHW格式
        image = image.transpose(2, 0, 1)
        
        return image
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """生成单个样本"""
        # 随机选择模式
        trajectory_pattern = self.rng.choice(list(self.trajectory_patterns.keys()))
        scene_type = self.rng.choice(self.scene_types)
        weather = self.rng.choice(self.weather_types)
        traffic_light = self.rng.choice(self.traffic_light_states)
        driving_intent = self.rng.choice(self.driving_intents)
        
        # 生成数据
        image = self._generate_image(scene_type, weather)
        trajectory = self._generate_trajectory(trajectory_pattern)
        
        # VQA标签
        scene_label = self.scene_types.index(scene_type)
        weather_label = self.weather_types.index(weather)
        traffic_light_label = self.traffic_light_states.index(traffic_light)
        driving_intent_label = self.driving_intents.index(driving_intent)
        
        # 组装样本
        sample = {
            'image': torch.from_numpy(image),
            'trajectory': torch.from_numpy(trajectory),
            'scene_classification_labels': torch.tensor(scene_label, dtype=torch.long),
            'weather_detection_labels': torch.tensor(weather_label, dtype=torch.long),
            'traffic_light_labels': torch.tensor(traffic_light_label, dtype=torch.long),
            'driving_intent_labels': torch.tensor(driving_intent_label, dtype=torch.long),
            'vqa_labels': torch.tensor(scene_label % self.vqa_classes, dtype=torch.long),  # 通用VQA标签
            'metadata': {
                'trajectory_pattern': trajectory_pattern,
                'scene_type': scene_type,
                'weather': weather,
                'traffic_light': traffic_light,
                'driving_intent': driving_intent
            }
        }
        
        return sample


class CarlaDataset(DrivingDataset):
    """
    CARLA仿真数据集
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = "train",
                 towns: Optional[List[str]] = None,
                 weather_conditions: Optional[List[str]] = None,
                 **kwargs):
        self.towns = towns or ['Town01', 'Town02', 'Town03']
        self.weather_conditions = weather_conditions or ['sunny', 'cloudy', 'rainy']
        
        super().__init__(data_path, split, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载CARLA数据样本"""
        samples = []
        
        split_file = os.path.join(self.data_path, f"{self.split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        for episode in split_data['episodes']:
            episode_path = os.path.join(self.data_path, episode['path'])
            
            # 检查文件是否存在
            if not os.path.exists(episode_path):
                continue
            
            # 加载episode metadata
            metadata_file = os.path.join(episode_path, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # 过滤town和天气条件
            if metadata['town'] not in self.towns:
                continue
            if metadata['weather'] not in self.weather_conditions:
                continue
            
            # 添加样本
            samples.append({
                'episode_path': episode_path,
                'metadata': metadata,
                'num_frames': metadata['num_frames']
            })
        
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取CARLA样本"""
        sample_info = self.samples[idx]
        episode_path = sample_info['episode_path']
        metadata = sample_info['metadata']
        
        # 随机选择起始帧
        max_start_frame = max(0, sample_info['num_frames'] - self.sequence_length)
        start_frame = random.randint(0, max_start_frame)
        
        # 加载图像序列
        images = []
        for i in range(self.sequence_length):
            frame_idx = start_frame + i
            img_path = os.path.join(episode_path, 'images', f'{frame_idx:06d}.png')
            
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform.image_transform(image)
                images.append(image)
            else:
                # 如果帧不存在，重复最后一帧
                if images:
                    images.append(images[-1])
                else:
                    # 创建空白图像
                    empty_img = torch.zeros(3, 224, 224)
                    images.append(empty_img)
        
        # 使用当前帧作为主图像
        current_image = images[-1] if images else torch.zeros(3, 224, 224)
        
        # 加载轨迹数据
        trajectory_file = os.path.join(episode_path, 'trajectories.h5')
        with h5py.File(trajectory_file, 'r') as f:
            trajectories = f['ego_future_trajectory'][start_frame:start_frame + self.sequence_length]
            # 使用当前帧的未来轨迹
            current_trajectory = trajectories[-1] if len(trajectories) > 0 else np.zeros((20, 2))
        
        # 加载VQA标签
        vqa_file = os.path.join(episode_path, 'vqa_labels.json')
        with open(vqa_file, 'r') as f:
            vqa_data = json.load(f)
        
        frame_vqa = vqa_data['frames'][min(start_frame + self.sequence_length - 1, len(vqa_data['frames']) - 1)]
        
        # 组装样本
        sample = {
            'image': current_image,
            'trajectory': torch.from_numpy(current_trajectory.astype(np.float32)),
            'scene_classification_labels': torch.tensor(frame_vqa.get('scene_type', 0), dtype=torch.long),
            'weather_detection_labels': torch.tensor(frame_vqa.get('weather', 0), dtype=torch.long),
            'traffic_light_labels': torch.tensor(frame_vqa.get('traffic_light', 3), dtype=torch.long),
            'driving_intent_labels': torch.tensor(frame_vqa.get('driving_intent', 0), dtype=torch.long),
            'vqa_labels': torch.tensor(frame_vqa.get('general_vqa', 0), dtype=torch.long),
            'metadata': {
                'episode_id': metadata['episode_id'],
                'town': metadata['town'],
                'weather': metadata['weather'],
                'start_frame': start_frame
            }
        }
        
        return sample


class NuScenesDataset(DrivingDataset):
    """
    nuScenes数据集
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = "train",
                 version: str = "v1.0-trainval",
                 **kwargs):
        self.version = version
        
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version=version, dataroot=data_path, verbose=False)
        except ImportError:
            raise ImportError("nuscenes-devkit is required for NuScenesDataset")
        
        super().__init__(data_path, split, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载nuScenes样本"""
        samples = []
        
        # 获取对应split的场景
        if self.split == 'train':
            scenes = [s for s in self.nusc.scene if s['name'] in self.nusc.list_scenes(train=True)]
        elif self.split == 'val':
            scenes = [s for s in self.nusc.scene if s['name'] in self.nusc.list_scenes(val=True)]
        else:
            scenes = self.nusc.scene
        
        for scene in scenes:
            # 获取场景中的所有样本
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                samples.append({
                    'sample_token': sample_token,
                    'scene_token': scene['token'],
                    'scene_name': scene['name']
                })
                sample_token = sample['next']
        
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取nuScenes样本"""
        sample_info = self.samples[idx]
        sample = self.nusc.get('sample', sample_info['sample_token'])
        
        # 获取前置摄像头图像
        cam_front_token = sample['data']['CAM_FRONT']
        cam_front = self.nusc.get('sample_data', cam_front_token)
        
        # 加载图像
        img_path = os.path.join(self.nusc.dataroot, cam_front['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform.image_transform(image)
        
        # 获取ego vehicle的未来轨迹
        trajectory = self._get_future_trajectory(sample)
        
        # 生成VQA标签（简化版本）
        scene_description = self._get_scene_description(sample)
        
        sample_dict = {
            'image': image,
            'trajectory': torch.from_numpy(trajectory.astype(np.float32)),
            'scene_classification_labels': torch.tensor(scene_description.get('scene_type', 0), dtype=torch.long),
            'weather_detection_labels': torch.tensor(scene_description.get('weather', 0), dtype=torch.long),
            'traffic_light_labels': torch.tensor(3, dtype=torch.long),  # 默认无交通灯
            'driving_intent_labels': torch.tensor(scene_description.get('driving_intent', 0), dtype=torch.long),
            'vqa_labels': torch.tensor(scene_description.get('scene_type', 0), dtype=torch.long),
            'metadata': {
                'sample_token': sample_info['sample_token'],
                'scene_name': sample_info['scene_name'],
                'timestamp': sample['timestamp']
            }
        }
        
        return sample_dict
    
    def _get_future_trajectory(self, sample: dict, horizon: float = 6.0) -> np.ndarray:
        """获取未来轨迹"""
        # 这里需要实现从nuScenes数据中提取未来轨迹的逻辑
        # 简化版本：返回零轨迹
        return np.zeros((20, 2), dtype=np.float32)
    
    def _get_scene_description(self, sample: dict) -> Dict[str, int]:
        """获取场景描述"""
        # 这里需要实现场景分析逻辑
        # 简化版本：返回默认值
        return {
            'scene_type': 0,  # urban
            'weather': 0,     # sunny
            'driving_intent': 0  # go_straight
        }
