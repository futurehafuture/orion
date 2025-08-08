#!/usr/bin/env python3
"""自定义数据集示例"""

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from orion.data import DrivingDataset, DrivingTransforms


class CustomDrivingDataset(DrivingDataset):
    """
    自定义驾驶数据集示例
    展示如何创建自己的数据集
    """
    
    def __init__(self, data_path: str, split: str = "train", **kwargs):
        # 不调用父类__init__，因为我们要自定义实现
        self.data_path = data_path
        self.split = split
        self.transform = kwargs.get('transform', DrivingTransforms())
        
        # 加载数据索引
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """加载数据样本索引"""
        # 这里应该根据您的数据格式实现
        # 示例：从文件列表或数据库加载
        samples = []
        
        # 假设数据结构：
        # data_path/
        #   ├── images/
        #   │   ├── 001.jpg
        #   │   └── 002.jpg
        #   ├── trajectories/
        #   │   ├── 001.npy
        #   │   └── 002.npy
        #   └── annotations/
        #       ├── 001.json
        #       └── 002.json
        
        import os
        import glob
        
        image_pattern = os.path.join(self.data_path, "images", "*.jpg")
        image_files = sorted(glob.glob(image_pattern))
        
        for img_file in image_files:
            basename = os.path.splitext(os.path.basename(img_file))[0]
            
            traj_file = os.path.join(self.data_path, "trajectories", f"{basename}.npy")
            ann_file = os.path.join(self.data_path, "annotations", f"{basename}.json")
            
            if os.path.exists(traj_file) and os.path.exists(ann_file):
                samples.append({
                    'image_path': img_file,
                    'trajectory_path': traj_file,
                    'annotation_path': ann_file,
                    'sample_id': basename
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample_info['image_path']).convert('RGB')
        
        # 加载轨迹
        trajectory = np.load(sample_info['trajectory_path']).astype(np.float32)
        trajectory = torch.from_numpy(trajectory)
        
        # 加载标注
        import json
        with open(sample_info['annotation_path'], 'r') as f:
            annotations = json.load(f)
        
        # 组装样本
        sample = {
            'image': image,
            'trajectory': trajectory,
            'scene_classification_labels': torch.tensor(
                annotations.get('scene_type', 0), dtype=torch.long
            ),
            'weather_detection_labels': torch.tensor(
                annotations.get('weather', 0), dtype=torch.long
            ),
            'traffic_light_labels': torch.tensor(
                annotations.get('traffic_light', 3), dtype=torch.long
            ),
            'driving_intent_labels': torch.tensor(
                annotations.get('driving_intent', 0), dtype=torch.long
            ),
            'vqa_labels': torch.tensor(
                annotations.get('scene_type', 0), dtype=torch.long
            ),
            'metadata': {
                'sample_id': sample_info['sample_id'],
                'image_path': sample_info['image_path']
            }
        }
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_sample_data(output_dir: str, num_samples: int = 50):
    """创建示例数据"""
    import os
    import json
    from PIL import Image
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    for i in range(num_samples):
        sample_id = f"{i:03d}"
        
        # 创建示例图像
        image = Image.new('RGB', (640, 480), color=(100, 150, 200))
        image_path = os.path.join(output_dir, "images", f"{sample_id}.jpg")
        image.save(image_path)
        
        # 创建示例轨迹
        t = np.linspace(0, 1, 20)
        trajectory = np.stack([
            t * 10 + np.random.normal(0, 0.1, len(t)),
            np.sin(t * np.pi) * 2 + np.random.normal(0, 0.1, len(t))
        ], axis=1)
        
        traj_path = os.path.join(output_dir, "trajectories", f"{sample_id}.npy")
        np.save(traj_path, trajectory)
        
        # 创建示例标注
        annotation = {
            'scene_type': np.random.randint(0, 8),
            'weather': np.random.randint(0, 5),
            'traffic_light': np.random.randint(0, 4),
            'driving_intent': np.random.randint(0, 6),
            'description': f"Sample driving scene {i+1}"
        }
        
        ann_path = os.path.join(output_dir, "annotations", f"{sample_id}.json")
        with open(ann_path, 'w') as f:
            json.dump(annotation, f, indent=2)
    
    print(f"Created {num_samples} sample data files in {output_dir}")


def main():
    """演示自定义数据集使用"""
    print("Custom Dataset Example")
    print("=" * 30)
    
    # 创建示例数据
    sample_data_dir = "sample_data"
    print("Creating sample data...")
    create_sample_data(sample_data_dir, num_samples=20)
    
    # 创建自定义数据集
    print("Creating custom dataset...")
    dataset = CustomDrivingDataset(
        data_path=sample_data_dir,
        split="train",
        transform=DrivingTransforms(is_training=True)
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # 测试数据加载
    print("Testing data loading...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i+1}:")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Trajectory shape: {sample['trajectory'].shape}")
        print(f"  - Scene type: {sample['scene_classification_labels'].item()}")
        print(f"  - Metadata: {sample['metadata']}")
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    from orion.data.utils import collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"DataLoader created with batch size 4")
    
    # 测试批次加载
    print("Testing batch loading...")
    batch = next(iter(dataloader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch trajectory shape: {batch['trajectory'].shape}")
    
    print("\nCustom dataset example completed! ✓")
    print("\nTo use with ORION:")
    print("1. Replace CustomDrivingDataset with your data loading logic")
    print("2. Ensure your dataset returns the required fields")
    print("3. Use create_dataloader() or DataLoader directly")


if __name__ == "__main__":
    main()
