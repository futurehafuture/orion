#!/usr/bin/env python3
"""测试重构后的ORION架构"""

import sys
import traceback
import torch


def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        # 测试核心模块
        from orion import OrionConfig, OrionSystem, OrionTrainer
        print("✓ Core modules imported successfully")
        
        # 测试配置模块
        from orion.config import VisionConfig, QTFormerConfig, LLMConfig, PlannerConfig
        print("✓ Config modules imported successfully")
        
        # 测试模型模块
        from orion.models import VisionBackbone, QTFormer, LLMInterface, ConditionalTrajectoryVAE
        print("✓ Model modules imported successfully")
        
        # 测试数据模块
        from orion.data import ToyDrivingDataset, DrivingTransforms, create_dataloader
        print("✓ Data modules imported successfully")
        
        # 测试训练模块
        from orion.training import OrionLoss, OrionMetrics
        print("✓ Training modules imported successfully")
        
        # 测试工具模块
        from orion.utils import setup_logging, TrajectoryVisualizer, save_checkpoint
        print("✓ Utils modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_config_creation():
    """测试配置创建"""
    print("\nTesting config creation...")
    
    try:
        from orion.config import OrionConfig
        
        # 测试默认配置
        config = OrionConfig()
        print("✓ Default config created")
        
        # 测试配置验证
        config.validate()
        print("✓ Config validation passed")
        
        # 测试配置保存和加载
        config.to_yaml("test_config.yaml")
        loaded_config = OrionConfig.from_yaml("test_config.yaml")
        print("✓ Config save/load works")
        
        # 清理
        import os
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    
    try:
        from orion import OrionConfig, OrionSystem
        
        # 创建小型配置用于测试
        config = OrionConfig()
        config.vision.output_dim = 128
        config.qt_former.vision_dim = 128
        config.qt_former.token_dim = 128
        config.llm.token_dim = 128
        config.planner.token_dim = 128
        config.planner.traj_len = 10
        
        # 创建模型
        model = OrionSystem(config)
        print("✓ Model created successfully")
        
        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\nTesting data loading...")
    
    try:
        from orion.data import ToyDrivingDataset, create_dataloader
        
        # 创建玩具数据集
        dataset = ToyDrivingDataset(length=10, seed=42)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # 测试单个样本
        sample = dataset[0]
        print(f"✓ Sample loaded: image {sample['image'].shape}, trajectory {sample['trajectory'].shape}")
        
        # 创建数据加载器
        dataloader = create_dataloader(
            dataset_type="toy",
            data_path="",
            batch_size=2,
            num_workers=0,
            length=10
        )
        print("✓ DataLoader created")
        
        # 测试批次加载
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded: {batch['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\nTesting forward pass...")
    
    try:
        from orion import OrionConfig, OrionSystem
        from orion.data import ToyDrivingDataset, create_dataloader
        
        # 小型配置
        config = OrionConfig()
        config.vision.output_dim = 64
        config.qt_former.vision_dim = 64
        config.qt_former.token_dim = 64
        config.llm.token_dim = 64
        config.planner.token_dim = 64
        config.planner.traj_len = 5
        
        # 创建模型和数据
        model = OrionSystem(config)
        model.eval()
        
        dataloader = create_dataloader(
            dataset_type="toy",
            data_path="",
            batch_size=2,
            num_workers=0,
            length=5,
            trajectory_length=5,
            image_size=(3, 64, 64)
        )
        
        batch = next(iter(dataloader))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(batch, training=False)
            print("✓ Forward pass completed")
            
            # 测试预测
            predictions = model.predict(batch['image'])
            print("✓ Prediction completed")
            print(f"  Generated {predictions['trajectories'].shape[1]} trajectories")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        traceback.print_exc()
        return False


def test_training_components():
    """测试训练组件"""
    print("\nTesting training components...")
    
    try:
        from orion.training import OrionLoss, OrionMetrics
        from orion.config.model_configs import TrainingConfig
        
        # 创建损失函数
        training_config = TrainingConfig()
        loss_fn = OrionLoss(training_config)
        print("✓ Loss function created")
        
        # 创建评估指标
        metrics = OrionMetrics()
        print("✓ Metrics created")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """测试可视化工具"""
    print("\nTesting visualization...")
    
    try:
        from orion.utils import TrajectoryVisualizer
        import numpy as np
        
        # 创建可视化器
        visualizer = TrajectoryVisualizer()
        print("✓ Visualizer created")
        
        # 创建测试轨迹
        trajectory = np.random.randn(20, 2).cumsum(axis=0)
        
        # 测试可视化（不保存）
        fig = visualizer.plot_single_trajectory(trajectory, show_velocity=False)
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("✓ Trajectory visualization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("ORION Architecture Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_creation,
        test_model_creation,
        test_data_loading,
        test_forward_pass,
        test_training_components,
        test_visualization,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
