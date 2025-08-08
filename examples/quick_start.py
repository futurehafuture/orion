#!/usr/bin/env python3
"""ORION快速开始示例"""

import torch
from orion import OrionConfig, OrionSystem, OrionTrainer
from orion.data import create_dataloader


def main():
    """快速开始示例"""
    print("ORION Quick Start Example")
    print("=" * 40)
    
    # 1. 创建配置
    config = OrionConfig()
    config.training.epochs = 2
    config.training.batch_size = 4
    config.dataset_type = "toy"
    
    print("✓ Configuration created")
    
    # 2. 创建模型
    model = OrionSystem(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. 创建数据加载器
    train_loader = create_dataloader(
        dataset_type="toy",
        data_path="",
        split="train",
        batch_size=config.training.batch_size,
        length=100  # 只用100个样本进行快速演示
    )
    print(f"✓ Data loader created with {len(train_loader)} batches")
    
    # 4. 创建训练器
    trainer = OrionTrainer(config, model)
    print("✓ Trainer created")
    
    # 5. 快速训练
    print("\nStarting quick training...")
    try:
        trainer.train(train_loader)
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # 6. 快速测试
    print("\nTesting model...")
    model.eval()
    
    # 获取一个测试样本
    test_batch = next(iter(train_loader))
    
    with torch.no_grad():
        # 预测
        predictions = model.predict(test_batch['image'][:1])  # 只取第一个样本
        
        print("✓ Model prediction successful!")
        print(f"  - Generated {predictions['trajectories'].shape[1]} trajectory candidates")
        print(f"  - VQA predictions: {len(predictions['vqa_predictions'])} tasks")
        
        # 决策解释
        explanations = model.explain_decision(test_batch['image'][:1])
        print(f"  - Decision explanations available: {list(explanations.keys())}")
    
    print("\n" + "=" * 40)
    print("Quick start completed successfully! 🎉")
    print("\nNext steps:")
    print("1. Try training on your own dataset")
    print("2. Experiment with different model configurations")
    print("3. Explore the visualization tools")
    print("4. Check out the demo script for interactive usage")


if __name__ == "__main__":
    main()
