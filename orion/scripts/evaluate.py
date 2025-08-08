#!/usr/bin/env python3
"""ORION评估脚本"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from orion.config import OrionConfig
from orion.models import OrionSystem
from orion.training import OrionTrainer, OrionMetrics
from orion.data import create_dataloader
from orion.utils import setup_logging, load_checkpoint, TrajectoryVisualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate ORION autonomous driving model")
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (if not in checkpoint)')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='data/',
                       help='Dataset path')
    parser.add_argument('--dataset-type', type=str, default='toy',
                       choices=['toy', 'carla', 'nuscenes'],
                       help='Dataset type')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 评估参数
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--num-trajectories', type=int, default=5,
                       help='Number of trajectories to generate per sample')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to file')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--num-vis-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--save-attention', action='store_true',
                       help='Save attention maps')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='evaluation_results/',
                       help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='evaluation',
                       help='Experiment name')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    # 其他参数
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path: str, config_path: str = None, device: torch.device = None):
    """加载模型和配置"""
    # 加载检查点
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    
    # 加载配置
    if config_path:
        config = OrionConfig.from_yaml(config_path)
    elif 'config' in checkpoint:
        config = checkpoint['config']
        if not isinstance(config, OrionConfig):
            # 如果配置是字典，转换为OrionConfig
            config = OrionConfig(**config)
    else:
        raise ValueError("No config found. Please provide --config or ensure config is in checkpoint")
    
    # 创建模型
    model = OrionSystem(config)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("No model state dict found in checkpoint")
    
    return model, config, checkpoint


def evaluate_model(model: OrionSystem, 
                  test_loader,
                  device: torch.device,
                  num_trajectories: int = 5,
                  save_predictions: bool = False,
                  output_dir: str = None) -> dict:
    """评估模型"""
    model.eval()
    metrics_calculator = OrionMetrics()
    
    all_predictions = []
    all_targets = []
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 预测
            predictions = model.predict(
                batch["image"], 
                batch.get("text_prompt", None),
                num_trajectories=num_trajectories
            )
            
            # 计算指标
            outputs = model(batch, training=False)
            batch_metrics = metrics_calculator.compute(outputs, batch)
            all_metrics.append(batch_metrics)
            
            # 保存预测结果
            if save_predictions:
                all_predictions.append(predictions)
                all_targets.append({
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                })
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # 计算整体指标
    overall_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        overall_metrics[key] = sum(values) / len(values)
    
    # 计算详细指标
    if save_predictions:
        detailed_metrics = metrics_calculator.compute_detailed_metrics(
            all_predictions, all_targets
        )
        overall_metrics.update(detailed_metrics)
    
    return overall_metrics, all_predictions, all_targets


def generate_visualizations(model: OrionSystem,
                          test_loader,
                          device: torch.device,
                          output_dir: str,
                          num_samples: int = 10):
    """生成可视化"""
    visualizer = TrajectoryVisualizer()
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 预测
            predictions = model.predict(batch["image"], num_trajectories=5)
            
            # 为批次中的每个样本生成可视化
            batch_size = batch["image"].size(0)
            for i in range(min(batch_size, num_samples - sample_count)):
                # 提取单个样本的数据
                if "trajectory" in batch:
                    gt_traj = batch["trajectory"][i].cpu().numpy()
                    pred_trajs = [predictions["trajectories"][i, j] for j in range(5)]
                    
                    # 绘制预测对比
                    fig = visualizer.plot_prediction_comparison(
                        ground_truth=gt_traj,
                        predictions=pred_trajs,
                        title=f"Sample {sample_count + 1}",
                        save_path=os.path.join(vis_dir, f"prediction_{sample_count + 1}.png")
                    )
                    fig.close()
                
                sample_count += 1
                if sample_count >= num_samples:
                    break


def save_results(metrics: dict,
                predictions: list,
                targets: list,
                output_dir: str,
                experiment_name: str):
    """保存评估结果"""
    results_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存指标
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        # 转换numpy类型为Python原生类型
        json_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                json_metrics[key] = value.item()
            elif hasattr(value, 'tolist'):
                json_metrics[key] = value.tolist()
            else:
                json_metrics[key] = value
        
        json.dump(json_metrics, f, indent=2)
    
    # 保存预测结果（如果有）
    if predictions and targets:
        import pickle
        predictions_file = os.path.join(results_dir, "predictions.pkl")
        with open(predictions_file, 'wb') as f:
            pickle.dump({
                'predictions': predictions,
                'targets': targets
            }, f)
    
    print(f"Results saved to {results_dir}")


def print_metrics(metrics: dict):
    """打印指标"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # 分类显示不同类型的指标
    trajectory_metrics = {}
    vqa_metrics = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        if any(traj_key in key.lower() for traj_key in ['ade', 'fde', 'miss', 'trajectory', 'velocity', 'direction']):
            trajectory_metrics[key] = value
        elif 'vqa' in key.lower() or any(vqa_key in key.lower() for vqa_key in ['accuracy', 'precision', 'recall', 'f1']):
            vqa_metrics[key] = value
        else:
            other_metrics[key] = value
    
    # 打印轨迹指标
    if trajectory_metrics:
        print("\nTrajectory Metrics:")
        print("-" * 20)
        for key, value in trajectory_metrics.items():
            print(f"{key:25}: {value:.4f}")
    
    # 打印VQA指标
    if vqa_metrics:
        print("\nVQA Metrics:")
        print("-" * 20)
        for key, value in vqa_metrics.items():
            if 'accuracy' in key.lower() or 'f1' in key.lower():
                print(f"{key:25}: {value*100:.2f}%")
            else:
                print(f"{key:25}: {value:.4f}")
    
    # 打印其他指标
    if other_metrics:
        print("\nOther Metrics:")
        print("-" * 20)
        for key, value in other_metrics.items():
            print(f"{key:25}: {value:.4f}")
    
    print("="*50)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(level=args.verbose and "DEBUG" or "INFO")
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 加载模型和配置
    print("Loading model and config...")
    try:
        model, config, checkpoint = load_model_and_config(
            args.checkpoint, args.config, device
        )
        model = model.to(device)
        print("Model loaded successfully")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    
    # 更新配置
    if args.data_path:
        config.data_path = args.data_path
    if args.dataset_type:
        config.dataset_type = args.dataset_type
    
    # 创建数据加载器
    print("Creating data loader...")
    try:
        test_loader = create_dataloader(
            dataset_type=config.dataset_type,
            data_path=config.data_path,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
        total_samples = len(test_loader.dataset)
        if args.num_samples:
            total_samples = min(total_samples, args.num_samples)
        
        print(f"Dataset: {config.dataset_type}")
        print(f"Split: {args.split}")
        print(f"Total samples: {total_samples}")
        
    except Exception as e:
        print(f"Failed to create data loader: {e}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估模型
    print("\nStarting evaluation...")
    try:
        metrics, predictions, targets = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            num_trajectories=args.num_trajectories,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        
        print("Evaluation completed successfully")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 打印结果
    print_metrics(metrics)
    
    # 生成可视化
    if args.visualize:
        print("\nGenerating visualizations...")
        try:
            generate_visualizations(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=args.output_dir,
                num_samples=args.num_vis_samples
            )
            print("Visualizations saved")
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # 保存结果
    print("\nSaving results...")
    try:
        save_results(
            metrics=metrics,
            predictions=predictions if args.save_predictions else [],
            targets=targets if args.save_predictions else [],
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
    except Exception as e:
        print(f"Failed to save results: {e}")
        return 1
    
    print("\nEvaluation finished successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
