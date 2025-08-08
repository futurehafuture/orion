#!/usr/bin/env python3
"""ORION演示脚本"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from orion.config import OrionConfig
from orion.models import OrionSystem
from orion.data import ToyDrivingDataset, DrivingTransforms
from orion.utils import load_checkpoint, TrajectoryVisualizer, create_demo_video


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ORION Demo - Interactive driving visualization")
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path (optional for toy demo)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path')
    
    # 演示模式
    parser.add_argument('--mode', type=str, default='toy',
                       choices=['toy', 'interactive', 'video', 'image'],
                       help='Demo mode')
    
    # 输入参数
    parser.add_argument('--input-image', type=str, default=None,
                       help='Input image path (for image mode)')
    parser.add_argument('--input-video', type=str, default=None,
                       help='Input video path (for video mode)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID (for interactive mode)')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='demo_outputs/',
                       help='Output directory')
    parser.add_argument('--save-results', action='store_true',
                       help='Save demo results')
    
    # 可视化参数
    parser.add_argument('--num-trajectories', type=int, default=5,
                       help='Number of trajectories to generate')
    parser.add_argument('--show-attention', action='store_true',
                       help='Show attention maps')
    parser.add_argument('--show-vqa', action='store_true',
                       help='Show VQA predictions')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    # 其他参数
    parser.add_argument('--fps', type=int, default=10,
                       help='Video FPS (for video output)')
    parser.add_argument('--window-size', type=str, default='800x600',
                       help='Display window size')
    
    return parser.parse_args()


def load_model_for_demo(checkpoint_path: str = None, config_path: str = None, device: torch.device = None):
    """为演示加载模型"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 加载训练好的模型
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        
        if config_path:
            config = OrionConfig.from_yaml(config_path)
        elif 'config' in checkpoint:
            config = checkpoint['config']
            if not isinstance(config, OrionConfig):
                config = OrionConfig(**config)
        else:
            print("Warning: Using default config")
            config = OrionConfig()
        
        model = OrionSystem(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        
    else:
        # 使用默认配置创建模型（用于演示）
        print("Creating model with default configuration for demo")
        config = OrionConfig()
        model = OrionSystem(config)
        print("Note: Model weights are randomly initialized")
    
    model.eval()
    return model, config


def create_toy_demo(model: OrionSystem, config: OrionConfig, device: torch.device, args):
    """创建玩具演示"""
    print("Running toy demo...")
    
    # 创建玩具数据集
    dataset = ToyDrivingDataset(length=20, seed=42)
    transform = DrivingTransforms(is_training=False)
    
    visualizer = TrajectoryVisualizer()
    
    results = []
    
    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            sample = transform(sample)
            
            # 添加批次维度
            batch = {
                'image': sample['image'].unsqueeze(0).to(device),
                'trajectory': sample['trajectory'].unsqueeze(0).to(device)
            }
            
            # 预测
            predictions = model.predict(batch['image'], num_trajectories=args.num_trajectories)
            
            # 可视化
            gt_traj = sample['trajectory'].numpy()
            pred_trajs = [predictions['trajectories'][0, j] for j in range(args.num_trajectories)]
            
            # 创建可视化
            fig = visualizer.plot_prediction_comparison(
                ground_truth=gt_traj,
                predictions=pred_trajs,
                title=f"Toy Demo Sample {i+1}",
                save_path=os.path.join(args.output_dir, f"toy_demo_{i+1}.png") if args.save_results else None
            )
            
            results.append({
                'sample_id': i,
                'ground_truth': gt_traj,
                'predictions': pred_trajs,
                'vqa_predictions': predictions.get('vqa_predictions', {}),
                'metadata': sample.get('metadata', {})
            })
            
            print(f"Processed sample {i+1}/10")
            
            if not args.save_results:
                # 显示图像
                import matplotlib.pyplot as plt
                plt.show()
                plt.pause(2)
                plt.close(fig)
    
    print("Toy demo completed!")
    return results


def process_single_image(model: OrionSystem, image_path: str, device: torch.device, args):
    """处理单张图像"""
    print(f"Processing image: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    transform = DrivingTransforms(is_training=False)
    
    # 创建假的样本（图像 + 零轨迹）
    sample = {
        'image': image,
        'trajectory': torch.zeros(20, 2)
    }
    sample = transform(sample)
    
    # 预测
    with torch.no_grad():
        batch_image = sample['image'].unsqueeze(0).to(device)
        predictions = model.predict(batch_image, num_trajectories=args.num_trajectories)
        
        # 解释决策
        explanations = model.explain_decision(batch_image)
    
    # 可视化结果
    visualizer = TrajectoryVisualizer()
    
    # 轨迹预测
    pred_trajs = [predictions['trajectories'][0, j] for j in range(args.num_trajectories)]
    fig1 = visualizer.plot_multiple_trajectories(
        pred_trajs,
        labels=[f'Prediction {i+1}' for i in range(len(pred_trajs))],
        title="Predicted Trajectories",
        save_path=os.path.join(args.output_dir, "trajectory_predictions.png") if args.save_results else None
    )
    
    # VQA结果
    if args.show_vqa and 'vqa_predictions' in predictions:
        print("\nVQA Predictions:")
        for task, pred in predictions['vqa_predictions'].items():
            if 'predictions' in pred:
                class_idx = pred['predictions'][0]
                confidence = pred['probabilities'][0][class_idx]
                print(f"{task}: Class {class_idx} (confidence: {confidence:.3f})")
    
    # 显示注意力图
    if args.show_attention and 'attention_maps' in explanations:
        print("Attention maps available in explanations")
    
    if not args.save_results:
        import matplotlib.pyplot as plt
        plt.show()
    
    return predictions, explanations


def run_interactive_demo(model: OrionSystem, device: torch.device, args):
    """运行交互式演示"""
    print("Starting interactive demo...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera_id}")
        return
    
    # 设置窗口大小
    width, height = map(int, args.window_size.split('x'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    transform = DrivingTransforms(is_training=False)
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera")
                break
            
            # 转换为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 预处理
            sample = {'image': pil_image, 'trajectory': torch.zeros(20, 2)}
            sample = transform(sample)
            
            # 预测
            with torch.no_grad():
                batch_image = sample['image'].unsqueeze(0).to(device)
                predictions = model.predict(batch_image, num_trajectories=3)
            
            # 在frame上绘制预测结果
            display_frame = frame.copy()
            
            # 绘制轨迹（简化版本）
            if 'trajectories' in predictions:
                trajectories = predictions['trajectories'][0]  # 第一个样本
                
                for i, traj in enumerate(trajectories[:3]):  # 只显示前3条轨迹
                    color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)][i]
                    
                    # 将轨迹坐标映射到图像坐标
                    traj_scaled = ((traj + 10) * np.array([width, height]) / 20).astype(int)
                    traj_scaled = np.clip(traj_scaled, [0, 0], [width-1, height-1])
                    
                    # 绘制轨迹线
                    for j in range(len(traj_scaled) - 1):
                        cv2.line(display_frame, 
                               tuple(traj_scaled[j]), 
                               tuple(traj_scaled[j+1]), 
                               color, 2)
            
            # 添加文本信息
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示结果
            cv2.imshow('ORION Interactive Demo', display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.save_results:
                save_path = os.path.join(args.output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(save_path, display_frame)
                print(f"Saved frame to {save_path}")
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("Interactive demo finished")


def process_video(model: OrionSystem, video_path: str, device: torch.device, args):
    """处理视频文件"""
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # 设置输出视频
    if args.save_results:
        output_path = os.path.join(args.output_dir, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    transform = DrivingTransforms(is_training=False)
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换和预测（与交互式演示类似）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            sample = {'image': pil_image, 'trajectory': torch.zeros(20, 2)}
            sample = transform(sample)
            
            with torch.no_grad():
                batch_image = sample['image'].unsqueeze(0).to(device)
                predictions = model.predict(batch_image, num_trajectories=3)
            
            # 绘制预测结果（与交互式演示类似的代码）
            display_frame = frame.copy()
            # ... 绘制轨迹的代码 ...
            
            if args.save_results:
                out.write(display_frame)
            
            processed_frames += 1
            if processed_frames % 30 == 0:
                print(f"Processed {processed_frames}/{frame_count} frames")
    
    finally:
        cap.release()
        if args.save_results:
            out.release()
            print(f"Output video saved to {output_path}")
    
    print("Video processing completed")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 创建输出目录
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    try:
        model, config = load_model_for_demo(args.checkpoint, args.config, device)
        model = model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    
    # 根据模式运行不同的演示
    try:
        if args.mode == 'toy':
            results = create_toy_demo(model, config, device, args)
            print(f"Generated {len(results)} toy samples")
        
        elif args.mode == 'image':
            if not args.input_image:
                print("Error: --input-image required for image mode")
                return 1
            process_single_image(model, args.input_image, device, args)
        
        elif args.mode == 'interactive':
            run_interactive_demo(model, device, args)
        
        elif args.mode == 'video':
            if not args.input_video:
                print("Error: --input-video required for video mode")
                return 1
            process_video(model, args.input_video, device, args)
        
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 0
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("Demo completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
