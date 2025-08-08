"""ORION评估指标"""

import torch
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class OrionMetrics:
    """
    ORION系统评估指标
    
    包含：
    1. 轨迹规划指标：ADE, FDE, 碰撞率等
    2. VQA分类指标：准确率, F1分数等
    3. 时序一致性指标
    4. 系统整体性能指标
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置累积指标"""
        self.trajectory_errors = []
        self.vqa_predictions = []
        self.vqa_targets = []
        self.temporal_consistency_scores = []
    
    def compute(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算批次指标
        
        Args:
            outputs: 模型输出
            targets: 目标数据
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 轨迹指标
        if "planning_recon_trajectory" in outputs and "trajectory" in targets:
            traj_metrics = self._compute_trajectory_metrics(
                outputs["planning_recon_trajectory"], targets["trajectory"]
            )
            metrics.update(traj_metrics)
        
        # VQA指标
        vqa_metrics = self._compute_vqa_metrics(outputs, targets)
        metrics.update(vqa_metrics)
        
        # 时序一致性指标
        if "temporal_consistency" in outputs:
            temporal_metric = self._compute_temporal_metrics(outputs["temporal_consistency"])
            metrics.update(temporal_metric)
        
        return metrics
    
    def _compute_trajectory_metrics(self, pred_traj: torch.Tensor, 
                                  gt_traj: torch.Tensor) -> Dict[str, float]:
        """计算轨迹指标"""
        pred_traj = pred_traj.detach().cpu().numpy()  # (B, T, 2)
        gt_traj = gt_traj.detach().cpu().numpy()      # (B, T, 2)
        
        metrics = {}
        
        # Average Displacement Error (ADE)
        displacement_errors = np.linalg.norm(pred_traj - gt_traj, axis=2)  # (B, T)
        ade = np.mean(displacement_errors)
        metrics["ade"] = ade
        
        # Final Displacement Error (FDE)
        final_errors = np.linalg.norm(pred_traj[:, -1] - gt_traj[:, -1], axis=1)  # (B,)
        fde = np.mean(final_errors)
        metrics["fde"] = fde
        
        # Miss Rate (预测终点距离真实终点超过2米的比例)
        miss_rate = np.mean(final_errors > 2.0)
        metrics["miss_rate"] = miss_rate
        
        # 轨迹相似度（使用DTW距离的简化版本）
        similarity_scores = []
        for i in range(pred_traj.shape[0]):
            similarity = self._compute_trajectory_similarity(pred_traj[i], gt_traj[i])
            similarity_scores.append(similarity)
        metrics["trajectory_similarity"] = np.mean(similarity_scores)
        
        # 速度一致性
        pred_velocities = np.diff(pred_traj, axis=1)  # (B, T-1, 2)
        gt_velocities = np.diff(gt_traj, axis=1)      # (B, T-1, 2)
        velocity_errors = np.linalg.norm(pred_velocities - gt_velocities, axis=2)
        metrics["velocity_consistency"] = np.mean(velocity_errors)
        
        # 方向一致性
        pred_directions = pred_velocities / (np.linalg.norm(pred_velocities, axis=2, keepdims=True) + 1e-8)
        gt_directions = gt_velocities / (np.linalg.norm(gt_velocities, axis=2, keepdims=True) + 1e-8)
        direction_similarity = np.sum(pred_directions * gt_directions, axis=2)  # cosine similarity
        metrics["direction_consistency"] = np.mean(direction_similarity)
        
        return metrics
    
    def _compute_vqa_metrics(self, outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算VQA指标"""
        metrics = {}
        
        # 遍历所有VQA任务
        vqa_tasks = ["scene_classification", "weather_detection", "traffic_light", "driving_intent"]
        
        for task in vqa_tasks:
            logits_key = f"vqa_{task}_logits"
            labels_key = f"{task}_labels"
            
            if logits_key in outputs and labels_key in targets:
                task_metrics = self._compute_classification_metrics(
                    outputs[logits_key], targets[labels_key], prefix=f"vqa_{task}_"
                )
                metrics.update(task_metrics)
        
        # 通用VQA指标
        if "vqa_logits" in outputs and "vqa_labels" in targets:
            general_metrics = self._compute_classification_metrics(
                outputs["vqa_logits"], targets["vqa_labels"], prefix="vqa_"
            )
            metrics.update(general_metrics)
        
        # VQA置信度指标
        if "vqa_confidence" in outputs:
            confidence_scores = outputs["vqa_confidence"].detach().cpu().numpy()
            metrics["vqa_avg_confidence"] = np.mean(confidence_scores)
            metrics["vqa_confidence_std"] = np.std(confidence_scores)
        
        return metrics
    
    def _compute_classification_metrics(self, logits: torch.Tensor, 
                                      labels: torch.Tensor,
                                      prefix: str = "") -> Dict[str, float]:
        """计算分类指标"""
        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        metrics = {}
        
        # 准确率
        accuracy = accuracy_score(labels, predictions)
        metrics[f"{prefix}accuracy"] = accuracy
        
        # 精确率、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        metrics[f"{prefix}precision"] = precision
        metrics[f"{prefix}recall"] = recall
        metrics[f"{prefix}f1"] = f1
        
        # Top-k准确率（如果类别数量>1）
        if logits.size(1) > 1:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            top3_accuracy = self._compute_topk_accuracy(probs, labels, k=min(3, logits.size(1)))
            metrics[f"{prefix}top3_accuracy"] = top3_accuracy
        
        return metrics
    
    def _compute_temporal_metrics(self, consistency_scores: torch.Tensor) -> Dict[str, float]:
        """计算时序一致性指标"""
        scores = consistency_scores.detach().cpu().numpy()
        
        return {
            "temporal_consistency_mean": np.mean(scores),
            "temporal_consistency_std": np.std(scores),
            "temporal_consistency_min": np.min(scores),
            "temporal_consistency_max": np.max(scores)
        }
    
    def _compute_trajectory_similarity(self, pred_traj: np.ndarray, 
                                     gt_traj: np.ndarray) -> float:
        """计算轨迹相似度（简化版DTW）"""
        # 使用欧几里得距离作为相似度度量
        distances = np.linalg.norm(pred_traj - gt_traj, axis=1)
        # 转换为相似度（0-1之间）
        max_distance = np.sqrt(2) * 10  # 假设最大可能距离
        similarity = np.exp(-distances / max_distance)
        return np.mean(similarity)
    
    def _compute_topk_accuracy(self, probs: np.ndarray, labels: np.ndarray, k: int) -> float:
        """计算Top-k准确率"""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
        return np.mean(correct)
    
    def compute_detailed_metrics(self, all_predictions: List[Dict[str, Any]], 
                               all_targets: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算详细的评估指标"""
        metrics = {}
        
        # 聚合所有轨迹数据
        all_pred_trajs = []
        all_gt_trajs = []
        
        for pred, target in zip(all_predictions, all_targets):
            if "trajectories" in pred and "trajectory" in target:
                # 选择最佳轨迹（第一个）
                best_traj = pred["trajectories"][:, 0]  # (B, T, 2)
                all_pred_trajs.append(best_traj)
                all_gt_trajs.append(target["trajectory"])
        
        if all_pred_trajs:
            pred_trajs = np.concatenate(all_pred_trajs, axis=0)
            gt_trajs = np.concatenate(all_gt_trajs, axis=0)
            
            # 计算详细轨迹指标
            detailed_traj_metrics = self._compute_detailed_trajectory_metrics(pred_trajs, gt_trajs)
            metrics.update(detailed_traj_metrics)
        
        # 聚合VQA数据
        vqa_metrics = self._compute_detailed_vqa_metrics(all_predictions, all_targets)
        metrics.update(vqa_metrics)
        
        return metrics
    
    def _compute_detailed_trajectory_metrics(self, pred_trajs: np.ndarray, 
                                           gt_trajs: np.ndarray) -> Dict[str, float]:
        """计算详细的轨迹指标"""
        metrics = {}
        
        # 基础指标
        displacement_errors = np.linalg.norm(pred_trajs - gt_trajs, axis=2)
        
        # 时间步级别的ADE
        for t in [1, 2, 3, 5, 8]:  # 不同预测时间步
            if t < pred_trajs.shape[1]:
                metrics[f"ade_t{t}"] = np.mean(displacement_errors[:, :t])
        
        # 距离级别的miss rate
        final_errors = np.linalg.norm(pred_trajs[:, -1] - gt_trajs[:, -1], axis=1)
        for threshold in [1.0, 2.0, 3.0, 5.0]:
            metrics[f"miss_rate_{threshold}m"] = np.mean(final_errors > threshold)
        
        # 轨迹类型分析（直线、左转、右转）
        gt_angles = self._compute_trajectory_angles(gt_trajs)
        pred_angles = self._compute_trajectory_angles(pred_trajs)
        
        # 按轨迹类型分组计算ADE
        straight_mask = np.abs(gt_angles) < 15  # 小于15度为直线
        left_mask = gt_angles >= 15            # 大于15度为左转
        right_mask = gt_angles <= -15          # 小于-15度为右转
        
        for mask, name in [(straight_mask, "straight"), (left_mask, "left"), (right_mask, "right")]:
            if np.any(mask):
                metrics[f"ade_{name}"] = np.mean(displacement_errors[mask])
                metrics[f"fde_{name}"] = np.mean(final_errors[mask])
        
        return metrics
    
    def _compute_detailed_vqa_metrics(self, all_predictions: List[Dict[str, Any]], 
                                    all_targets: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算详细的VQA指标"""
        metrics = {}
        
        # 聚合各任务的预测结果
        task_predictions = {}
        task_targets = {}
        
        for pred, target in zip(all_predictions, all_targets):
            vqa_preds = pred.get("vqa_predictions", {})
            for task, task_pred in vqa_preds.items():
                if task not in task_predictions:
                    task_predictions[task] = []
                    task_targets[task] = []
                
                task_predictions[task].extend(task_pred["predictions"])
                
                # 从target中获取对应标签
                if f"{task}_labels" in target:
                    task_targets[task].extend(target[f"{task}_labels"])
        
        # 计算每个任务的详细指标
        for task in task_predictions:
            if task in task_targets and len(task_targets[task]) > 0:
                preds = np.array(task_predictions[task])
                targets = np.array(task_targets[task])
                
                # 混淆矩阵
                cm = confusion_matrix(targets, preds)
                
                # 每类别的精确率、召回率、F1
                precision, recall, f1, support = precision_recall_fscore_support(
                    targets, preds, average=None, zero_division=0
                )
                
                for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
                    metrics[f"{task}_class{i}_precision"] = p
                    metrics[f"{task}_class{i}_recall"] = r
                    metrics[f"{task}_class{i}_f1"] = f
                    metrics[f"{task}_class{i}_support"] = s
                
                # 宏平均和微平均
                metrics[f"{task}_macro_f1"] = np.mean(f1)
                metrics[f"{task}_weighted_f1"] = np.average(f1, weights=support)
        
        return metrics
    
    def _compute_trajectory_angles(self, trajectories: np.ndarray) -> np.ndarray:
        """计算轨迹的总体转向角度"""
        # 计算起点到终点的向量
        start_to_end = trajectories[:, -1] - trajectories[:, 0]  # (B, 2)
        
        # 计算角度（弧度转角度）
        angles = np.arctan2(start_to_end[:, 1], start_to_end[:, 0]) * 180 / np.pi
        
        return angles
    
    def visualize_results(self, all_predictions: List[Dict[str, Any]], 
                         all_targets: List[Dict[str, Any]], 
                         save_dir: str = "visualizations"):
        """可视化评估结果"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 轨迹可视化
        self._plot_trajectory_examples(all_predictions, all_targets, save_dir)
        
        # 2. VQA混淆矩阵
        self._plot_vqa_confusion_matrices(all_predictions, all_targets, save_dir)
        
        # 3. 错误分布
        self._plot_error_distributions(all_predictions, all_targets, save_dir)
    
    def _plot_trajectory_examples(self, predictions: List[Dict[str, Any]], 
                                targets: List[Dict[str, Any]], save_dir: str):
        """绘制轨迹示例"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(min(6, len(predictions))):
            if "trajectories" in predictions[i] and "trajectory" in targets[i]:
                pred_traj = predictions[i]["trajectories"][0, 0]  # 第一个batch的第一个轨迹
                gt_traj = targets[i]["trajectory"][0]
                
                axes[i].plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', label='Ground Truth', linewidth=2)
                axes[i].plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Prediction', linewidth=2)
                axes[i].scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=100, marker='o', label='Start')
                axes[i].scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=100, marker='s', label='End')
                axes[i].set_title(f'Trajectory {i+1}')
                axes[i].legend()
                axes[i].grid(True)
                axes[i].axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trajectory_examples.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_vqa_confusion_matrices(self, predictions: List[Dict[str, Any]], 
                                   targets: List[Dict[str, Any]], save_dir: str):
        """绘制VQA混淆矩阵"""
        # 实现VQA混淆矩阵绘制
        pass
    
    def _plot_error_distributions(self, predictions: List[Dict[str, Any]], 
                                targets: List[Dict[str, Any]], save_dir: str):
        """绘制错误分布图"""
        # 实现错误分布绘制
        pass
