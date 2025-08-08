"""VQA头部实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class VQAHead(nn.Module):
    """
    视觉问答头部
    支持多种类型的VQA任务
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 num_classes: int = 16,
                 hidden_dim: int = 512,
                 dropout: float = 0.1,
                 task_types: Optional[List[str]] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task_types = task_types or ["scene_classification"]
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 任务特定的分类头
        self.task_heads = nn.ModuleDict()
        for task in self.task_types:
            self.task_heads[task] = self._create_task_head(task, hidden_dim, dropout)
        
        # 注意力机制（用于多任务融合）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _create_task_head(self, task_type: str, hidden_dim: int, dropout: float) -> nn.Module:
        """为不同任务创建特定的分类头"""
        
        if task_type == "scene_classification":
            # 场景分类：室外/室内、白天/夜晚等
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 8)  # 8种场景类型
            )
        
        elif task_type == "weather_detection":
            # 天气检测：晴天、雨天、雪天等
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 5)  # 5种天气类型
            )
        
        elif task_type == "traffic_light":
            # 交通灯状态：红、黄、绿、无
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 4)  # 4种交通灯状态
            )
        
        elif task_type == "object_detection":
            # 关键对象检测：车辆、行人、障碍物等
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 10)  # 10种对象类型
            )
        
        elif task_type == "driving_intent":
            # 驾驶意图：直行、左转、右转、停车等
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 6)  # 6种驾驶意图
            )
        
        else:
            # 默认分类头
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.num_classes)
            )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: (B, D) 输入特征
            
        Returns:
            包含各任务预测结果的字典
        """
        # 特征提取
        shared_features = self.feature_extractor(features)  # (B, hidden_dim)
        
        # 自注意力增强
        enhanced_features, _ = self.attention(
            shared_features.unsqueeze(1),
            shared_features.unsqueeze(1), 
            shared_features.unsqueeze(1)
        )
        enhanced_features = enhanced_features.squeeze(1)  # (B, hidden_dim)
        
        # 任务特定预测
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[f"{task_name}_logits"] = task_head(enhanced_features)
        
        # 置信度预测
        outputs["confidence"] = self.confidence_head(enhanced_features)
        
        # 添加特征输出（用于下游任务）
        outputs["vqa_features"] = enhanced_features
        
        return outputs
    
    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> Dict[str, any]:
        """
        预测并返回可解释的结果
        
        Args:
            features: (B, D) 输入特征
            threshold: 置信度阈值
            
        Returns:
            包含预测结果和置信度的字典
        """
        with torch.no_grad():
            outputs = self.forward(features)
            
            predictions = {}
            for task_name in self.task_types:
                logits_key = f"{task_name}_logits"
                if logits_key in outputs:
                    logits = outputs[logits_key]
                    probs = F.softmax(logits, dim=-1)
                    pred_classes = torch.argmax(probs, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    
                    predictions[task_name] = {
                        "predicted_class": pred_classes.cpu().numpy(),
                        "probabilities": probs.cpu().numpy(),
                        "confidence": max_probs.cpu().numpy(),
                        "high_confidence": (max_probs > threshold).cpu().numpy()
                    }
            
            # 整体置信度
            predictions["overall_confidence"] = outputs["confidence"].cpu().numpy()
            
            return predictions
    
    def get_class_names(self, task_type: str) -> List[str]:
        """获取任务的类别名称"""
        class_names = {
            "scene_classification": [
                "urban", "highway", "rural", "parking", 
                "intersection", "tunnel", "bridge", "residential"
            ],
            "weather_detection": [
                "sunny", "cloudy", "rainy", "snowy", "foggy"
            ],
            "traffic_light": [
                "red", "yellow", "green", "none"
            ],
            "object_detection": [
                "car", "truck", "bus", "motorcycle", "bicycle",
                "pedestrian", "traffic_sign", "barrier", "animal", "other"
            ],
            "driving_intent": [
                "go_straight", "turn_left", "turn_right", 
                "stop", "park", "change_lane"
            ]
        }
        
        return class_names.get(task_type, [f"class_{i}" for i in range(self.num_classes)])
    
    def explain_prediction(self, features: torch.Tensor, task_type: str) -> Dict[str, any]:
        """
        解释预测结果
        
        Args:
            features: (B, D) 输入特征
            task_type: 任务类型
            
        Returns:
            包含解释信息的字典
        """
        predictions = self.predict(features)
        class_names = self.get_class_names(task_type)
        
        if task_type not in predictions:
            return {"error": f"Task {task_type} not found"}
        
        task_pred = predictions[task_type]
        explanations = []
        
        for i in range(len(task_pred["predicted_class"])):
            pred_class = task_pred["predicted_class"][i]
            confidence = task_pred["confidence"][i]
            probs = task_pred["probabilities"][i]
            
            # 获取top-3预测
            top_indices = torch.argsort(torch.tensor(probs), descending=True)[:3]
            
            explanation = {
                "predicted_class": class_names[pred_class] if pred_class < len(class_names) else f"unknown_{pred_class}",
                "confidence": float(confidence),
                "top_predictions": [
                    {
                        "class": class_names[idx] if idx < len(class_names) else f"unknown_{idx}",
                        "probability": float(probs[idx])
                    }
                    for idx in top_indices
                ]
            }
            explanations.append(explanation)
        
        return {"explanations": explanations}
