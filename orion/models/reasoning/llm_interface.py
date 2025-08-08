"""LLM接口实现"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from abc import ABC, abstractmethod

from ...config.model_configs import LLMConfig

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        GenerationConfig
    )
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class LLMInterface(nn.Module, ABC):
    """LLM接口基类"""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, visual_tokens: torch.Tensor, 
                text_input: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            visual_tokens: (B, Q, D) 视觉token
            text_input: 可选的文本输入
            
        Returns:
            包含planning_token和vqa_logits的字典
        """
        pass
    
    @abstractmethod
    def generate_response(self, visual_tokens: torch.Tensor,
                         prompt: str, max_length: int = 100) -> List[str]:
        """生成文本响应"""
        pass


class ToyLLM(LLMInterface):
    """
    轻量级Transformer模拟LLM
    用于快速原型开发和测试
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_sequence_length, config.token_dim) * 0.02
        )
        
        # 输出头
        self.planning_head = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim, config.token_dim)
        )
        
        self.vqa_head = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vqa_classes)
        )
        
        # 场景理解头（可选）
        self.scene_understanding_head = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 128)  # 场景特征维度
        )
    
    def forward(self, visual_tokens: torch.Tensor,
                text_input: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_tokens: (B, Q, D) 视觉token
            text_input: 文本输入（此实现中忽略）
            
        Returns:
            包含planning_token、vqa_logits等的字典
        """
        B, Q, D = visual_tokens.shape
        
        # 添加位置编码
        if Q <= self.config.max_sequence_length:
            pos_emb = self.pos_embedding[:, :Q, :]
            x = visual_tokens + pos_emb
        else:
            # 序列太长时截断
            x = visual_tokens[:, :self.config.max_sequence_length, :]
            x = x + self.pos_embedding
        
        # Transformer编码
        encoded = self.transformer(x)  # (B, Q, D)
        
        # 聚合特征（使用注意力池化）
        attention_weights = torch.softmax(
            torch.sum(encoded, dim=-1), dim=-1
        )  # (B, Q)
        pooled = torch.sum(
            encoded * attention_weights.unsqueeze(-1), dim=1
        )  # (B, D)
        
        # 生成输出
        planning_token = self.planning_head(pooled)
        vqa_logits = self.vqa_head(pooled)
        scene_features = self.scene_understanding_head(pooled)
        
        return {
            "planning_token": planning_token,
            "vqa_logits": vqa_logits,
            "scene_features": scene_features,
            "encoded_tokens": encoded,
            "attention_weights": attention_weights
        }
    
    def generate_response(self, visual_tokens: torch.Tensor,
                         prompt: str, max_length: int = 100) -> List[str]:
        """简单的响应生成（基于VQA分类）"""
        with torch.no_grad():
            output = self.forward(visual_tokens)
            vqa_logits = output["vqa_logits"]
            predicted_classes = torch.argmax(vqa_logits, dim=-1)
            
            # 简单的类别到文本映射
            class_responses = [
                "The scene appears safe for driving.",
                "There are obstacles ahead, proceed with caution.",
                "Traffic light detected, preparing to stop.",
                "Pedestrians visible, reducing speed.",
                "Clear road ahead, maintaining speed.",
                # 添加更多响应...
            ]
            
            responses = []
            for class_id in predicted_classes:
                class_idx = class_id.item() % len(class_responses)
                responses.append(class_responses[class_idx])
            
            return responses


class HuggingFaceLLM(LLMInterface):
    """
    HuggingFace模型包装器
    支持GPT、LLaMA等预训练模型
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not _HF_AVAILABLE:
            raise ImportError("transformers library is required for HuggingFaceLLM")
        
        if config.model_name is None:
            raise ValueError("model_name must be specified for HuggingFaceLLM")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 获取模型隐藏层维度
        self.hidden_dim = self.model.config.hidden_size
        
        # 视觉token投影
        self.vision_proj = nn.Linear(config.token_dim, self.hidden_dim)
        
        # 输出投影
        self.planning_proj = nn.Linear(self.hidden_dim, config.token_dim)
        self.vqa_proj = nn.Linear(self.hidden_dim, config.vqa_classes)
        
        # 生成配置
        self.generation_config = GenerationConfig(
            max_length=config.max_sequence_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, visual_tokens: torch.Tensor,
                text_input: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_tokens: (B, Q, D) 视觉token
            text_input: 可选的文本输入
            
        Returns:
            包含planning_token、vqa_logits等的字典
        """
        B, Q, D = visual_tokens.shape
        
        # 投影视觉token到模型空间
        visual_embeds = self.vision_proj(visual_tokens)  # (B, Q, hidden_dim)
        
        # 准备文本输入
        if text_input is None:
            text_input = "Analyze the driving scene and provide planning guidance."
        
        # 编码文本
        text_inputs = self.tokenizer(
            [text_input] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length - Q
        )
        text_inputs = {k: v.to(visual_tokens.device) for k, v in text_inputs.items()}
        
        # 获取文本嵌入
        text_embeds = self.model.get_input_embeddings()(text_inputs["input_ids"])
        
        # 融合视觉和文本嵌入
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 前向传播
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=torch.ones(
                B, combined_embeds.size(1),
                device=visual_tokens.device
            ),
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
        
        # 聚合特征（使用视觉部分的平均）
        visual_part = last_hidden[:, :Q, :]  # (B, Q, hidden_dim)
        pooled = visual_part.mean(dim=1)  # (B, hidden_dim)
        
        # 生成输出
        planning_token = self.planning_proj(pooled)
        vqa_logits = self.vqa_proj(pooled)
        
        return {
            "planning_token": planning_token,
            "vqa_logits": vqa_logits,
            "hidden_states": last_hidden,
            "visual_features": visual_part
        }
    
    def generate_response(self, visual_tokens: torch.Tensor,
                         prompt: str, max_length: int = 100) -> List[str]:
        """生成文本响应"""
        B = visual_tokens.size(0)
        
        # 投影视觉token
        visual_embeds = self.vision_proj(visual_tokens)
        
        # 编码提示
        prompt_inputs = self.tokenizer(
            [prompt] * B,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        prompt_inputs = {k: v.to(visual_tokens.device) for k, v in prompt_inputs.items()}
        
        # 获取提示嵌入
        prompt_embeds = self.model.get_input_embeddings()(prompt_inputs["input_ids"])
        
        # 融合嵌入
        combined_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeds,
                generation_config=self.generation_config,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        responses = []
        for output in outputs:
            # 跳过输入部分
            response_ids = output[combined_embeds.size(1):]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def freeze_llm(self):
        """冻结LLM参数，只训练投影层"""
        for param in self.model.parameters():
            param.requires_grad = False
