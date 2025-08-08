"""QT-Former时序模块实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ...config.model_configs import QTFormerConfig


class CrossAttention(nn.Module):
    """交叉注意力模块"""
    
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q, 
            kdim=dim_kv, 
            vdim=dim_kv, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.proj = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_q)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: (B, Q, dim_q)
            keys: (B, K, dim_kv) 
            attn_mask: (Q, K) 注意力掩码
            
        Returns:
            output: (B, Q, dim_q)
            attn_weights: (B, num_heads, Q, K)
        """
        # 残差连接
        residual = queries
        
        # 交叉注意力
        attn_output, attn_weights = self.attn(
            queries, keys, keys, 
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        # 投影和残差连接
        output = self.proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output, attn_weights


class QTFormer(nn.Module):
    """
    基于查询的时序变换器（Query-based Temporal Former）
    - 维护历史帧特征的记忆库
    - 使用可学习的历史查询通过交叉注意力提取长期上下文
    - 为LLM生成紧凑的token，为规划器生成规划token
    """
    
    def __init__(self, config: QTFormerConfig):
        super().__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.memory_size = config.memory_size
        self.num_queries = config.num_history_queries
        
        # 记忆库 - 使用register_buffer确保在设备间正确移动
        self.register_buffer(
            'memory', 
            torch.zeros(config.memory_size, config.vision_dim),
            persistent=False
        )
        self.register_buffer(
            'memory_length',
            torch.zeros((), dtype=torch.long),
            persistent=False  
        )
        self.register_buffer(
            'memory_ptr',
            torch.zeros((), dtype=torch.long),
            persistent=False
        )
        
        # 可学习的历史查询
        self.history_queries = nn.Parameter(
            torch.randn(config.num_history_queries, config.token_dim) * 0.02
        )
        
        # 投影层
        self.query_proj = nn.Linear(config.token_dim, config.token_dim)
        self.key_value_proj = nn.Linear(config.vision_dim, config.token_dim)
        
        # 多层交叉注意力
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(
                dim_q=config.token_dim,
                dim_kv=config.token_dim, 
                num_heads=config.num_heads,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # 输出头
        self.llm_head = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim, config.token_dim)
        )
        
        self.planning_head = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim, config.token_dim)
        )
        
        # 时序融合权重
        self.temporal_fusion = nn.Parameter(torch.tensor(0.5))
    
    def reset_memory(self):
        """重置记忆库"""
        self.memory.zero_()
        self.memory_length.zero_()
        self.memory_ptr.zero_()
    
    def update_memory(self, features: torch.Tensor):
        """
        更新记忆库
        
        Args:
            features: (B, D) 当前帧特征
        """
        with torch.no_grad():
            # 计算批次平均特征
            avg_features = features.mean(dim=0)  # (D,)
            
            # 更新记忆库
            if self.memory_length < self.memory_size:
                # 记忆库未满，直接添加
                self.memory[self.memory_length].copy_(avg_features)
                self.memory_length += 1
            else:
                # 记忆库已满，循环覆盖
                self.memory[self.memory_ptr].copy_(avg_features)
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
    
    def get_memory_features(self, batch_size: int) -> torch.Tensor:
        """
        获取记忆库特征
        
        Args:
            batch_size: 批次大小
            
        Returns:
            memory_features: (B, M, D) 记忆特征
        """
        memory_len = int(self.memory_length.item())
        if memory_len == 0:
            # 记忆库为空，返回零特征
            return torch.zeros(
                batch_size, 1, self.config.vision_dim,
                device=self.memory.device,
                dtype=self.memory.dtype
            )
        
        # 获取有效记忆并扩展到批次维度
        valid_memory = self.memory[:memory_len]  # (M, D)
        memory_features = valid_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, D)
        
        return memory_features
    
    def forward(self, current_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            current_features: (B, D) 当前帧视觉特征
            
        Returns:
            llm_tokens: (B, Q, D) LLM输入token
            planning_token: (B, D) 规划token
        """
        batch_size = current_features.size(0)
        
        # 更新记忆库
        self.update_memory(current_features.detach())
        
        # 构建键值序列：记忆 + 当前特征
        memory_features = self.get_memory_features(batch_size)  # (B, M, D)
        current_expanded = current_features.unsqueeze(1)  # (B, 1, D)
        key_value_sequence = torch.cat([memory_features, current_expanded], dim=1)  # (B, M+1, D)
        
        # 投影到token空间
        keys_values = self.key_value_proj(key_value_sequence)  # (B, M+1, token_dim)
        
        # 准备查询
        queries = self.query_proj(self.history_queries).unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, Q, token_dim)
        
        # 多层交叉注意力
        for layer in self.cross_attention_layers:
            queries, _ = layer(queries, keys_values)
        
        # 生成输出token
        llm_tokens = self.llm_head(queries)  # (B, Q, token_dim)
        
        # 生成规划token（聚合所有查询）
        aggregated = queries.mean(dim=1)  # (B, token_dim)
        planning_token = self.planning_head(aggregated)  # (B, token_dim)
        
        return llm_tokens, planning_token
    
    def get_attention_maps(self, current_features: torch.Tensor) -> torch.Tensor:
        """
        获取注意力图（用于可视化）
        
        Args:
            current_features: (B, D) 当前帧特征
            
        Returns:
            attention_maps: (B, num_heads, Q, M+1) 注意力权重
        """
        batch_size = current_features.size(0)
        
        # 构建键值序列
        memory_features = self.get_memory_features(batch_size)
        current_expanded = current_features.unsqueeze(1)
        key_value_sequence = torch.cat([memory_features, current_expanded], dim=1)
        keys_values = self.key_value_proj(key_value_sequence)
        
        # 准备查询
        queries = self.query_proj(self.history_queries).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # 获取第一层的注意力权重
        _, attention_weights = self.cross_attention_layers[0](queries, keys_values)
        
        return attention_weights
    
    def get_memory_usage(self) -> float:
        """获取记忆库使用率"""
        return self.memory_length.item() / self.memory_size
