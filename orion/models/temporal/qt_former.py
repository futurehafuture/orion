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
        self.num_history_queries = config.num_history_queries
        self.num_scene_queries = getattr(config, 'num_scene_queries', 4)
        self.num_perception_queries = getattr(config, 'num_perception_queries', 4)
        
        # 记忆库M存储历史查询Q_h（而非视觉均值），形状对齐token_dim
        self.register_buffer(
            'memory', 
            torch.zeros(config.memory_size, config.token_dim),
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
        
        # 可学习查询：历史Q_h、场景Q_s、感知Q_p
        self.history_queries = nn.Parameter(torch.randn(self.num_history_queries, self.token_dim) * 0.02)
        self.scene_queries = nn.Parameter(torch.randn(self.num_scene_queries, self.token_dim) * 0.02)
        self.perception_queries = nn.Parameter(torch.randn(self.num_perception_queries, self.token_dim) * 0.02)
        
        # 投影层：视觉到token空间；查询自投影
        self.key_value_proj = nn.Linear(config.vision_dim, config.token_dim)
        # 3D位置编码与时间编码
        self.use_pos3d = getattr(config, 'use_pos3d_embedding', False)
        self.use_time = getattr(config, 'use_time_embedding', True)
        if self.use_pos3d:
            # 简化实现：基于(H,W)网格生成固定正弦余弦位置编码，投影到token_dim
            self.pos3d_proj = nn.Linear(6, self.token_dim)  # (sin/cos for x,y,z)
        if self.use_time:
            self.time_embed = nn.Embedding(self.memory_size + 1, self.token_dim)
        self.query_proj = nn.Linear(config.token_dim, config.token_dim)
        
        # As per image: x6 perception-scene decoder
        # It consists of self-attention between Qs and Qp, and cross-attention with image features.
        ps_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.token_dim, 
            nhead=config.num_heads,
            dim_feedforward=config.token_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.perception_scene_decoder = nn.TransformerDecoder(ps_decoder_layer, num_layers=6)

        # As per image: x1 history decoder, which has two cross-attention stages
        # 1. Qh attends to Memory Bank
        self.history_decoder_mem = CrossAttention(
            dim_q=config.token_dim,
            dim_kv=config.token_dim, 
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        # 2. Qh attends to updated Qs
        self.history_decoder_scene = CrossAttention(
            dim_q=config.token_dim,
            dim_kv=config.token_dim, 
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 输出头：映射到LLM空间的历史标记x_h与场景标记x_s
        self.llm_head_history = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim, config.token_dim)
        )
        self.llm_head_scene = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim, config.token_dim)
        )

        # 规划token头（可融合x_h与x_s）
        self.planning_head = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim, config.token_dim)
        )

        # 感知辅助头：检测/交通状态/运动预测（简化版）
        self.det_head = nn.Linear(self.token_dim, getattr(config, 'detection_num_classes', 8))
        self.traffic_head = nn.Linear(self.token_dim, getattr(config, 'traffic_state_classes', 4))
        self.motion_head = nn.Linear(self.token_dim, 4)  # 预测2D速度均值(2)与方差(2)的占位
        
        # 时序融合权重
        self.temporal_fusion = nn.Parameter(torch.tensor(0.5))
    
    def reset_memory(self):
        """重置记忆库"""
        self.memory.zero_()
        self.memory_length.zero_()
        self.memory_ptr.zero_()
    
    def update_memory(self, hist_queries: torch.Tensor):
        """
        更新记忆库
        
        Args:
            hist_queries: (B, N_h, D) 当前步更新后的历史查询
        """
        with torch.no_grad():
            # 将当前历史查询聚合为单条写入（可用mean或第一个token）
            avg_features = hist_queries.mean(dim=(0, 1))  # (D,)
            
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
                batch_size, 1, self.token_dim,
                device=self.memory.device,
                dtype=self.memory.dtype
            )
        
        # 获取有效记忆并扩展到批次维度
        valid_memory = self.memory[:memory_len]  # (M, D_token)
        memory_features = valid_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, D)
        
        return memory_features
    
    def forward(self, current_features: torch.Tensor, current_patches: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            current_features: (B, D) 当前帧视觉特征
            
        Returns:
            x_h: (B, N_h, D) LLM历史标记
            x_s: (B, N_s, D) LLM场景标记
            planning_token: (B, D) 规划token
        """
        batch_size = current_features.size(0)
        
        # 视觉特征映射到token空间 (Image Features)
        current_tok = self.key_value_proj(current_features).unsqueeze(1)  # (B,1,D)
        if current_patches is not None:
            # 合并patch token作为更细粒度的键值
            # 假设patch已在视觉侧投影到vision_dim，这里再次线性到token_dim
            if current_patches.size(-1) != self.token_dim:
                # 简化：若维度不同，做一次投影
                patch_proj = nn.functional.linear(current_patches, self.key_value_proj.weight, self.key_value_proj.bias)
            else:
                patch_proj = current_patches
            vision_kv = torch.cat([current_tok, patch_proj], dim=1)  # (B, 1+N, D)
        else:
            vision_kv = current_tok
            
        # 准备查询 Qs, Qp
        Qs = self.query_proj(self.scene_queries).unsqueeze(0).expand(batch_size, -1, -1)
        Qp = self.query_proj(self.perception_queries).unsqueeze(0).expand(batch_size, -1, -1)

        # Phase 1: Perception and Scene Processing (x6 layers as in diagram)
        # -----------------------------------------------------------------
        sp_queries = torch.cat([Qs, Qp], dim=1) # (B, Ns+Np, D)
        
        # Decoder: Self-Attention(sp_queries) -> Cross-Attention(sp_queries, vision_kv)
        sp_output = self.perception_scene_decoder(tgt=sp_queries, memory=vision_kv)
        
        # Separate updated queries
        Qs, Qp = sp_output[:, :self.num_scene_queries, :], sp_output[:, self.num_scene_queries:, :]

        # Phase 2: History Processing (x1 block with two cross-attention stages)
        # --------------------------------------------------------------------
        # Get history queries
        Qh = self.query_proj(self.history_queries).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get memory features and add timestamp
        memory_features = self.get_memory_features(batch_size)  # (B, M, D)
        if self.use_time and int(self.memory_length.item()) > 0:
            mem_len = memory_features.size(1)
            time_ids = torch.arange(mem_len, device=memory_features.device).unsqueeze(0).expand(batch_size, -1)
            time_emb = self.time_embed(time_ids)  # (B, M, D)
            memory_features = memory_features + time_emb

        # Stage 2.1: Qh cross-attends with Memory Bank
        Qh, _ = self.history_decoder_mem(Qh, memory_features)
        
        # Stage 2.2: Qh cross-attends with updated Scene Queries (Qs)
        Qh, _ = self.history_decoder_scene(Qh, Qs)

        # MLP Heads
        # ----------------------------------------------------
        # 输出到LLM空间的标记
        x_h = self.llm_head_history(Qh)
        x_s = self.llm_head_scene(Qs)

        # 感知查询的辅助输出
        det_logits = self.det_head(Qp)  # (B, Np, C_det)
        traffic_logits = self.traffic_head(Qp.mean(dim=1))  # (B, C_traffic)
        motion_params = self.motion_head(Qp.mean(dim=1))  # (B, 4)

        # 规划token：融合x_h与x_s（平均后拼接的简化为均值再MLP）
        fused = (x_h.mean(dim=1) + x_s.mean(dim=1)) * 0.5
        planning_token = self.planning_head(fused)

        # 将更新后的Q_h写回记忆（FIFO）
        self.update_memory(Qh.detach())

        # 返回附加信息供上层使用（可选）
        self._aux_outputs = {
            'det_logits': det_logits,
            'traffic_logits': traffic_logits,
            'motion_params': motion_params,
        }

        return x_h, x_s, planning_token
    
    def get_attention_maps(self, current_features: torch.Tensor) -> torch.Tensor:
        """
        获取注意力图（用于可视化）
        
        Args:
            current_features: (B, D) 当前帧特征
            
        Returns:
            attention_maps: (B, num_heads, Q, M+1) 注意力权重
        """
        batch_size = current_features.size(0)
        
        # 历史对记忆的注意力权重可视化
        memory_features = self.get_memory_features(batch_size)
        Qh = self.query_proj(self.history_queries).unsqueeze(0).expand(batch_size, -1, -1)
        _, attention_weights = self.cross_attention_layers[0](Qh, memory_features)
        
        return attention_weights
    
    def get_memory_usage(self) -> float:
        """获取记忆库使用率"""
        return self.memory_length.item() / self.memory_size
