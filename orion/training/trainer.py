"""ORION训练器"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR
from typing import Dict, Optional, Any, List
import logging
from tqdm import tqdm
import wandb
from collections import defaultdict
import json

from ..config import OrionConfig
from ..models import OrionSystem
from ..data import DrivingDataset
from .losses import OrionLoss
from .metrics import OrionMetrics
from ..utils import save_checkpoint, load_checkpoint, setup_logging


class OrionTrainer:
    """
    ORION系统训练器
    
    支持功能：
    - 多GPU训练
    - 混合精度训练
    - 梯度累积
    - 学习率调度
    - 早停机制
    - Wandb日志
    - 断点续训
    """
    
    def __init__(self, config: OrionConfig, model: Optional[OrionSystem] = None):
        self.config = config
        self.device = self._setup_device()
        
        # 设置日志
        self.logger = setup_logging(config.log_dir, "trainer")
        
        # 初始化模型
        if model is None:
            self.model = OrionSystem(config)
        else:
            self.model = model
        self.model = self.model.to(self.device)
        
        # 损失函数和评估指标
        self.loss_fn = OrionLoss(config.training)
        self.metrics = OrionMetrics()
        
        # 优化器和调度器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 混合精度训练
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # 历史记录
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # Wandb初始化
        if hasattr(config, 'use_wandb') and config.use_wandb:
            self._setup_wandb()
    
    def _setup_device(self) -> torch.device:
        """设置设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """设置优化器"""
        if self.config.training.optimizer.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
        
        self.logger.info(f"Optimizer: {optimizer.__class__.__name__}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """设置学习率调度器"""
        if self.config.training.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler.lower() == "linear":
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.training.min_lr / self.config.training.learning_rate,
                total_iters=self.config.training.epochs
            )
        elif self.config.training.scheduler.lower() == "exponential":
            gamma = (self.config.training.min_lr / self.config.training.learning_rate) ** (1.0 / self.config.training.epochs)
            scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        else:
            scheduler = None
        
        if scheduler:
            self.logger.info(f"Scheduler: {scheduler.__class__.__name__}")
        return scheduler
    
    def _setup_wandb(self):
        """设置Wandb"""
        wandb.init(
            project="orion-autonomous-driving",
            name=self.config.experiment_name,
            config=self.config.__dict__
        )
        wandb.watch(self.model, log="all", log_freq=100)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch, training=True)
                    losses = self.loss_fn(outputs, batch)
                    total_loss = losses["total_loss"] / self.config.training.accumulation_steps
            else:
                outputs = self.model(batch, training=True)
                losses = self.loss_fn(outputs, batch)
                total_loss = losses["total_loss"] / self.config.training.accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.training.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 更新指标
            with torch.no_grad():
                batch_metrics = self.metrics.compute(outputs, batch)
                for key, value in losses.items():
                    epoch_metrics[f"train_{key}"] += value.item()
                for key, value in batch_metrics.items():
                    epoch_metrics[f"train_{key}"] += value
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # 记录到Wandb
            if hasattr(self, 'wandb') and self.global_step % 100 == 0:
                log_dict = {f"train_step_{k}": v.item() if hasattr(v, 'item') else v 
                           for k, v in losses.items()}
                log_dict.update({f"train_step_{k}": v for k, v in batch_metrics.items()})
                log_dict["learning_rate"] = current_lr
                log_dict["global_step"] = self.global_step
                wandb.log(log_dict, step=self.global_step)
        
        # 计算epoch平均指标
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(batch, training=False)
                losses = self.loss_fn(outputs, batch)
                
                # 更新指标
                batch_metrics = self.metrics.compute(outputs, batch)
                for key, value in losses.items():
                    epoch_metrics[f"val_{key}"] += value.item()
                for key, value in batch_metrics.items():
                    epoch_metrics[f"val_{key}"] += value
                
                # 更新进度条
                pbar.set_postfix({
                    'val_loss': f"{losses['total_loss'].item():.4f}"
                })
        
        # 计算epoch平均指标
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """完整训练流程"""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config.training.epochs}")
        self.logger.info(f"Steps per epoch: {len(train_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            self.train_history["epoch"].append(epoch)
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                for key, value in val_metrics.items():
                    self.val_history[key].append(value)
                
                # 早停检查
                current_metric = val_metrics.get("val_total_loss", float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    # 保存最佳模型
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                if self.config.training.early_stopping and \
                   self.patience_counter >= self.config.training.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 记录日志
            log_str = f"Epoch {epoch + 1}/{self.config.training.epochs}"
            for key, value in train_metrics.items():
                log_str += f" | {key}: {value:.4f}"
            if val_loader is not None:
                for key, value in val_metrics.items():
                    log_str += f" | {key}: {value:.4f}"
            self.logger.info(log_str)
            
            # Wandb记录
            if hasattr(self, 'wandb'):
                log_dict = {**train_metrics}
                if val_loader is not None:
                    log_dict.update(val_metrics)
                log_dict["epoch"] = epoch
                wandb.log(log_dict, step=epoch)
            
            # 定期保存
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # 保存训练历史
        self.save_training_history()
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_epoch_{self.current_epoch}.pt"
        )
        save_checkpoint(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            save_checkpoint(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
        
        # 清理旧检查点
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.train_history = defaultdict(list, checkpoint.get('train_history', {}))
        self.val_history = defaultdict(list, checkpoint.get('val_history', {}))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点文件"""
        if self.config.training.keep_checkpoints <= 0:
            return
        
        checkpoint_files = []
        for file in os.listdir(self.config.checkpoint_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                epoch_num = int(file.split("_")[2].split(".")[0])
                checkpoint_files.append((epoch_num, file))
        
        # 按epoch排序，保留最新的几个
        checkpoint_files.sort(key=lambda x: x[0])
        files_to_delete = checkpoint_files[:-self.config.training.keep_checkpoints]
        
        for _, filename in files_to_delete:
            file_path = os.path.join(self.config.checkpoint_dir, filename)
            try:
                os.remove(file_path)
                self.logger.debug(f"Deleted old checkpoint: {filename}")
            except OSError:
                pass
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'config': self.config.__dict__
        }
        
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """在测试集上评估"""
        self.logger.info("Starting evaluation...")
        
        self.model.eval()
        test_metrics = defaultdict(float)
        num_batches = len(test_loader)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluation")
            
            for batch in pbar:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 预测
                predictions = self.model.predict(
                    batch["image"], 
                    batch.get("text_prompt", None)
                )
                
                # 计算指标
                outputs = self.model(batch, training=False)
                losses = self.loss_fn(outputs, batch)
                batch_metrics = self.metrics.compute(outputs, batch)
                
                # 累积指标
                for key, value in losses.items():
                    test_metrics[f"test_{key}"] += value.item()
                for key, value in batch_metrics.items():
                    test_metrics[f"test_{key}"] += value
                
                # 收集预测结果
                all_predictions.append(predictions)
                all_targets.append({
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                })
        
        # 计算平均指标
        for key in test_metrics:
            test_metrics[key] /= num_batches
        
        # 计算详细评估指标
        detailed_metrics = self.metrics.compute_detailed_metrics(
            all_predictions, all_targets
        )
        test_metrics.update(detailed_metrics)
        
        # 记录结果
        self.logger.info("Evaluation Results:")
        for key, value in test_metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        return dict(test_metrics)
