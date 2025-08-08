"""输入输出工具"""

import os
import torch
import json
import pickle
import yaml
from typing import Any, Dict, Optional
import logging
from pathlib import Path


def save_checkpoint(checkpoint: Dict[str, Any], 
                   checkpoint_path: str,
                   is_best: bool = False) -> None:
    """
    保存模型检查点
    
    Args:
        checkpoint: 检查点数据
        checkpoint_path: 保存路径
        is_best: 是否为最佳模型
    """
    # 创建目录
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，创建副本
    if is_best:
        best_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "best_" + os.path.basename(checkpoint_path)
        )
        torch.save(checkpoint, best_path)
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, 
                   device: torch.device = None,
                   map_location: str = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        device: 目标设备
        map_location: 映射位置
        
    Returns:
        检查点数据
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if map_location is None and device is not None:
        map_location = str(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return checkpoint


def save_config(config: Any, config_path: str, format: str = "yaml") -> None:
    """
    保存配置文件
    
    Args:
        config: 配置对象
        config_path: 保存路径
        format: 文件格式 ('yaml', 'json')
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # 转换为字典
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    elif hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    # 保存文件
    if format.lower() == "yaml":
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    elif format.lower() == "json":
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Config saved to {config_path}")


def load_config(config_path: str, format: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        format: 文件格式，如果为None则根据扩展名推断
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 推断格式
    if format is None:
        ext = os.path.splitext(config_path)[1].lower()
        if ext in ['.yaml', '.yml']:
            format = 'yaml'
        elif ext == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot infer format from extension: {ext}")
    
    # 加载文件
    with open(config_path, 'r', encoding='utf-8') as f:
        if format.lower() == 'yaml':
            config = yaml.safe_load(f)
        elif format.lower() == 'json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Config loaded from {config_path}")
    return config


def save_model_weights(model: torch.nn.Module, 
                      weights_path: str,
                      optimizer: torch.optim.Optimizer = None,
                      scheduler: Any = None,
                      epoch: int = None,
                      metadata: Dict[str, Any] = None) -> None:
    """
    保存模型权重
    
    Args:
        model: 模型
        weights_path: 权重保存路径
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        metadata: 额外元数据
    """
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        save_dict['optimizer_class'] = optimizer.__class__.__name__
    
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        save_dict['scheduler_class'] = scheduler.__class__.__name__
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, weights_path)
    logging.info(f"Model weights saved to {weights_path}")


def load_model_weights(model: torch.nn.Module,
                      weights_path: str,
                      optimizer: torch.optim.Optimizer = None,
                      scheduler: Any = None,
                      device: torch.device = None,
                      strict: bool = True) -> Dict[str, Any]:
    """
    加载模型权重
    
    Args:
        model: 模型
        weights_path: 权重文件路径
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 目标设备
        strict: 是否严格匹配状态字典
        
    Returns:
        加载的元数据
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    map_location = str(device) if device else None
    checkpoint = torch.load(weights_path, map_location=map_location)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logging.info("Model weights loaded")
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Optimizer state loaded")
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info("Scheduler state loaded")
    
    # 返回元数据
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'metadata': checkpoint.get('metadata', {}),
        'model_class': checkpoint.get('model_class', 'Unknown'),
    }
    
    logging.info(f"Weights loaded from {weights_path}")
    return metadata


def save_pickle(obj: Any, file_path: str) -> None:
    """保存pickle文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    logging.info(f"Pickle saved to {file_path}")


def load_pickle(file_path: str) -> Any:
    """加载pickle文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    logging.info(f"Pickle loaded from {file_path}")
    return obj


def save_text(text: str, file_path: str, encoding: str = 'utf-8') -> None:
    """保存文本文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(text)
    
    logging.info(f"Text saved to {file_path}")


def load_text(file_path: str, encoding: str = 'utf-8') -> str:
    """加载文本文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()
    
    logging.info(f"Text loaded from {file_path}")
    return text


def ensure_dir(dir_path: str) -> str:
    """确保目录存在"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    if not os.path.exists(file_path):
        return 0
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def copy_file(src: str, dst: str, create_dirs: bool = True) -> None:
    """复制文件"""
    import shutil
    
    if create_dirs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    shutil.copy2(src, dst)
    logging.info(f"File copied from {src} to {dst}")


def move_file(src: str, dst: str, create_dirs: bool = True) -> None:
    """移动文件"""
    import shutil
    
    if create_dirs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    shutil.move(src, dst)
    logging.info(f"File moved from {src} to {dst}")


def list_files(directory: str, 
              pattern: str = "*",
              recursive: bool = False,
              include_dirs: bool = False) -> list[str]:
    """列出目录中的文件"""
    from glob import glob
    
    if recursive:
        pattern = os.path.join(directory, "**", pattern)
        files = glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, pattern)
        files = glob(pattern)
    
    if not include_dirs:
        files = [f for f in files if os.path.isfile(f)]
    
    return sorted(files)


def cleanup_old_files(directory: str,
                     pattern: str = "*",
                     keep_count: int = 5,
                     dry_run: bool = False) -> list[str]:
    """清理旧文件，保留最新的几个"""
    files = list_files(directory, pattern)
    
    if len(files) <= keep_count:
        return []
    
    # 按修改时间排序，保留最新的
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    files_to_delete = files[keep_count:]
    
    if not dry_run:
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                logging.info(f"Deleted old file: {file_path}")
            except OSError as e:
                logging.warning(f"Failed to delete {file_path}: {e}")
    else:
        logging.info(f"Would delete {len(files_to_delete)} old files")
    
    return files_to_delete


class FileManager:
    """文件管理器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, *paths: str) -> Path:
        """获取相对于基础目录的路径"""
        return self.base_dir / Path(*paths)
    
    def save_json(self, data: Any, *paths: str, **kwargs) -> Path:
        """保存JSON文件"""
        file_path = self.get_path(*paths)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str, **kwargs)
        
        return file_path
    
    def load_json(self, *paths: str) -> Any:
        """加载JSON文件"""
        file_path = self.get_path(*paths)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], *paths: str) -> Path:
        """保存检查点"""
        file_path = self.get_path(*paths)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, file_path)
        return file_path
    
    def exists(self, *paths: str) -> bool:
        """检查文件是否存在"""
        return self.get_path(*paths).exists()
    
    def list_files(self, pattern: str = "*", recursive: bool = False) -> list[Path]:
        """列出文件"""
        if recursive:
            return list(self.base_dir.rglob(pattern))
        else:
            return list(self.base_dir.glob(pattern))
