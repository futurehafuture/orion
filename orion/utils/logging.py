"""日志工具"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
import colorlog


def setup_logging(log_dir: str = "logs", 
                 name: str = "orion",
                 level: str = "INFO",
                 console: bool = True,
                 file_log: bool = True) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录
        name: 日志器名称
        level: 日志级别
        console: 是否输出到控制台
        file_log: 是否保存到文件
        
    Returns:
        配置好的日志器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取或创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 控制台处理器（带颜色）
    if console:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # 彩色格式
        color_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        color_formatter = colorlog.ColoredFormatter(
            color_format,
            datefmt=date_format,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file_log:
        # 按日期命名日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 文件格式（不带颜色）
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 创建符号链接到最新日志
        latest_log = os.path.join(log_dir, f"{name}_latest.log")
        if os.path.exists(latest_log):
            os.remove(latest_log)
        if os.name != 'nt':  # Unix系统
            os.symlink(os.path.basename(log_file), latest_log)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


class LoggerContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, extra_info: dict = None):
        self.logger = logger
        self.extra_info = extra_info or {}
        self.original_extra = getattr(logger, '_extra_info', {})
    
    def __enter__(self):
        # 设置额外信息
        self.logger._extra_info = {**self.original_extra, **self.extra_info}
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始信息
        self.logger._extra_info = self.original_extra


class ProgressLogger:
    """进度日志器"""
    
    def __init__(self, logger: logging.Logger, total: int, description: str = "Progress"):
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0
        self.last_percentage = -1
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        percentage = int(100 * self.current / self.total)
        
        # 每10%记录一次
        if percentage >= self.last_percentage + 10:
            self.logger.info(f"{self.description}: {percentage}% ({self.current}/{self.total})")
            self.last_percentage = percentage
    
    def finish(self):
        """完成进度"""
        self.logger.info(f"{self.description}: Completed ({self.total}/{self.total})")


class TimedLogger:
    """计时日志器"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration.total_seconds():.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration.total_seconds():.2f}s: {exc_val}")


class MetricsLogger:
    """指标日志器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_history = []
    
    def log_metrics(self, metrics: dict, step: int = None, prefix: str = ""):
        """记录指标"""
        # 添加到历史
        entry = {"step": step, "metrics": metrics.copy()}
        if prefix:
            entry["prefix"] = prefix
        self.metrics_history.append(entry)
        
        # 格式化并记录
        metric_str = self._format_metrics(metrics, prefix)
        if step is not None:
            self.logger.info(f"Step {step} - {metric_str}")
        else:
            self.logger.info(metric_str)
    
    def _format_metrics(self, metrics: dict, prefix: str = "") -> str:
        """格式化指标字符串"""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{prefix}{key}: {value:.4f}")
            else:
                formatted.append(f"{prefix}{key}: {value}")
        return " | ".join(formatted)
    
    def log_summary(self, num_recent: int = 10):
        """记录最近指标的摘要"""
        if not self.metrics_history:
            return
        
        recent_metrics = self.metrics_history[-num_recent:]
        
        # 计算平均值
        all_keys = set()
        for entry in recent_metrics:
            all_keys.update(entry["metrics"].keys())
        
        averages = {}
        for key in all_keys:
            values = []
            for entry in recent_metrics:
                if key in entry["metrics"] and isinstance(entry["metrics"][key], (int, float)):
                    values.append(entry["metrics"][key])
            if values:
                averages[key] = sum(values) / len(values)
        
        self.logger.info(f"Average metrics (last {len(recent_metrics)} steps):")
        summary_str = self._format_metrics(averages)
        self.logger.info(summary_str)


class FileLogger:
    """文件专用日志器"""
    
    def __init__(self, file_path: str, mode: str = 'a'):
        self.file_path = file_path
        self.mode = mode
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def log(self, message: str, timestamp: bool = True):
        """写入日志消息"""
        with open(self.file_path, self.mode, encoding='utf-8') as f:
            if timestamp:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{now}] {message}\n")
            else:
                f.write(f"{message}\n")
    
    def log_dict(self, data: dict, separator: str = ": "):
        """记录字典数据"""
        for key, value in data.items():
            self.log(f"{key}{separator}{value}")
    
    def log_list(self, items: list, prefix: str = "- "):
        """记录列表数据"""
        for item in items:
            self.log(f"{prefix}{item}")


def create_experiment_logger(experiment_name: str, 
                           base_dir: str = "experiments") -> tuple[logging.Logger, str]:
    """
    为实验创建专门的日志器
    
    Args:
        experiment_name: 实验名称
        base_dir: 基础目录
        
    Returns:
        (logger, experiment_dir)
    """
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    log_dir = os.path.join(experiment_dir, "logs")
    
    # 设置日志器
    logger = setup_logging(
        log_dir=log_dir,
        name=experiment_name,
        level="INFO",
        console=True,
        file_log=True
    )
    
    logger.info(f"Experiment '{experiment_name}' started")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    return logger, experiment_dir
