# ORION: A Holistic End-to-End Autonomous Driving Framework

本项目是基于论文《ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation》的实现，提供了一个完整的端到端自动驾驶框架。

## 🌟 主要特性

- **QT-Former**: 基于查询的时序模块，具有记忆库和交叉注意力机制
- **LLM接口**: 可插拔的大语言模型后端，支持推理和VQA任务
- **生成式规划器**: 条件轨迹VAE，将推理空间与动作空间统一
- **统一训练目标**: VQA损失 + 轨迹重建损失 + KL散度 + 对齐损失
- **模块化设计**: 清晰的代码结构，易于扩展和维护

## 🏗️ 项目架构

```
orion/
├── orion/                      # 核心包
│   ├── __init__.py
│   ├── config/                 # 配置管理
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   └── model_configs.py
│   ├── models/                 # 模型实现
│   │   ├── __init__.py
│   │   ├── vision/
│   │   │   ├── __init__.py
│   │   │   └── backbone.py
│   │   ├── temporal/
│   │   │   ├── __init__.py
│   │   │   └── qt_former.py
│   │   ├── reasoning/
│   │   │   ├── __init__.py
│   │   │   ├── llm_interface.py
│   │   │   └── vqa_head.py
│   │   ├── planning/
│   │   │   ├── __init__.py
│   │   │   └── generative_planner.py
│   │   └── orion_system.py
│   ├── data/                   # 数据处理
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   └── utils.py
│   ├── training/               # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── visualization.py
│   │   └── io.py
│   └── scripts/                # 可执行脚本
│       ├── __init__.py
│       ├── train.py
│       ├── evaluate.py
│       └── demo.py
├── tests/                      # 测试文件
├── examples/                   # 使用示例
├── configs/                    # 配置文件
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd orion

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 基础使用

```python
from orion.models import OrionSystem
from orion.config import OrionConfig

# 创建配置
config = OrionConfig()

# 初始化模型
model = OrionSystem(config)

# 训练
orion-train --config configs/default.yaml

# 评估
orion-eval --config configs/default.yaml --checkpoint path/to/checkpoint

# 演示
orion-demo --config configs/demo.yaml
```

## 📖 文档

详细的API文档和使用指南请参考 `docs/` 目录。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可证

本项目基于MIT许可证开源。

## 📚 引用

如果您在研究中使用了本项目，请引用原始论文：

```bibtex
@article{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Fu, Haoyu and Zhang, Diankun and Zhao, Zongchuang and others},
  journal={arXiv preprint arXiv:2503.19755},
  year={2025}
}
```
