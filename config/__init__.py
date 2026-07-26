"""
配置模块

提供配置文件加载功能。
"""

import yaml
from pathlib import Path


def load_config(path="config/config.yaml"):
    """
    加载 YAML 配置文件

    Args:
        path: 配置文件路径，相对于项目根目录

    Returns:
        配置字典
    """
    config_file = Path(__file__).parent.parent / path
    with open(config_file) as f:
        return yaml.safe_load(f)