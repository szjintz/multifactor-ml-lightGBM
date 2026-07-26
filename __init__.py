"""
包初始化文件

本项目是一个基于 LightGBM 和 Qlib 的多因子量化投资策略框架。
主要功能模块包括：
- data: 数据获取（Qlib + Akshare）
- factors: 因子计算（量价、基本面、另类）
- features: 特征工程和选择
- model: 模型训练（Walk-Forward + Optuna）
- portfolio: 组合优化（CVXPY）
- backtest: 回测引擎
- risk: 风险分析（蒙特卡洛、拥挤度）
- interpretation: 模型可解释性（SHAP、归因）
- scripts: 执行脚本和分析工具
"""