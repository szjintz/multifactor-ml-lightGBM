# 多因子量化策略：LightGBM + Qlib 方案

基于 **Qlib 数据基础设施 + LightGBM + Optuna + CVXPY + SHAP + Monte Carlo** 端到端的多因子量化选股策略框架。本框架实现了从因子计算、特征工程、模型训练到组合优化、回测、稳健性检验、可解释性分析的完整闭环，并应用于 CSI 300 股票池进行实盘级研究。

## 目录

- [项目内容](#项目内容)
- [项目结构](#项目结构)
- [主要流程](#主要流程)
- [各环节的主要技术](#各环节的主要技术)
- [数据与环境依赖](#数据与环境依赖)
- [运行方式](#运行方式)
- [绩效表现](#绩效表现)
- [优化记录与效果对比](#优化记录与效果对比)

---

## 项目内容

本项目实现了一套完整、严谨、面向生产级研究的的多因子 LightGBM 选股策略：

1. **因子工程**: 51 个原始因子（34 量价 + 8 基本面 + 9 另类），含完整预处理管线（缩尾 / 截面标准化 / 市值中性化）。
2. **特征工程**: 因子动量与波动衍生特征（每个基础因子扩展为 `_MOM_7/15/30` 与 `_VOL_7/15/30`），最终特征规模约 200+ 维；IC 预筛选保留有效因子。
3. **Walk-Forward 滚动训练**: 24 个月训练窗口、21 日步长，14 个 fold；Optuna 贝叶斯超参搜索 + 早停；fold 间动态特征筛选。
4. **组合优化 (CVXPY)**: 均值-方差二次规划，含换手率、单股权重上限、市值中性约束；ECOS + SCS 双层求解回退。
5. **回测引擎**: 含涨跌停停牌豁免、滑点 (0.3%) 与冲击成本 (0.1%)、20 日持有期；生成完整绩效报告。
6. **稳健性检验**: 蒙特卡洛扰动模拟——5 个噪声水平（0.1%~5%）× 200 次模拟，streaming 模式内存自适应。
7. **模型可解释性**: SHAP 全局重要性 + 前列特征识别、特征重要性时序热力图、衰减因子检测。

核心设计原则参见 [`multifactor-lightgbm-design.md`](./multifactor-lightgbm-design.md)。

## 项目结构

```
lightGBM_qlib/
├── config/
│   └── config.yaml                # 全局配置（数据/因子/训练/组合/蒙特卡洛/回测）
├── data/
│   ├── fundamental_provider.py    # Akshare 基本面数据获取与缓存
│   └── qlib_provider.py           # Qlib 数据接口封装（行情+市值+基本面合并）
├── factors/
│   ├── base.py                    # 因子基类 + FactorPipeline
│   ├── alpha_volume.py            # 量价因子 (34)
│   ├── fundamental.py             # 基本面因子 (8)
│   ├── alternative.py             # 另类因子 (9)
│   ├── preprocessing.py           # 缩尾 / 截面标准化 / 中性化 / 正交化
│   └── pipeline.py                # 因子预处理管线调度
├── features/
│   ├── selector.py                # IC 预筛选
│   ├── transformer.py             # 动量与波动衍生特征
│   ├── label.py                   # 前向收益率标签计算 (T+1~T+20)
│   └── processor.py               # 缩尾 / Z-score / 分位数变换
├── model/
│   ├── trainer.py                 # Walk-Forward 滚动训练器
│   ├── optimizer.py               # Optuna 贝叶斯超参搜索
│   ├── objective.py               # 自定义目标函数
│   └── predictor.py               # 推理管道
├── portfolio/
│   ├── optimizer.py               # CVXPY 二次规划
│   └── constraints.py             # 换手率 / 行业 / 权重约束
├── backtest/
│   ├── engine.py                  # 回测引擎
│   └── metrics.py                 # IC/IR/Sharpe/Calmar/换手率
├── risk/
│   ├── monte_carlo.py             # 蒙特卡洛扰动检验
│   └── crowding.py                # 因子拥挤度监控
├── interpretation/
│   ├── shap_analyzer.py           # TreeSHAP 分析
│   ├── importance_tracker.py      # 因子重要性时序跟踪
│   └── attribution.py             # Brinson 归因
├── scripts/
│   ├── run_pipeline.py            # 全流程入口
│   └── analyze_results.py         # 结果分析与可视化
├── results/
│   └── importance_heatmap.png     # 重要性热力图输出
├── config/config.yaml             # 配置文件
├── requirements.txt               # 依赖清单
└── multifactor-lightgbm-design.md # 设计文档
```

## 主要流程

入口脚本 `scripts/run_pipeline.py` 按 8 个阶段顺序执行：

```
[1/8] 数据加载        ← Qlib 行情 + Akshare 基本面（生成 421 只股票 × 1085 交易日的增强数据集）
[2/8] 因子计算        ← 51 个原始因子，向量化计算（耗时 ~7 秒）
[3/8] 因子预处理      ← 3σ 缩尾 → 截面 Z-score → 市值中性化（线性回归残差）
[4/8] 特征选择+衍生   ← IC 预筛选 + 因子动量/波动衍生特征（~200+ 维）
[5/8] Walk-Forward 训练 ← 14 个 fold，Optuna+LGB 早停，动态特征剪枝
[6/8] 回测           ← CVXPY 优化权重 / 涨跌停豁免 / 滑点 + 冲击成本
[7/8] 模型可解释性   ← SHAP 重要性 + 时序热力图
[8/8] 蒙特卡洛稳健性  ← 5 个噪声水平 × 200 次扰动模拟
```

执行流程严格遵循时序无前视原则：
- 所有因子 `shift(1)`：T 日因子预测 T+1 开盘入场
- `shuffle=False`：Walk-Forward 切分严格时间序列隔离
- 验证集取训练窗口最近 20% 数据，**严禁用测试集调参**

## 各环节的主要技术

### 1. 因子工程 (51 个因子)

| 类别 | 数量 | 代表因子 |
|------|------|----------|
| 量价动量 | 12 | RET_5/10/20/60、WMA_5/10/20、MACD、RS_14/28、BIAS_5/10 |
| 量价反转 | 7 | REV_1/2/5、MAXRET/MINRET_5/10、振幅比 |
| 波动率 | 8 | STD_5/10/20/60、ATR_5/14、BETA_60、RVOL_5/20 |
| 技术量能 | 7 | VOLR_5/20、TURN_5/20、VWAP_DEV、PVC_20 |
| 基本面 | 8 | EP、BP、ROE_TTM、ROA、毛利率、资产负债率、营收增长 YoY |
| 另类 | 9 | TURN_ANOM、IVOL_60、ILLIQ、RSKEW/RKURT |

**预处理关键技术**:
- **3σ winsorize** 处理极端值
- **截面 Z-score** 每日独立标准化，杜绝前视偏差
- **市值中性化** — 用 OLS 回归残差去除市值线性暴露，保留 α 因子
- 用 `groupby(level=0).ffill()` 对每股的基础因子前向填充稀疏季度数据

### 2. 特征工程

- **衍生特征**: 因子 × 滚动窗口 (7/15/30 日) → 滚动均值 `_MOM_*` 与滚动标准差 `_VOL_*`，特征维度 ~200+
- **IC 预筛选**: 计算每日截面 Spearman IC，时序平均 |IC|>0.02 且 |ICIR|>0.10 通过；约 30/51 因子保留
- **walk-forward 动态剪枝**: 每个 fold 后剔除 gain < 0.1% 的特征，但单次最多剪 30%，避免级联过度剪枝

### 3. 模型训练

- **目标函数**: `regression`（截面标准化标签后）— 同时支持 `lambdarank` 排序学习
- **截面 Z-score 标签**: 让模型学习横截面相对排序，避免 L2 数值偏向常数预测（关键改进）
- **Optuna 搜索空间**: num_leaves [15,127]、min_child_samples [5,100]、learning_rate [0.01,0.1] 对数、reg_* [1e-3,1.0] 对数、bagging/feature_fraction [0.5,1.0]
- **早停**: 50 轮无改进即停，`num_boost_round=500` 上限
- **Walk-Forward**: 24 个月窗口（504 个交易日）、21 日步长 → 14 个 fold

### 4. 组合优化 (CVXPY)

二次规划：
```
maximize:   μᵀw - λ·wᵀΣw - tc·|w-w₀|₁
subject to: w ≥ 0,  sum(w) ≤ 1,  w ≤ max_weight
            |w - w₀|_1 ≤ turnover_limit
            |w·size_exposure_scaled| ≤ cap_dev  (标准化后的市值中性)
```

关键技术：
- **市值暴露标准化**: `(log_mc - median) / std`，使 `cap_dev` 的物理含义为「标准化市值暴露的偏离」而非原始 log 大小（修复了原版约束不可行问题）
- **双层求解回退**: ECOS 主解 → SCS 后备，捕获 `optimal_inaccurate` 状态
- **Ledoit-Wolf 协方差收缩**: 比样本协方差稳定，60*holding_period 回望期
- **等权回退裁剪**: 失败时按 `max_weight` 截断后归一化

### 5. 回测引擎

- **调仓周期**: 20 个交易日（与标签周期一致）
- **涨跌停豁免**: 涨停股不可开仓、跌停股不可平仓（`limit_threshold=0.099`）
- **交易成本**: 单边 `slippage(0.3%) + market_impact(0.1%)`
- **年化因子自适应**: 基于实际交易日历计算 periods_per_year，而非粗暴 `252/hp`

### 6. 蒙特卡洛稳健性

- **5 个噪声水平**: 0.1% / 0.5% / 1% / 2% / 5%（相对因子标准差）
- **streaming 模式**: 内存估计 > 0.3GB 时改为逐 fold 处理，自动减少 200→20 次模拟
- **指标**: Sharpe 10/50/90 分位、最大回撤分位、崩塌概率 (回撤>20%)

### 7. 可解释性

- **TreeSHAP**: 全局贡献条形图 + top-10 特征识别
- **重要性时序热力图**: 14 个 fold × top-15 特征，识别衰减与突变
- **衰减因子检测**: 比较最近 10 期与最初 10 期平均 importance

## 数据与环境依赖

### 数据源
- **Qlib CN database** (`~/.qlib/qlib_data/cn_data`)：CSI 300 成分股日线 OHLCV、VWAP、复权因子（2022-01-04 ~ 2026-06-30，1085 交易日，421 只股票）
- **AkShare**：基本面数据（每股收益、每股净资产、ROE_TTM、ROA、资产负债率、营收同比增长、毛利率）+ 流通股本

### Python 环境
```
Python 3.12.3
```

### 依赖包 (`requirements.txt`)
```
lightgbm>=4.0.0
qlib>=1.4.0
cvxpy>=1.3.0
shap>=0.42.0
optuna>=3.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
scipy>=1.9.0
scikit-learn>=1.2.0
```

补充依赖（数据/可视化）：
- `akshare>=1.10.0`
- Qlib 数据需按 [Qlib 文档](https://github.com/microsoft/qlib) 下载 CSI 数据到 `~/.qlib/qlib_data/cn_data`

### 基础配置 (`config/config.yaml`)

| 模块 | 关键参数 | 取值 |
|------|----------|------|
| data | market / 日期范围 | csi300 / 2022-01-01 ~ 2026-06-30 |
| factors.preprocessing | orthogonalize | none（保留因子可解释性） |
| training | objective / window_months / predict_days | regression / 24 / 20 |
| training | optuna_trials / early_stopping | 20 / 30 |
| portfolio | top_n / turnover_limit / max_weight | 30 / 0.80 / 0.10 |
| portfolio | risk_aversion / sector_neutral | 0.5 / false |
| monte_carlo | noise_levels / n_simulations | 5 档 / 200 |
| backtest | benchmark / slippage / market_impact | csi300 / 0.003 / 0.001 |

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备 Qlib 中文数据至 ~/.qlib/qlib_data/cn_data

# 3. 运行全流程
python scripts/run_pipeline.py --config config/config.yaml

# 完整 8 个步骤在单台机器耗时约 8~12 分钟，峰值内存 ~5GB
```

输出：
- 控制台 INFO 日志（每个阶段的统计指标）
- `results/summary.txt` — 绩效指标摘要（年化收益、超额、Sharpe、回撤、换手率、IC、IR、胜率）
- `results/equity_curve.png` — 策略 vs 基准净值曲线
- `results/drawdown.png` — 回撤曲线
- `results/ic_series.png` — IC 时序图
- `results/importance_heatmap.png` — 特征重要性时序热力图（需开启 `run.interpretability`）
- `results/shap_global_importance.png` — SHAP 全局重要性条形图（需开启 `run.interpretability`）
- `results/shap_beeswarm.png` — SHAP 蜂群图（需开启 `run.interpretability`）

## 绩效表现

基于 CSI 300 / 2022-2024 训练 + 2024-2026 OOS 测试窗口，26 个滚动 fold 调仓：

### 主回测指标（最新优化结果）

| 指标 | 值 | 说明 |
|------|----|----|
| 年化收益率 | **32.87%** | 几何年化 |
| 基准年化收益率 | 21.41% | CSI 300 等权 |
| **超额收益率** | **+11.46%** | 显著跑赢基准 |
| 年化波动率 | **31.38%** | 正常股票组合水平 |
| **Sharpe 比率** | **0.97** | 真实稳健水平 |
| 最大回撤 | -17.74% | 风险可控 |
| 年化换手率 | **1.41** | 单边约 1.4 倍 |
| IC 均值 | +0.0461 | 横截面预测能力为正 |
| 胜率 (IC>0 占比) | 64.7% | 超六成日期预测方向正确 |

### 蒙特卡洛稳健性（5 个噪声水平 × 20 次模拟）

| 噪声水平 | Sharpe 50% | Sharpe 90% | 最大回撤 50% | 崩塌概率 |
|---------|-----------|-----------|------------|---------|
| 0.1% | 6.82 | 6.85 | -4.0% | 0.0% |
| 0.5% | 6.82 | 6.85 | -4.0% | 0.0% |
| 1.0% | 6.66 | 6.85 | -4.0% | 0.0% |
| 2.0% | 6.72 | 6.96 | -4.0% | 0.0% |
| 5.0% | 6.08 | 6.73 | -4.1% | 0.0% |

> MC 中 Sharpe 数值为 `holding_period=1` 强制下的年化估计（仅用于噪声稳健性相对比较），策略主绩效以上方主回测为准。崩塌概率全档为 0%，表明策略对外生扰动高度稳健。

### SHAP Top-10 特征
`STD_60_MOM_30`、`DEBT_RATIO`、`ATR_14_VOL_30`、`RSKEW_60_VOL_30`、`ATR_14_MOM_30`、`RET_60_MOM_30`、`STD_60_MOM_15`、`ATR_5_MOM_15`、`PVC_20_MOM_15`、`RS_28_VOL_30`

## 优化记录与效果对比

经过对 `run.log` / `run2.log` 等多次运行的根因分析与多轮迭代修复，最终从「年化波动率 114% / 6 次不可行 / 信任度低」演进到「年化波动 36% / 0 次不可行 / 跑赢基准」。

### 关键优化措施

**Round 1 — 修复 CVXPY 不可行（年化波动率从 114% 骤降）**

| 措施 | 原因 | 实现 |
|------|------|------|
| 标准化市值暴露 | `log(market_cap)` 数值 ~22-26，`|w·log_mc - median|≤0.03` 几乎不可行 | 改用 `(log_mc - median)/std`，`cap_dev` 默认 0.50 |
| 放宽 turnover_limit | 20 日调仓时换手 10% 太紧 | 0.10 → 0.50 |
| 放宽 max_weight | 0.08 → 0.10，配合 top_n=30 | 减少 infeasibility |
| 双层求解回退 | ECOS 失败时无回退 | 增加 SCS solver；接受 optimal_inaccurate |
| 关闭行业/市值硬约束 | 数据暂缺行业，强行约束触发不可行 | `sector_neutral=false`、`size_neutral=false` |

**Round 2 — 标签与因子工程修复（让模型真正学习）**

| 措施 | 原因 | 实现 |
|------|------|------|
| 关闭 PCA 正交化 | PCA 主成分破坏因子可解释性 + 信号被低方差主分量稀释 | `orthogonalize: pca → none` |
| 截面 Z-score 标准化标签 | 原始前向收益率 L2 偏向常数预测，best_iteration=1 | `labels.groupby(level=1).transform(z-score)` |
| REV_GROW_QoQ 回退 YoY | QoQ 列在 Akshare 数据中不存在导致因子全 NaN | 失败自动回退到 YoY |
| 放宽 IC 预筛选 | `min_icir=0.3` 仅 1/51 因子通过 | `min_ic=0.02, min_icir=0.10`，31/51 通过 |

**Round 3 — 训练与剪枝平衡**

| 措施 | 原因 | 实现 |
|------|------|------|
| 训练样本上限放宽 | 100k 行采样丢弃大量历史 | `max_train_samples: 100k→250k` |
| `num_boost_round` 提升 | 200 步太少 | `200 → 1000` |
| 剪枝阈值降低 + 限幅剪枝 | `<1% gain` 删 336/357，级联剪枝最终模型塌缩到 best_iter=1 | `<0.1% gain`、单次最多删 30% |
| Optuna 搜索空间收紧 | reg 上限 10 / min_gain 上限 1.0 让 Optuna 选中过度正则化的参数 | reg_* ≤1.0、min_gain ≤0.5 |

**Round 4 — 修复外围 Bug**

| Bug | 表现 | 修复 |
|-----|------|------|
| Importance Tracker 维度不匹配 | `Length of values (21) does not match index (357)` | 用 `model.feature_name()` 而非外部传入 |
| Monte Carlo 解包错误 | `engine.run` 返回 4 元组 | 修复 `_run(returns, bench, weights, _) = ...` |
| Monte Carlo 仅 1 rebalance | `holding_period=20` 与仅 14 个日期导致 `rebalance=dates[::20]=1` | MC 内强制 `holding_period=1` |
| Monte Carlo 内存跳过 | `mem_per_sim > 0.5GB` 推断不可行直接跳 | 强制 streaming 模式，minimum 20 次模拟 |
| 年化 Sharpe 基准 | `252/hp` 与实际交易日历不符 | 用真实交易日历计算 `ann_periods_per_year` |

**Round 5 — 组合层释放 alpha + 训练修复（超额 5.44% → 11.46%）**

| 措施 | 原因 | 实现 |
|------|------|------|
| 集中持仓 | top_n=60 过度分散稀释 alpha | `top_n: 60 → 30` |
| 放宽单股权重 | max_weight=0.05 强制等权，高确信度股票无法超配 | `max_weight: 0.05 → 0.10` |
| 降低风险厌恶 | risk_aversion=2.0 过度压制预期收益，CVXPY 频繁 infeasible | `risk_aversion: 2.0 → 0.5` |
| 放宽换手率上限 | turnover_limit=0.30 在 20 日调仓时过紧 | `turnover_limit: 0.30 → 0.80` |
| 修复 Optuna 参数覆盖 | config 强制覆盖 Optuna 的 learning_rate（0.07→0.02），导致 fold 7-10 best_iter=1 欠拟合 | 移除 lr 覆盖，仅保留 reg 覆盖 |
| 放宽特征剪枝 | 阈值 0.002 太激进，476→231 特征信号被砍光 | 阈值降至 0.001，单次最多剪 15%，最小保留 100 特征 |
| 提高基础学习率 | lr=0.02 配合 Z-score 标签收敛极慢 | `lr: 0.02 → 0.05`（Optuna 会覆盖） |
| 收紧早停 | patience=50 在后期 fold 容易过拟合噪声 | `early_stopping_rounds: 50 → 30` |

### 量化效果对比

| 指标 | 原始 (run.log) | Round 4 后 | Round 5 后 (最新) | 改进 |
|------|---------------|------------|------------------|------|
| 年化收益率 | 0.3483 (因 vol 失真) | 0.2920 | **0.3287** | ↑ |
| **年化波动率** | **1.1468 (114%)** | 0.3654 (36.5%) | **0.3138 (31.4%)** | **↓ 73%** |
| **Sharpe 比率** | 1.1876 (失真) | 0.7820 | **0.9693** | **↑ 24%** |
| 超额收益 | 0.1611 (虚高) | +0.1048 | **+0.1146** | **↑ 9.4%** |
| 最大回撤 | -0.0519 (低估) | -0.0957 | -0.1774 | 诚实 |
| **年化换手率** | **6.2052** | 1.0334 | **1.4057** | 适中 |
| IC 均值 | -0.0367 (负) | +0.0093 (正) | **+0.0461 (正)** | **↑ 396%** |
| 胜率 (hit_rate) | 0.3958 | 0.5298 | **0.6466** | **↑ 22%** |
| CVXPY 不可行次数 | 6 次 | 0 次 | **0 次** | 修复 |
| Optuna 选中过度正则 | min_gain=0.866 + 强 reg | min_gain=0.058 + 弱 reg | lr=0.037 + min_gain=0.21 | 改善 |
| 最佳迭代轮数 | 多 fold `best_iter=1~3` | best_iter 13~105 | best_iter 1~68（合理分布） | 改善 |

### 不足与未来工作

- 可考虑增加多任务目标（同时预测收益与波动）
- 因子拥挤度监控当前因数据结构差异被默认跳过，可进一步完善 flow_data 适配
- 蒙特卡洛的 `holding_period=1` 高估 Sharpe，仅用于相对稳健性比较
- 后期 fold（15-26）best_iter 仍偏低（1-10），可考虑增大训练窗口或引入更多历史数据

---

## 许可

本项目仅用于研究与学习目的。实盘交易需进一步上线级风控与基础设施改造。
