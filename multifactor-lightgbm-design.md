# 多因子策略：LightGBM + Qlib方案设计

## 1. 系统架构总览

```
Qlib 数据层 → 因子管线 → 特征工程
    → Walk-Forward 滚动训练 (LightGBM + Optuna)
    → 预测 → 组合优化 (CVXPY)
    → 回测 → 蒙特卡洛稳定性检验
    → 归因分析 (SHAP + Brinson)
```

### 核心设计原则

**Qlib 仅作为数据基础设施**。训练、优化、归因全部使用自定义管道，最大化灵活性并可深入各项技术。

### 项目结构

```
lightGBM_qlib/
├── config/
│   └── config.yaml                # 全局配置
├── data/
│   ├── fundamental_provider.py    # 基本面数据获取
│   └── qlib_provider.py           # Qlib 数据接口封装
├── factors/
│   ├── base.py                    # 因子基类
│   ├── alpha_volume.py            # 量价因子 (30+)
│   ├── fundamental.py             # 基本面因子 (15+)
│   ├── alternative.py             # 另类因子 (10+)
│   ├── preprocessing.py           # 中性化 / 正交化 / 标准化
│   └── pipeline.py                # 因子管线调度
├── features/
│   ├── selector.py                # IC 预筛选 + L1 + 后筛选
│   ├── transformer.py             # 衍生特征 & 截面特征
│   └── processor.py               # 缩尾 / Z-score / 分位数变换
├── model/
│   ├── trainer.py                 # Walk-Forward 滚动训练器
│   ├── objective.py               # 自定义目标函数 (Rank-L2 / 多任务)
│   ├── optimizer.py               # Optuna 贝叶斯超参搜索
│   └── predictor.py               # 推理管道
├── portfolio/
│   ├── optimizer.py               # CVXPY 二次规划
│   └── constraints.py             # 换手率 / 行业 / 权重约束
├── backtest/
│   ├── engine.py                  # 回测引擎
│   └── metrics.py                 # 绩效指标 (IC/IR/Sharpe/Calmar)
├── risk/
│   ├── monte_carlo.py             # 蒙特卡洛扰动检验
│   └── crowding.py                # 因子拥挤度监控
├── interpretation/
│   ├── shap_analyzer.py           # TreeSHAP 分析
│   ├── importance_tracker.py      # 因子重要性时序跟踪
│   └── attribution.py             # Brinson 归因
└── scripts/
    ├── run_pipeline.py            # 全流程入口
    └── analyze_results.py         # 结果分析与可视化
```

---

## 2. 因子工程 (55+ 因子)

### 2.1 因子分类

#### A. 量价因子 (30+)

| 类别     | 因子                                                                        | 数量 |
|---------|----------------------------------------------------------------------------|------|
| **动量** | RET_5/10/20/60, WMA_5/10/20, MACD, RS_14/28, BIAS_5/10                     |  12  |
| **反转** | RET_1/2/5_rev, MAXRET_5/10, MINRET_5/10, 振幅比                             |  7   |
| **波动** | STD_5/10/20/60, ATR_5/14, BETA_60, 已实现波动率_5/20                         |  8   |
| **技术** | VOLUME_RATIO_5/20, AMOUNT_RATIO_5/20, TURNOVER_1/5/20, VWAP偏离度, 量价相关性 |  8   |

#### B. 基本面因子 (15+)

| 类别     | 因子                                                 |
|---------|------------------------------------------------------|
| **估值** | EP, BP, SP, CP, 股息率, PEG                          |
| **质量** | ROE_TTM, ROA, 毛利率, 净利率, 经营现金流/营收, 资产负债率 |
| **成长** | 营收增长(QoQ/YoY), 净利润增长(QoQ/YoY), ROE增长         |

#### C. 另类因子 (10+)

| 类别       | 因子                                     |
|-----------|----------------------------------------- |
| **情绪**   | 换手率异常, 波动率异常, 资金流强度            |
| **特质**   | IVOL_FF3, 下行波动率, 已实现偏度, 已实现峰度  |
| **流动性** | Amihud非流动性指标, 换手率衰减, 日内振幅      |

### 2.2 因子预处理管线

```
原始因子
  ├→ Step 1: 异常值处理 (3σ 缩尾 + 1%/99% 分位数截断)
  ├→ Step 2: 截面 Z-score 标准化 (每日独立)
  ├→ Step 3: 市值 + 行业中性化 (线性回归残差)
  ├→ Step 4: 因子正交化 (Gram-Schmidt / PCA)
  ├→ Step 5: IC 时序特征 (滚动 IC、ICIR、IC 衰减)
  ├→ Step 6: 二阶交叉特征 (因子×市值、因子近N日动量)
  └→ Step 7: 特征筛选 (三阶段)
```

**关键规则：**
- **禁止全局标准化** — 必须使用截面标准化（前视偏差大坑）
- 所有因子 `shift(1)` — T 日因子预测 T+1 开盘入场

### 2.3 三阶段特征筛选

1. **前置过滤:** 平均 |IC| > 2σ, ICIR > 0.5, Bonferroni/FDR 校正 (p < 0.05)
2. **L1 正则:** `reg_alpha` 在训练中自动淘汰弱特征
3. **后筛选:** 滚动特征池 — 每期剔除 gain < 1% 的冗余特征

---

## 3. 标签设计

| 参数      | 值                                    |
|----------|---------------------------------------|
| **标签**  | `Ref(20, -1)` — T+1 到 T+20 累计收益率 |
| **Skip** | 跳过 T 日（规避 T+1 交割规则影响）        |
| **降噪**  | 若噪声过高：使用指数加权平滑收益替代原始收益 |

---

## 4. 模型训练策略

### 4.1 Walk-Forward 滚动框架

```
时间轴：
|←←←← 训练窗口 24个月 →→→→|← 预测 1个月 →|
T-24                         T-1   T        T+1
                                    ↑
                                 预测点

滚动：
  - 窗口：24个月训练，1个月预测
  - 步长：月度滚动
  - 严格隔离：训练集 [T-24, T-1]，测试集 [T, T+1mo]
```

### 4.2 关键约束

- `shuffle=False` — 严格时间序列分割
- 所有特征 `shift(1)` — 严格前视规避
- 分层抽样 — 每期按行业×市值五分位分层，保证样本分布一致

### 4.3 自定义目标函数: LambdaRank (Rank-L2)

```python
# LightGBM 原生 LambdaRank
objective = 'lambdarank'

# 或自定义 Rank-L2 损失
def rank_l2_loss(pred, dtrain):
    y = dtrain.get_label()
    ranked_pred = y.shape[0] * rank_normalize(pred)
    grad = 2 * (ranked_pred - y) / y.shape[0]
    hess = 2 * np.ones_like(y) / y.shape[0]
    return grad, hess
```

**原理：** 只关心股票截面排序（做多前10%，做空后10%），不关心绝对收益大小。

### 4.4 多任务扩展 (可选)

同时预测：(1) 未来20日收益 (2) 未来20日波动率。实现风险感知排序。

### 4.5 Optuna 贝叶斯超参优化

| 参数               | 搜索范围      | 说明      |
|-------------------|--------------|----------|
| num_leaves        | [15, 127]    | 树复杂度  |
| min_child_samples | [5, 100]     | 过拟合预防 |
| learning_rate     | [0.005, 0.1] | 步长      |
| reg_alpha         | [1e-3, 10]   | L1 正则   |
| reg_lambda        | [1e-3, 10]   | L2 正则   |
| bagging_fraction  | [0.5, 1.0]   | 行采样    |
| feature_fraction  | [0.5, 1.0]   | 列采样    |
| min_gain_to_split | [0.0, 1.0]   | 分裂阈值   |

**验证集：** 训练窗口最后3个月。**严禁用测试集调参。**

### 4.6 四层过拟合防御体系

```
第一层 - 模型: L1/L2 + 早停 (基于IC, patience=30) + 行列采样
第二层 - 数据: shuffle=False + shift(1) + walk-forward
第三层 - 验证: 蒙特卡洛扰动检验
第四层 - 后约束: 截面敞口限制
```

---

## 5. 组合构建与风险控制

### 5.1 二次规划组合优化 (CVXPY)

```
最大化:   μᵀw - λ·wᵀΣw - tc·|w - w₀|₁
约束条件:
  Σ|wᵢ - w₀ᵢ| ≤ turnover_limit      (换手率约束)
  |w'B - b'B| ≤ sector_dev          (行业中性)
  |wᵢ| ≤ max_weight                 (单股权重上限)
  0 ≤ w ≤ 1                          (仅做多)
  market_cap_exposure ≤ cap_dev     (市值中性)
```

- **λ**: 时变风险厌恶系数（滚动窗口估计）
- **Σ**: 协方差矩阵（样本协方差 + Ledoit-Wolf 压缩估计）
- **tc**: 交易成本 (双边 0.3% + 冲击成本 0.1%)

### 5.2 因子拥挤度监控

- 计算拥挤度得分：因子收益与资金流的相关性
- 自动降权或剔除拥挤因子
- 监控因子收益衰减曲线

---

## 6. 蒙特卡洛稳定性检验

### 6.1 方法论

```
对每个预测期：
  对每个噪声水平 ε ∈ {0.1%, 0.5%, 1%, 2%, 5%}:
    重复 200 次:
      扰动因子矩阵 = 原始因子 + N(0, ε · σ_f)
      预测 → 优化 → 记录组合收益
    → 每个噪声水平生成 200 条模拟净值曲线
```

### 6.2 输出报告

```
蒙特卡洛模拟 (ε=1%):
  Sharpe 比: [0.85, 1.23, 1.67]  (10%/50%/90% 分位)
  最大回撤:  [12%, 18%, 27%]     (10%/50%/90% 分位)
  崩塌概率 (回撤>20%): 8.5%
```

### 6.3 解读

- **ε ≤ 1%**: 策略稳健（绩效分布集中）
- **ε ≥ 5%**: 预期退化 — 用于识别策略脆弱性
- **对比**: 对朴素基线做同样测试，展示改进幅度

---

## 7. 回测严谨性

### 7.1 设计

- **执行:** T 日预测 → T+1 开盘成交
- **成本:** 双边滑点 0.3% + 冲击成本 0.1%
- **涨跌停:** 涨停不开仓，跌停不平仓
- **停牌:** 停牌期间持仓不变
- **基准:** CSI 300 / CSI 500 / CSI 800 (可配置)

### 7.2 绩效指标

| 指标      | 说明               |
|----------|-------------------|
| 年化收益率 | 几何平均           |
| 超额收益   | 相对基准           |
| 年化波动率 | 日收益标准差        |
| Sharpe 比 | 无风险利率 2.5%    |
| Calmar 比 | 年化收益 / 最大回撤 |
| 最大回撤   | 峰值到谷值         |
| 年化换手率 | 单边换手           |
| IC (均值) | 截面 IC           |
| IR       | IC / std(IC)      |
| RankIC   | Spearman 秩相关 IC |
| 胜率      | IC>0 的天数占比    |

### 7.3 分层收益

- **Top-bottom 五分位差:** 验证单调性
- **因子贡献分解:** 按类别（动量、基本面等）

---

## 8. 模型可解释性

### 8.1 SHAP 分析

- **全局:** SHAP 条形图 + 蜂群图（全周期）
- **时序:** 各因子 SHAP 重要性随时间变化曲线
- **情境:** 牛市/熊市/震荡市下的 SHAP 差异
- **单样本:** 单只股票预测的瀑布分解图

### 8.2 Brinson 归因

```
组合超额收益 =
  行业配置收益  +  个股选择收益  +  交互效应
  Σ(wᵢ - bᵢ) × R_bᵢ    Σbᵢ × (R_pᵢ - R_bᵢ)
```

### 8.3 因子重要性时序跟踪

- 每次滚动训练后记录特征 gain/split 重要性
- 生成 **重要性热力图**（纵轴因子 × 横轴时间）
- 识别：贡献衰减的因子 → 自动淘汰；特定市场环境下突变的因子 → 情境化建模

---

## 9. 配置 (config.yaml)

```yaml
data:
  market: csi300
  start_date: 2018-01-01
  end_date: 2026-06-30

factors:
  use_qlib: true
  custom_factors: true
  preprocessing:
    winsorize: 3sigma
    standardize: cross_sectional
    neutralize: [market_cap, industry, beta]
    orthogonalize: gram_schmidt

training:
  window_months: 24
  predict_days: 20
  rolling_step: monthly
  objective: lambdarank
  early_stopping_rounds: 30
  optuna_trials: 50

portfolio:
  top_n: 50
  turnover_limit: 0.30
  risk_aversion: 1.0
  max_weight: 0.05
  sector_neutral: true
  size_neutral: true

monte_carlo:
  noise_levels: [0.001, 0.005, 0.01, 0.02, 0.05]
  n_simulations: 200
  seed: 42

backtest:
  benchmark: csi300
  slippage: 0.003
  market_impact: 0.001
```

---

## 10. 执行流程

```
$ python scripts/run_pipeline.py --config config/config.yaml

Step 1:  从 Qlib 加载数据 (CSI 300 成分股 + 日频数据)
Step 2:  生成 55+ 因子 → 预处理管线
Step 3:  Walk-Forward 滚动训练 (24mo 窗口, 1mo 步长)
Step 4:  组合优化 (CVXPY 每期)
Step 5:  回测 (含交易成本 & 约束)
Step 6:  蒙特卡洛稳定性检验
Step 7:  SHAP 分析 + Brinson 归因
Step 8:  生成报告
```

---

## 核心技术差异化总结

| 维度      | 朴素做法        | 本项目设计                      |
|----------|----------------|-------------------------------|
| 因子预处理 | 原始因子直接使用 | 中性化 + 正交化 + IC时序特征      |
| 特征筛选  | 无              | 三阶段 (IC + L1 + 后筛选)       |
| 训练方式  | 单次划分         | Walk-Forward 滚动训练          |
| 目标函数  | MSE            | LambdaRank (Rank-L2)          |
| 超参调优  | 网格搜索         | Optuna (验证集调参)            |
| 过拟合防御 | 无             | 四层防御体系                    |
| 组合构建  | Top-N 等权      | CVXPY 二次规划                 |
| 稳定性检验 | 无             | 蒙特卡洛扰动检验                 |
| 可解释性  | 无              | SHAP + Brinson + 重要性时序跟踪 |
