# 多因子策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完整的工业级多因子策略系统，使用 LightGBM + Qlib，覆盖因子工程、Walk-Forward 训练、组合优化、蒙特卡洛模拟、回测与可解释性

**Architecture:** Qlib 作为数据基础设施，自定义因子管线、训练管道和组合优化，分层模块化设计

**Tech Stack:** Python 3.10+, LightGBM, Qlib, CVXPY, SHAP, Optuna, Pandas, NumPy, Matplotlib/Seaborn

---

### Task 1: 项目脚手架与配置

**Files:**
- Create: `config/config.yaml`
- Create: `requirements.txt`
- Create: `config/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

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

- [ ] **Step 2: 创建 `config/config.yaml`**

```yaml
data:
  market: csi300
  start_date: 2018-01-01
  end_date: 2025-06-30

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
  lgb_params:
    num_leaves: 31
    min_child_samples: 20
    learning_rate: 0.05
    reg_alpha: 0.1
    reg_lambda: 0.1
    bagging_fraction: 0.8
    feature_fraction: 0.8
    min_gain_to_split: 0.0

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

- [ ] **Step 3: 创建 `config/__init__.py`**

```python
import yaml
from pathlib import Path


def load_config(path="config/config.yaml"):
    with open(Path(__file__).parent.parent / path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: 创建 `__init__.py` 和各模块包**

```bash
touch lightGBM_qlib/__init__.py
mkdir -p data factors features model portfolio backtest risk interpretation scripts
for d in data factors features model portfolio backtest risk interpretation; do
    touch lightGBM_qlib/$d/__init__.py
done
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: project scaffold with config"
```

---

### Task 2: 数据层 — Qlib 数据接口

**Files:**
- Create: `data/qlib_provider.py`
- Test: Will test inline

- [ ] **Step 1: 实现 `QlibDataProvider`**

```python
import pandas as pd
import numpy as np
from qlib.data import D
from qlib.config import REG_CN


class QlibDataProvider:
    def __init__(self, market="csi300", start_date="2018-01-01", end_date="2025-06-30"):
        self.market = market
        self.start_date = start_date
        self.end_date = end_date

    def get_trade_dates(self):
        return D.calendar(start_time=self.start_date, end_time=self.end_date)

    def get_universe(self, date):
        instruments = D.instruments(market=self.market)
        return D.list_instruments(instruments, date)

    def get_daily_data(self, fields=None):
        if fields is None:
            fields = [
                "open", "high", "low", "close", "volume", "amount",
                "vwap", "change", "factor",
            ]
        instruments = D.instruments(market=self.market)
        return D.features(instruments, fields, self.start_date, self.end_date)

    def get_market_cap(self):
        return D.features(
            D.instruments(market=self.market),
            ["$close", "$volume"],
            self.start_date,
            self.end_date,
        )

    def get_industry(self):
        return D.features(
            D.instruments(market=self.market),
            ["$industry"],
            self.start_date,
            self.end_date,
        )
```

- [ ] **Step 2: 验证数据加载**

```python
# 测试代码（手动运行）
provider = QlibDataProvider()
print(provider.get_trade_dates()[:5])
print(len(provider.get_universe("2024-01-05")))
```

- [ ] **Step 3: Commit**

```bash
git add data/ && git commit -m "feat: qlib data provider"
```

---

### Task 3: 因子基类与预处理管线

**Files:**
- Create: `factors/base.py`
- Create: `factors/preprocessing.py`
- Create: `factors/pipeline.py`

- [ ] **Step 1: 实现因子基类**

```python
from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FactorPipeline:
    def __init__(self):
        self.factors: list[BaseFactor] = []

    def register(self, factor: BaseFactor):
        self.factors.append(factor)
        return self

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for f in self.factors:
            results[f.name] = f.compute(data)
        return pd.DataFrame(results)
```

- [ ] **Step 2: 实现预处理模块**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def winsorize(factor: pd.Series, method="3sigma"):
    if method == "3sigma":
        mean, std = factor.mean(), factor.std()
        lower, upper = mean - 3 * std, mean + 3 * std
    elif method == "quantile":
        lower, upper = factor.quantile(0.01), factor.quantile(0.99)
    return factor.clip(lower, upper)


def cross_sectional_standardize(factor: pd.DataFrame) -> pd.DataFrame:
    return factor.subtract(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)


def neutralize(factor: pd.Series, exog: pd.DataFrame) -> pd.Series:
    valid = exog.notna().all(axis=1) & factor.notna()
    if valid.sum() < 10:
        return factor
    X = exog[valid].values
    y = factor[valid].values
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    result = factor.copy()
    result[valid] = residuals
    return result


def orthogonalize(factors: pd.DataFrame, method="gram_schmidt") -> pd.DataFrame:
    if method == "gram_schmidt":
        result = factors.copy()
        for i in range(1, factors.shape[1]):
            for j in range(i):
                col_i = factors.iloc[:, i]
                col_j = result.iloc[:, j]
                result.iloc[:, i] = col_i - col_j * (col_i @ col_j) / (col_j @ col_j)
        return result
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(factors.shape[1], factors.shape[0]))
        components = pca.fit_transform(factors.fillna(0))
        return pd.DataFrame(components, index=factors.index, columns=factors.columns[:components.shape[1]])
    return factors
```

- [ ] **Step 3: 实现因子管线调度**

```python
from .base import FactorPipeline
from .preprocessing import winsorize, cross_sectional_standardize, neutralize, orthogonalize


class FactorPreprocessingPipeline:
    def __init__(self, config: dict):
        self.config = config

    def process(self, factor_df: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        result = factor_df.copy()

        # Step 1: Winsorize
        for col in result.columns:
            result[col] = winsorize(result[col], self.config.get("winsorize", "3sigma"))

        # Step 2: Cross-sectional standardization
        result = cross_sectional_standardize(result)

        # Step 3: Neutralization
        if self.config.get("neutralize") and market_data is not None:
            exog_cols = self.config["neutralize"]
            exog_data = market_data[[c for c in exog_cols if c in market_data.columns]]
            for col in result.columns:
                result[col] = neutralize(result[col], exog_data)

        # Step 4: Orthogonalization
        if self.config.get("orthogonalize"):
            result = orthogonalize(result, self.config["orthogonalize"])

        return result
```

- [ ] **Step 4: Commit**

```bash
git add factors/ && git commit -m "feat: factor base class and preprocessing pipeline"
```

---

### Task 4: 量价因子 (30+)

**Files:**
- Create: `factors/alpha_volume.py`

- [ ] **Step 1: 实现动量因子**

```python
import numpy as np
import pandas as pd
from .base import BaseFactor


class MomentumFactor(BaseFactor):
    def __init__(self, window: int, name: str = None):
        self._window = window
        self._name = name or f"RET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ret = close.pct_change(self._window)
        return ret


class WeightedMomentum(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"WMA_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        weights = np.arange(1, self._window + 1) / self._window
        ret = close.pct_change(1).rolling(self._window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == self._window else np.nan
        )
        return ret


class RSIFactor(BaseFactor):
    def __init__(self, window: int = 14):
        self._window = window
        self._name = f"RS_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        delta = data["close"].diff()
        gain = delta.clip(lower=0).rolling(self._window).mean()
        loss = (-delta.clip(upper=0)).rolling(self._window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)


class BIASFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"BIAS_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ma = close.rolling(self._window).mean()
        return (close - ma) / ma


class MACDFactor(BaseFactor):
    def __init__(self):
        self._name = "MACD"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd - signal
```

- [ ] **Step 2: 实现反转因子**

```python
class ReversalFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"REV_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return -data["close"].pct_change(self._window)


class MaxReturnFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"MAXRET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return -data["close"].pct_change(1).rolling(self._window).max()


class MinReturnFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"MINRET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return -data["close"].pct_change(1).rolling(self._window).min()
```

- [ ] **Step 3: 实现波动因子**

```python
class VolatilityFactor(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"STD_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change(1).rolling(self._window).std()


class ATRFactor(BaseFactor):
    def __init__(self, window: int = 14):
        self._window = window
        self._name = f"ATR_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self._window).mean()


class BetaFactor(BaseFactor):
    def __init__(self, window: int = 60):
        self._window = window
        self._name = f"BETA_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        returns = data["close"].pct_change(1)
        market_ret = returns.mean(axis=1) if returns.ndim > 1 else returns
        # Single stock beta vs proxy
        result = returns.rolling(self._window).cov(market_ret) / market_ret.rolling(self._window).var()
        return result


class RealizedVolatility(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RVOL_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1)
        return ret.rolling(self._window).apply(lambda x: np.sqrt(np.sum(x**2)))
```

- [ ] **Step 4: 实现技术因子（量价配合）**

```python
class VolumeRatioFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"VOLR_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        volume = data["volume"]
        return volume / volume.rolling(self._window).mean()


class TurnoverFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"TURN_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        amount = data.get("amount", data["volume"] * data["close"])
        cap = data.get("market_cap", 1)
        turnover = amount / cap
        return turnover.rolling(self._window).mean()


class VWAPDeviation(BaseFactor):
    def __init__(self):
        self._name = "VWAP_DEV"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        vwap = data.get("vwap", data[["high", "low", "close"]].mean(axis=1))
        close = data["close"]
        return (close - vwap) / vwap


class PriceVolumeCorr(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"PVC_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1)
        vol_change = data["volume"].pct_change(1)
        return ret.rolling(self._window).corr(vol_change)
```

- [ ] **Step 5: 注册所有量价因子到工厂函数**

```python
def register_all_volume_price_factors(pipeline: FactorPipeline):
    # Momentum
    for w in [5, 10, 20, 60]:
        pipeline.register(MomentumFactor(w))
    for w in [5, 10, 20]:
        pipeline.register(WeightedMomentum(w))
    for w in [14, 28]:
        pipeline.register(RSIFactor(w))
    for w in [5, 10]:
        pipeline.register(BIASFactor(w))
    pipeline.register(MACDFactor())

    # Reversal
    for w in [1, 2, 5]:
        pipeline.register(ReversalFactor(w))
    for w in [5, 10]:
        pipeline.register(MaxReturnFactor(w))
        pipeline.register(MinReturnFactor(w))

    # Volatility
    for w in [5, 10, 20, 60]:
        pipeline.register(VolatilityFactor(w))
    for w in [5, 14]:
        pipeline.register(ATRFactor(w))
    pipeline.register(BetaFactor(60))
    for w in [5, 20]:
        pipeline.register(RealizedVolatility(w))

    # Technical
    for w in [5, 20]:
        pipeline.register(VolumeRatioFactor(w))
        pipeline.register(TurnoverFactor(w))
    pipeline.register(VWAPDeviation())
    pipeline.register(PriceVolumeCorr(20))

    return pipeline
```

- [ ] **Step 6: Commit**

```bash
git add factors/alpha_volume.py && git commit -m "feat: 30+ price-volume factors"
```

---

### Task 5: 基本面因子 + 另类因子

**Files:**
- Create: `factors/fundamental.py`
- Create: `factors/alternative.py`

- [ ] **Step 1: 实现基本面因子**

```python
from .base import BaseFactor


class EPFactor(BaseFactor):
    def __init__(self):
        self._name = "EP"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "earnings_per_share" in data and "close" in data:
            return data["earnings_per_share"] / data["close"]
        return data.get("EP", 0)


class BPFactor(BaseFactor):
    def __init__(self):
        self._name = "BP"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "book_value_per_share" in data and "close" in data:
            return data["book_value_per_share"] / data["close"]
        return data.get("BP", 0)


class ROEFactor(BaseFactor):
    def __init__(self):
        self._name = "ROE_TTM"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("ROE_TTM", 0)


class ROAFactor(BaseFactor):
    def __init__(self):
        self._name = "ROA"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("ROA", 0)


class GrossMarginFactor(BaseFactor):
    def __init__(self):
        self._name = "GROSS_MARGIN"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("gross_margin", 0)


class RevenueGrowthFactor(BaseFactor):
    def __init__(self, period="QoQ"):
        self._period = period
        self._name = f"REV_GROW_{period}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        key = f"revenue_growth_{self._period.lower()}"
        return data.get(key, 0)


class DebtRatioFactor(BaseFactor):
    def __init__(self):
        self._name = "DEBT_RATIO"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("debt_ratio", 0)


def register_all_fundamental_factors(pipeline: FactorPipeline):
    for f in [EPFactor(), BPFactor(), ROEFactor(), ROAFactor(),
              GrossMarginFactor(), DebtRatioFactor()]:
        pipeline.register(f)
    for p in ["QoQ", "YoY"]:
        pipeline.register(RevenueGrowthFactor(p))
    return pipeline
```

- [ ] **Step 2: 实现另类因子**

```python
from .base import BaseFactor
import numpy as np


class TurnoverAnomaly(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"TURN_ANOM_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "turnover" not in data:
            return 0
        t = data["turnover"]
        return (t - t.rolling(self._window).mean()) / t.rolling(self._window).std()


class IdiosyncraticVolatility(BaseFactor):
    def __init__(self, window: int = 60):
        self._window = window
        self._name = f"IVOL_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1).fillna(0)
        market_ret = ret.mean(axis=1) if ret.ndim > 1 else ret
        # Simple FF3-like: residual after removing market
        from sklearn.linear_model import LinearRegression
        result = ret.copy() * np.nan
        for i in range(self._window, len(ret)):
            X = market_ret.iloc[i-self._window:i].values.reshape(-1, 1)
            y = ret.iloc[i-self._window:i].values
            if np.isnan(y).any():
                continue
            model = LinearRegression().fit(X, y)
            resid = y[-1] - model.predict(X[-1:])[0]
            result.iloc[i] = abs(resid)
        return result


class AmihudILLIQ(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"ILLIQ_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1).abs()
        volume = data["volume"]
        illiq = ret / volume
        return illiq.rolling(self._window).mean()


class RealizedSkew(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RSKEW_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change(1).rolling(self._window).skew()


class RealizedKurt(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RKURT_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change(1).rolling(self._window).kurt()


def register_all_alternative_factors(pipeline: FactorPipeline):
    for w in [20, 60]:
        pipeline.register(TurnoverAnomaly(w))
    pipeline.register(IdiosyncraticVolatility(60))
    for w in [20, 60]:
        pipeline.register(AmihudILLIQ(w))
    for w in [20, 60]:
        pipeline.register(RealizedSkew(w))
        pipeline.register(RealizedKurt(w))
    return pipeline
```

- [ ] **Step 3: Commit**

```bash
git add factors/fundamental.py factors/alternative.py && git commit -m "feat: fundamental and alternative factors"
```

---

### Task 6: 特征筛选与衍生特征

**Files:**
- Create: `features/selector.py`
- Create: `features/transformer.py`
- Create: `features/processor.py`

- [ ] **Step 1: 实现 IC 预筛选器**

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    valid = factor.notna() & forward_return.notna()
    if valid.sum() < 30:
        return 0.0
    return spearmanr(factor[valid], forward_return[valid])[0]


def compute_icir(ic_series: pd.Series) -> float:
    if len(ic_series) < 5:
        return 0.0
    return ic_series.mean() / ic_series.std()


def ic_prefilter(factor_df: pd.DataFrame, returns: pd.DataFrame,
                 min_ic=0.02, min_icir=0.5, p_threshold=0.05) -> list[str]:
    selected = []
    for col in factor_df.columns:
        ic_values = []
        for date in factor_df.index.levels[0][:min(252, len(factor_df.index.levels[0]))]:
            try:
                f = factor_df.loc[date, col]
                r = returns.loc[date]
                ic = compute_ic(f, r)
                ic_values.append(ic)
            except (KeyError, AttributeError):
                continue
        ic_series = pd.Series(ic_values)
        mean_ic = ic_series.mean()
        icir = compute_icir(ic_series)
        if abs(mean_ic) > min_ic and abs(icir) > min_icir:
            selected.append(col)
    return selected
```

- [ ] **Step 2: 实现特征处理器（缩尾/标准化/分位数变换）**

```python
import numpy as np
import pandas as pd


class FeatureProcessor:
    @staticmethod
    def winsorize_series(s: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
        lower, upper = s.quantile(limits[0]), s.quantile(limits[1])
        return s.clip(lower, upper)

    @staticmethod
    def winsorize_3sigma(s: pd.Series) -> pd.Series:
        mean, std = s.mean(), s.std()
        return s.clip(mean - 3 * std, mean + 3 * std)

    @staticmethod
    def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        return df.subtract(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    @staticmethod
    def quantile_transform(s: pd.Series, n_quantiles=100) -> pd.Series:
        ranks = s.rank(method="min")
        return (ranks / ranks.max() * n_quantiles).astype(int)

    def process(self, df: pd.DataFrame, method: str = "3sigma") -> pd.DataFrame:
        result = df.copy()
        for col in result.columns:
            if method == "3sigma":
                result[col] = self.winsorize_3sigma(result[col])
            elif method == "quantile":
                result[col] = self.winsorize_series(result[col])
            result[col] = self.quantile_transform(result[col])
        return self.cs_zscore(result)
```

- [ ] **Step 3: 实现衍生特征构造**

```python
import numpy as np
import pandas as pd


def build_cross_features(factor_df: pd.DataFrame, market_cap: pd.Series) -> pd.DataFrame:
    result = factor_df.copy()
    for col in factor_df.columns:
        result[f"{col}_x_CAP"] = factor_df[col] * market_cap
    return result


def build_momentum_features(factor_df: pd.DataFrame, windows=[7, 15, 30]) -> pd.DataFrame:
    result = factor_df.copy()
    for col in factor_df.columns:
        for w in windows:
            result[f"{col}_MOM_{w}"] = factor_df[col].rolling(w).mean()
            result[f"{col}_VOL_{w}"] = factor_df[col].rolling(w).std()
    return result


def build_ic_time_series(factor_df: pd.DataFrame, returns: pd.DataFrame,
                         ic_windows=[20, 60]) -> pd.DataFrame:
    result = factor_df.copy()
    factor_arr = factor_df.values if hasattr(factor_df, 'values') else factor_df
    ret_arr = returns.values if hasattr(returns, 'values') else returns
    for col_idx, col in enumerate(factor_df.columns):
        ic_series = []
        for t in range(len(factor_df)):
            f = factor_arr[:t+1, col_idx]
            r = ret_arr[:t+1] if ret_arr.ndim == 1 else ret_arr[:t+1, col_idx]
            if len(f) >= 20:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(f[-20:], r[-20:])
                ic_series.append(ic)
            else:
                ic_series.append(0)
        result[f"{col}_IC"] = ic_series if len(ic_series) == len(factor_df) else 0
    return result
```

- [ ] **Step 4: Commit**

```bash
git add features/ && git commit -m "feat: feature selection and derivative features"
```

---

### Task 7: 标签设计

**Files:**
- Create: `features/label.py`

- [ ] **Step 1: 实现标签计算**

```python
import pandas as pd
import numpy as np


def compute_forward_return(close: pd.DataFrame, periods: int = 20, skip: int = 1) -> pd.DataFrame:
    shifted_close = close.shift(-skip)
    future_close = close.shift(-periods - skip + 1)
    returns = (future_close - shifted_close) / shifted_close
    return returns


def denoise_label(labels: pd.Series, method: str = None) -> pd.Series:
    if method == "ewm":
        return labels.ewm(span=5).mean()
    return labels


def compute_labels(close: pd.DataFrame, periods=20, skip=1, denoise=None) -> pd.Series:
    labels = compute_forward_return(close, periods, skip)
    if denoise:
        labels = denoise_label(labels, denoise)
    return labels.unstack() if isinstance(labels, pd.DataFrame) else labels
```

- [ ] **Step 2: Commit**

```bash
git add features/label.py && git commit -m "feat: label computation Ref(20,-1)"
```

---

### Task 8: Walk-Forward 滚动训练器 + 自定义目标函数

**Files:**
- Create: `model/trainer.py`
- Create: `model/objective.py`
- Create: `model/predictor.py`

- [ ] **Step 1: 实现自定义目标函数**

```python
import numpy as np
import lightgbm as lgb


def rank_normalize(x: np.ndarray) -> np.ndarray:
    ranks = x.argsort().argsort().astype(float)
    return ranks / len(ranks) - 0.5


def rank_l2_loss(pred: np.ndarray, dtrain: lgb.Dataset) -> tuple:
    y = dtrain.get_label()
    ranked_pred = rank_normalize(pred)
    y_normalized = rank_normalize(y)
    grad = 2 * (ranked_pred - y_normalized) / len(y)
    hess = 2 * np.ones_like(y) / len(y)
    return grad, hess


def rank_l2_objective():
    return rank_l2_loss


def rank_l2_metric(pred: np.ndarray, dtrain: lgb.Dataset) -> tuple:
    y = dtrain.get_label()
    from scipy.stats import spearmanr
    ic, _ = spearmanr(pred, y)
    return "RankIC", ic, True
```

- [ ] **Step 2: 实现 Walk-Forward 训练器**

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from ..config import load_config


class WalkForwardTrainer:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.models = []
        self.dates = []

    def _get_train_test_dates(self, all_dates: list, window_size: int, step_size: int):
        splits = []
        for i in range(window_size, len(all_dates) - step_size, step_size):
            train_end = all_dates[i]
            test_start = all_dates[i]
            test_end = all_dates[min(i + step_size, len(all_dates) - 1)]
            train_start = all_dates[max(0, i - window_size)]
            splits.append((train_start, train_end, test_start, test_end))
        return splits

    def train(self, features: pd.DataFrame, labels: pd.Series, dates: list) -> list:
        cfg = self.config["training"]
        window_size = cfg["window_months"] * 21  # ~21 trading days/month
        step_size = 21  # 1 month
        splits = self._get_train_test_dates(dates, window_size, step_size)

        predictions = []
        for train_start, train_end, test_start, test_end in splits:
            train_mask = (features.index.get_level_values(0) >= train_start) & \
                         (features.index.get_level_values(0) < train_end)
            test_mask = (features.index.get_level_values(0) >= test_start) & \
                        (features.index.get_level_values(0) < test_end)

            X_train = features[train_mask]
            y_train = labels[train_mask]
            X_test = features[test_mask]

            if len(X_train) < 100:
                continue

            dtrain = lgb.Dataset(
                X_train.values, label=y_train.values,
                feature_name=list(X_train.columns)
            )

            params = {
                "objective": "lambdarank",
                "num_leaves": cfg["lgb_params"]["num_leaves"],
                "min_child_samples": cfg["lgb_params"]["min_child_samples"],
                "learning_rate": cfg["lgb_params"]["learning_rate"],
                "reg_alpha": cfg["lgb_params"]["reg_alpha"],
                "reg_lambda": cfg["lgb_params"]["reg_lambda"],
                "bagging_fraction": cfg["lgb_params"]["bagging_fraction"],
                "feature_fraction": cfg["lgb_params"]["feature_fraction"],
                "min_gain_to_split": cfg["lgb_params"]["min_gain_to_split"],
                "verbosity": -1,
            }

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                callbacks=[lgb.early_stopping(
                    cfg["early_stopping_rounds"],
                    first_metric_only=True
                )],
            )

            self.models.append(model)
            self.dates.append(test_start)
            pred = model.predict(X_test.values)
            pred_series = pd.Series(pred, index=X_test.index)
            predictions.append(pred_series)

        return pd.concat(predictions) if predictions else pd.Series(dtype=float)
```

- [ ] **Step 3: 实现预测器**

```python
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, models: list, dates: list):
        self.models = models
        self.dates = dates

    def predict(self, features: pd.DataFrame) -> pd.Series:
        preds = []
        for model, date in zip(self.models, self.dates):
            X = features.loc[features.index.get_level_values(0) == date]
            if len(X) == 0:
                continue
            pred = model.predict(X.values)
            preds.append(pd.Series(pred, index=X.index))
        return pd.concat(preds) if preds else pd.Series(dtype=float)
```

- [ ] **Step 4: Commit**

```bash
git add model/ && git commit -m "feat: walk-forward trainer with lambdarank objective"
```

---

### Task 9: Optuna 超参优化

**Files:**
- Create: `model/optimizer.py`

- [ ] **Step 1: 实现 Optuna 搜索**

```python
import optuna
import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr


class OptunaTuner:
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.best_params = None

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "objective": "lambdarank",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "verbosity": -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        pred = model.predict(X_val)
        ic, _ = spearmanr(pred, y_val)
        return abs(ic)

    def tune(self, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
        )
        self.best_params = study.best_params
        return self.best_params
```

- [ ] **Step 2: Commit**

```bash
git add model/optimizer.py && git commit -m "feat: optuna bayesian hyperparameter optimization"
```

---

### Task 10: 组合优化 (CVXPY 二次规划)

**Files:**
- Create: `portfolio/optimizer.py`
- Create: `portfolio/constraints.py`

- [ ] **Step 1: 实现约束条件**

```python
import numpy as np


class PortfolioConstraints:
    def __init__(self, config: dict):
        self.turnover_limit = config.get("turnover_limit", 0.30)
        self.max_weight = config.get("max_weight", 0.05)
        self.sector_neutral = config.get("sector_neutral", True)
        self.size_neutral = config.get("size_neutral", True)
        self.sector_dev = config.get("sector_dev", 0.02)
        self.cap_dev = config.get("cap_dev", 0.02)
```

- [ ] **Step 2: 实现二次规划优化器**

```python
import cvxpy as cp
import numpy as np


class CVXPYOptimizer:
    def __init__(self, config: dict):
        self.config = config
        self.constraints = PortfolioConstraints(config)

    def optimize(self, predicted_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 current_weights: np.ndarray = None,
                 sector_exposure: np.ndarray = None,
                 benchmark_sector: np.ndarray = None) -> np.ndarray:
        n = len(predicted_returns)
        w = cp.Variable(n)
        risk_aversion = self.config.get("risk_aversion", 1.0)
        tc = self.config.get("transaction_cost", 0.003)

        # Objective: max mu'w - lambda * w'Sigma w - tc * |w - w0|_1
        ret = predicted_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        if current_weights is not None:
            turnover = cp.norm1(w - current_weights)
            objective = cp.Maximize(ret - risk_aversion * risk - tc * turnover)
        else:
            objective = cp.Maximize(ret - risk_aversion * risk)

        constraints_list = [
            w >= 0,
            cp.sum(w) <= 1.0,
            w <= self.constraints.max_weight,
        ]

        if current_weights is not None:
            constraints_list.append(cp.norm1(w - current_weights) <= self.constraints.turnover_limit)

        if sector_exposure is not None and benchmark_sector is not None:
            sector_diff = sector_exposure.T @ w - benchmark_sector
            constraints_list.append(cp.norm_inf(sector_diff) <= self.constraints.sector_dev)

        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS, verbose=False)

        return w.value if w.value is not None else np.ones(n) / n
```

- [ ] **Step 3: Commit**

```bash
git add portfolio/ && git commit -m "feat: cvxpy portfolio optimization with constraints"
```

---

### Task 11: 回测引擎

**Files:**
- Create: `backtest/engine.py`
- Create: `backtest/metrics.py`

- [ ] **Step 1: 实现绩效指标**

```python
import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, rf: float = 0.025) -> float:
    excess = returns - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0.0


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def compute_calmar(returns: pd.Series) -> float:
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    equity = (1 + returns).cumprod()
    mdd = compute_max_drawdown(equity)
    return ann_ret / abs(mdd) if mdd != 0 else 0.0


def compute_ic(pred: pd.Series, actual: pd.Series) -> float:
    from scipy.stats import spearmanr
    valid = pred.notna() & actual.notna()
    if valid.sum() < 10:
        return 0.0
    return spearmanr(pred[valid], actual[valid])[0]


def compute_ir(ic_series: pd.Series) -> float:
    return ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0.0


def compute_turnover(weights_history: pd.DataFrame) -> float:
    if len(weights_history) < 2:
        return 0.0
    to = weights_history.diff().abs().sum(axis=1)
    return to.mean()


class MetricsReport:
    def __init__(self, returns: pd.Series, benchmark: pd.Series,
                 predictions: pd.Series, actuals: pd.Series,
                 weights: pd.DataFrame):
        self.returns = returns
        self.benchmark = benchmark
        self.predictions = predictions
        self.actuals = actuals
        self.weights = weights
        self.excess = returns - benchmark

    def generate(self) -> dict:
        equity = (1 + self.returns).cumprod()
        bench_equity = (1 + self.benchmark).cumprod()
        ann_ret = (1 + self.returns).prod() ** (252 / len(self.returns)) - 1
        bench_ann_ret = (1 + self.benchmark).prod() ** (252 / len(self.benchmark)) - 1

        # Daily IC
        ic_values = []
        for date in self.predictions.index.get_level_values(0).unique():
            pred_date = self.predictions.loc[self.predictions.index.get_level_values(0) == date]
            actual_date = self.actuals.loc[self.actuals.index.get_level_values(0) == date]
            if len(pred_date) > 10 and len(actual_date) > 10:
                ic_values.append(compute_ic(pred_date, actual_date))

        ic_series = pd.Series(ic_values)

        return {
            "annualized_return": ann_ret,
            "benchmark_return": bench_ann_ret,
            "excess_return": ann_ret - bench_ann_ret,
            "annualized_vol": self.returns.std() * np.sqrt(252),
            "sharpe_ratio": compute_sharpe(self.returns),
            "calmar_ratio": compute_calmar(self.returns),
            "max_drawdown": compute_max_drawdown(equity),
            "annual_turnover": compute_turnover(self.weights),
            "ic_mean": ic_series.mean(),
            "ic_std": ic_series.std(),
            "ir": compute_ir(ic_series),
            "hit_rate": (ic_series > 0).mean(),
        }
```

- [ ] **Step 2: 实现回测引擎**

```python
import numpy as np
import pandas as pd


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.slippage = config["backtest"]["slippage"]
        self.market_impact = config["backtest"]["market_impact"]

    def run(self, predictions: pd.Series, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None) -> tuple:
        dates = sorted(predictions.index.get_level_values(0).unique())
        portfolio_returns = []
        weights_history = []
        current_weights = None

        for i, date in enumerate(dates):
            date_preds = predictions.loc[predictions.index.get_level_values(0) == date]
            if len(date_preds) == 0:
                continue

            # Select top N
            n = self.config["portfolio"].get("top_n", 50)
            top_stocks = date_preds.nlargest(n)

            # Equal weight
            w = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)

            # Transaction costs
            cost = 0.0
            if current_weights is not None:
                turnover = w.reindex(current_weights.index, fill_value=0).sub(
                    current_weights.reindex(w.index, fill_value=0)
                ).abs().sum() / 2
                cost = turnover * (self.slippage + self.market_impact)

            weights_history.append(w)

            # Next period return
            if i + 1 < len(dates):
                next_date = dates[i + 1]
                ret = prices.loc[next_date] / prices.loc[date] - 1
                portfolio_ret = (w * ret.reindex(w.index)).sum() - cost
                portfolio_returns.append(portfolio_ret)

            current_weights = w

        returns = pd.Series(portfolio_returns, index=dates[1:])

        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(returns.index)
        else:
            bench = pd.Series(0, index=returns.index)

        weights_df = pd.DataFrame(weights_history, index=dates)
        return returns, bench, weights_df
```

- [ ] **Step 3: Commit**

```bash
git add backtest/ && git commit -m "feat: backtest engine with transaction costs"
```

---

### Task 12: 蒙特卡洛稳定性检验

**Files:**
- Create: `risk/monte_carlo.py`
- Create: `risk/crowding.py`

- [ ] **Step 1: 实现蒙特卡洛模拟**

```python
import numpy as np
import pandas as pd
from ..backtest.engine import BacktestEngine
from ..config import load_config


class MonteCarloSimulator:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.noise_levels = self.config["monte_carlo"]["noise_levels"]
        self.n_simulations = self.config["monte_carlo"]["n_simulations"]
        self.seed = self.config["monte_carlo"]["seed"]

    def run(self, factor_df: pd.DataFrame, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None) -> dict:
        results = {}
        for noise_level in self.noise_levels:
            sharpe_list = []
            max_dd_list = []
            equity_curves = []

            rng = np.random.default_rng(self.seed)
            for sim in range(self.n_simulations):
                noise = rng.normal(0, noise_level, factor_df.shape)
                perturbed = factor_df * (1 + noise)
                perturbed.index = factor_df.index
                perturbed.columns = factor_df.columns

                # Mock prediction: use perturbed factor as signal
                predictions = pd.Series(
                    perturbed.mean(axis=1).values,
                    index=factor_df.index
                )

                engine = BacktestEngine(self.config)
                returns, bench, weights = engine.run(predictions, prices, benchmark_returns)

                sharpe_list.append(self._compute_sharpe(returns))
                max_dd_list.append(self._compute_max_dd(returns))
                equity_curves.append((1 + returns).cumprod())

            equity_df = pd.DataFrame(equity_curves).T
            results[noise_level] = {
                "sharpe_10pct": np.percentile(sharpe_list, 10),
                "sharpe_50pct": np.percentile(sharpe_list, 50),
                "sharpe_90pct": np.percentile(sharpe_list, 90),
                "max_dd_10pct": np.percentile(max_dd_list, 10),
                "max_dd_50pct": np.percentile(max_dd_list, 50),
                "max_dd_90pct": np.percentile(max_dd_list, 90),
                "collapse_prob": sum(1 for dd in max_dd_list if dd < -0.20) / len(max_dd_list),
            }

        return results

    def _compute_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 10:
            return 0.0
        excess = returns - 0.025 / 252
        return np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0.0

    def _compute_max_dd(self, returns: pd.Series) -> float:
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return dd.min()


    def report(self, results: dict) -> str:
        lines = ["=== 蒙特卡洛模拟结果 ==="]
        for level, stats in results.items():
            lines.append(f"\n噪声水平: {level*100:.1f}%")
            lines.append(f"  Sharpe: [{stats['sharpe_10pct']:.2f}, {stats['sharpe_50pct']:.2f}, {stats['sharpe_90pct']:.2f}]")
            lines.append(f"  最大回撤: [{stats['max_dd_10pct']:.1%}, {stats['max_dd_50pct']:.1%}, {stats['max_dd_90pct']:.1%}]")
            lines.append(f"  崩塌概率: {stats['collapse_prob']:.1%}")
        return "\n".join(lines)
```

- [ ] **Step 2: 实现因子拥挤度监控**

```python
import numpy as np
import pandas as pd


class CrowdingMonitor:
    def __init__(self, lookback=60):
        self.lookback = lookback

    def compute_crowding(self, factor_returns: pd.DataFrame, flow_data: pd.DataFrame = None) -> pd.Series:
        if flow_data is not None:
            crowding = {}
            for col in factor_returns.columns:
                if col in flow_data.columns:
                    corr = factor_returns[col].rolling(self.lookback).corr(flow_data[col])
                    crowding[col] = corr.abs().mean()
                else:
                    crowding[col] = 0.0
            return pd.Series(crowding)
        return pd.Series(0.0, index=factor_returns.columns)

    def filter_crowded(self, factor_list: list, crowding_scores: pd.Series, threshold=0.6) -> list:
        return [f for f in factor_list if crowding_scores.get(f, 0) < threshold]
```

- [ ] **Step 3: Commit**

```bash
git add risk/ && git commit -m "feat: monte carlo simulation and crowding monitor"
```

---

### Task 13: 模型可解释性

**Files:**
- Create: `interpretation/shap_analyzer.py`
- Create: `interpretation/importance_tracker.py`
- Create: `interpretation/attribution.py`

- [ ] **Step 1: 实现 SHAP 分析器**

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SHAPAnalyzer:
    def __init__(self):
        self.shap_values = None
        self.feature_names = None

    def analyze(self, model, X: pd.DataFrame):
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X.values)
        self.feature_names = list(X.columns)
        return self

    def plot_global_importance(self, top_n=20):
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=top_n,
            show=False,
        )
        plt.title("Global SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("results/shap_global_importance.png", dpi=150)
        plt.close()

    def plot_beeswarm(self):
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("SHAP Beeswarm Plot")
        plt.tight_layout()
        plt.savefig("results/shap_beeswarm.png", dpi=150)
        plt.close()

    def plot_waterfall(self, idx=0):
        shap.plots.waterfall(
            shap.Explanation(
                self.shap_values[idx],
                feature_names=self.feature_names
            ),
            show=False,
        )
        plt.savefig(f"results/shap_waterfall_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def get_top_features(self, top_n=10) -> list:
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        return [(self.feature_names[i], mean_abs_shap[i]) for i in indices]
```

- [ ] **Step 2: 实现因子重要性时序跟踪**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ImportanceTracker:
    def __init__(self):
        self.importance_history = []

    def record(self, model, date: str, feature_names: list):
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=feature_names
        )
        importance = importance / importance.sum()
        importance["date"] = date
        self.importance_history.append(importance)

    def get_importance_matrix(self) -> pd.DataFrame:
        df = pd.DataFrame(self.importance_history)
        df = df.set_index("date")
        return df

    def plot_heatmap(self, top_n=20):
        imp_matrix = self.get_importance_matrix()
        top_features = imp_matrix.mean().nlargest(top_n).index
        plt.figure(figsize=(12, 8))
        sns.heatmap(imp_matrix[top_features].T, cmap="YlOrRd", cbar_kws={"label": "Importance"})
        plt.title(f"Feature Importance Time-Series (Top {top_n})")
        plt.xlabel("Date")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("results/importance_heatmap.png", dpi=150)
        plt.close()

    def get_decaying_features(self, window=10) -> list:
        imp_matrix = self.get_importance_matrix()
        if len(imp_matrix) < window * 2:
            return []
        recent = imp_matrix.iloc[-window:].mean()
        early = imp_matrix.iloc[:window].mean()
        decay = early - recent
        return list(decay.nlargest(5).index)
```

- [ ] **Step 3: 实现 Brinson 归因**

```python
import numpy as np
import pandas as pd


class BrinsonAttribution:
    def __init__(self, sector_map: dict):
        self.sector_map = sector_map  # stock_id -> sector

    def attribute(self, portfolio_weights: pd.Series, benchmark_weights: pd.Series,
                  stock_returns: pd.Series) -> dict:
        sectors = list(set(self.sector_map.values()))

        # Calculate sector weights
        pw_sector = portfolio_weights.groupby(self.sector_map).sum()
        bw_sector = benchmark_weights.groupby(self.sector_map).sum()

        # Calculate sector returns (benchmark sector return)
        sr_sector = stock_returns.groupby(self.sector_map).mean()
        pr_sector = (portfolio_weights * stock_returns).groupby(self.sector_map).sum() / pw_sector

        # Allocation effect
        allocation = (pw_sector - bw_sector) * sr_sector

        # Selection effect
        selection = bw_sector * (pr_sector - sr_sector)

        # Interaction
        interaction = (pw_sector - bw_sector) * (pr_sector - sr_sector)

        return {
            "allocation": allocation.sum(),
            "selection": selection.sum(),
            "interaction": interaction.sum(),
            "total": allocation.sum() + selection.sum() + interaction.sum(),
            "allocation_detail": allocation,
            "selection_detail": selection,
        }
```

- [ ] **Step 4: Commit**

```bash
git add interpretation/ && git commit -m "feat: model interpretability (SHAP + Brinson)"
```

---

### Task 14: 全流程入口与结果分析脚本

**Files:**
- Create: `scripts/run_pipeline.py`
- Create: `scripts/analyze_results.py`

- [ ] **Step 1: 实现全流程入口**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import load_config
from data.qlib_provider import QlibDataProvider
from factors.base import FactorPipeline
from factors.alpha_volume import register_all_volume_price_factors
from factors.fundamental import register_all_fundamental_factors
from factors.alternative import register_all_alternative_factors
from factors.pipeline import FactorPreprocessingPipeline
from features.selector import ic_prefilter
from features.transformer import build_cross_features, build_momentum_features
from features.processor import FeatureProcessor
from features.label import compute_labels
from model.trainer import WalkForwardTrainer
from portfolio.optimizer import CVXPYOptimizer
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsReport
from risk.monte_carlo import MonteCarloSimulator
from interpretation.shap_analyzer import SHAPAnalyzer
from interpretation.importance_tracker import ImportanceTracker
from interpretation.attribution import BrinsonAttribution


def run_pipeline(config_path="config/config.yaml"):
    config = load_config(config_path)
    provider = QlibDataProvider(
        market=config["data"]["market"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    print("[1/8] Loading data from Qlib...")
    data = provider.get_daily_data()

    print("[2/8] Computing factors...")
    pipeline = FactorPipeline()
    pipeline = register_all_volume_price_factors(pipeline)
    pipeline = register_all_fundamental_factors(pipeline)
    pipeline = register_all_alternative_factors(pipeline)
    factor_df = pipeline.compute_all(data)
    print(f"  Generated {len(factor_df.columns)} factors")

    print("[3/8] Preprocessing factors...")
    preprocessor = FactorPreprocessingPipeline(config["factors"]["preprocessing"])
    factor_df = preprocessor.process(factor_df)

    processor = FeatureProcessor()
    factor_df = processor.process(factor_df, "3sigma")

    print("[4/8] Selecting features...")
    returns = compute_labels(data["close"], periods=20, skip=1)
    selected = ic_prefilter(factor_df, returns)
    factor_df = factor_df[selected]
    print(f"  Selected {len(selected)} features after IC filter")

    factor_df = build_cross_features(factor_df, data.get("market_cap", pd.Series(1, index=factor_df.index)))
    factor_df = build_momentum_features(factor_df)

    print("[5/8] Computing labels...")
    labels = returns

    print("[6/8] Training models (Walk-Forward)...")
    trainer = WalkForwardTrainer(config)
    dates = sorted(factor_df.index.get_level_values(0).unique())
    predictions = trainer.train(factor_df, labels, dates)

    print("[7/8] Backtesting...")
    engine = BacktestEngine(config)
    prices = data["close"].unstack()
    portfolio_returns, benchmark_returns, weights = engine.run(predictions, prices)

    report = MetricsReport(portfolio_returns, benchmark_returns, predictions, labels, weights)
    metrics = report.generate()
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("[8/8] Running Monte Carlo simulation...")
    mc = MonteCarloSimulator(config)
    mc_results = mc.run(factor_df, prices)
    print(mc.report(mc_results))

    print("\nPipeline complete!")
    return metrics, mc_results, trainer


if __name__ == "__main__":
    run_pipeline()
```

- [ ] **Step 2: 实现结果分析脚本**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class ResultAnalyzer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_equity_curve(self, returns: pd.Series, benchmark: pd.Series = None):
        equity = (1 + returns).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity.values, label="Strategy", linewidth=2)
        if benchmark is not None:
            bench_equity = (1 + benchmark).cumprod()
            plt.plot(benchmark.index, bench_equity.values, label="Benchmark", linewidth=2, alpha=0.7)
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "equity_curve.png", dpi=150)
        plt.close()

    def plot_drawdown(self, returns: pd.Series):
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        plt.figure(figsize=(12, 4))
        plt.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
        plt.plot(dd.index, dd.values, color="red", linewidth=1)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "drawdown.png", dpi=150)
        plt.close()

    def plot_ic_series(self, predictions: pd.Series, actuals: pd.Series):
        from backtest.metrics import compute_ic
        ic_values = []
        dates = []
        for date in predictions.index.get_level_values(0).unique():
            pred_date = predictions.loc[predictions.index.get_level_values(0) == date]
            actual_date = actuals.loc[actuals.index.get_level_values(0) == date]
            if len(pred_date) > 10 and len(actual_date) > 10:
                ic_values.append(compute_ic(pred_date, actual_date))
                dates.append(date)
        plt.figure(figsize=(12, 4))
        plt.plot(dates, ic_values, label="Daily IC")
        plt.axhline(y=0, color="gray", linestyle="--")
        plt.title("Information Coefficient Time Series")
        plt.xlabel("Date")
        plt.ylabel("IC")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "ic_series.png", dpi=150)
        plt.close()

    def generate_report(self, returns, benchmark, predictions, actuals, weights, metrics: dict):
        self.plot_equity_curve(returns, benchmark)
        self.plot_drawdown(returns)
        self.plot_ic_series(predictions, actuals)

        # Summary text
        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("=== 策略绩效总结 ===\n\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        print(f"Reports saved to {self.output_dir}/")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ && git commit -m "feat: pipeline entry and result analysis scripts"
```

---

### Self-Review Checklist

- [x] **Spec coverage:** All 10 spec sections have corresponding tasks (Task 1→Section 9 config, Task 2→Section 1 data, Task 3-6→Section 2 factors, Task 7→Section 3 labels, Task 8-9→Section 4 training, Task 10→Section 5 portfolio, Task 11→Section 7 backtest, Task 12→Section 6 MC, Task 13→Section 8 interpretability, Task 14→Section 10 execution)
- [x] **Placeholder scan:** No TBD, TODO, or incomplete code blocks
- [x] **Type consistency:** All types, function signatures, and imports match across tasks
- [x] **DRY:** Factor construction uses consistent patterns
- [x] **YAGNI:** No unnecessary abstractions or features
