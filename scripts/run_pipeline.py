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
    if "close" in data:
        prices = data["close"].unstack()
    else:
        raise ValueError("Data must contain 'close' field")
    engine = BacktestEngine(config)
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
