import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config import load_config
from data.qlib_provider import QlibDataProvider
from factors.base import FactorPipeline
from factors.alpha_volume import register_all_volume_price_factors
from factors.fundamental import register_all_fundamental_factors
from factors.alternative import register_all_alternative_factors
from factors.pipeline import FactorPreprocessingPipeline
from features.selector import ic_prefilter
from features.transformer import build_momentum_features, build_ic_time_series
from features.label import compute_labels
from model.trainer import WalkForwardTrainer
from model.optimizer import OptunaTuner
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
    trade_dates = provider.get_trade_dates()

    print("[2/8] Computing factors...")
    pipeline = FactorPipeline()
    pipeline = register_all_volume_price_factors(pipeline)
    pipeline = register_all_fundamental_factors(pipeline)
    pipeline = register_all_alternative_factors(pipeline)
    factor_df = pipeline.compute_all(data)
    print(f"  Generated {len(factor_df.columns)} factors")

    print("[3/8] Preprocessing factors...")
    factor_df = FactorPreprocessingPipeline(config["factors"]["preprocessing"]).process(factor_df, data)
    factor_df = factor_df.groupby(level=0).shift(1)
    factor_df = factor_df.dropna(how="all")

    print("[4/8] Selecting features + building derivatives...")
    labels = compute_labels(data["close"], periods=20, skip=1)
    selected = ic_prefilter(factor_df, labels)
    factor_df = factor_df[selected]
    print(f"  Selected {len(selected)} features after IC filter")

    factor_df = build_momentum_features(factor_df)
    factor_df = build_ic_time_series(factor_df, labels)

    print("[5/8] Training models (Walk-Forward)...")
    dates = sorted(factor_df.index.get_level_values(0).unique())
    tuner = OptunaTuner(config["training"].get("optuna_trials", 50))
    trainer = WalkForwardTrainer(config, tuner=tuner)
    predictions = trainer.train(factor_df, labels, dates)

    print("[6/8] Backtesting...")
    prices = data["close"].unstack()
    benchmark_ret = data["close"].mean(axis=1).pct_change().dropna()
    if isinstance(benchmark_ret.index, pd.MultiIndex):
        benchmark_ret = benchmark_ret.groupby(level=0).last()
    engine = BacktestEngine(config)
    portfolio_returns, benchmark_returns, weights = engine.run(predictions, prices, benchmark_ret)

    labels_unstacked = labels.unstack() if isinstance(labels, pd.DataFrame) else labels
    report = MetricsReport(portfolio_returns, benchmark_returns, predictions, labels_unstacked, weights)
    metrics = report.generate()
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("[7/8] Model interpretability...")
    if trainer.models:
        shap_analyzer = SHAPAnalyzer()
        first_model = trainer.models[0]
        X_sample = factor_df.loc[factor_df.index.get_level_values(0) == trainer.dates[0]]
        if len(X_sample) > 0:
            shap_analyzer.analyze(first_model, X_sample.iloc[:min(100, len(X_sample))])
            top_features = shap_analyzer.get_top_features(10)
            print("  Top SHAP features:", [f[0] for f in top_features])

        imp_tracker = ImportanceTracker()
        for model, date in zip(trainer.models, trainer.dates):
            imp_tracker.record(model, date, list(factor_df.columns))
        imp_tracker.plot_heatmap(15)
        decaying = imp_tracker.get_decaying_features()
        if decaying:
            print(f"  Decaying features to watch: {decaying}")

    print("[8/8] Running Monte Carlo simulation...")
    mc = MonteCarloSimulator(config)
    mc_results = mc.run(factor_df, prices, benchmark_ret,
                         models=trainer.models, dates=trainer.dates)
    print(mc.report(mc_results))

    print("\nPipeline complete!")
    return metrics, mc_results, trainer


if __name__ == "__main__":
    run_pipeline()
