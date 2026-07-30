"""
主流程运行脚本

整合整个多因子策略 pipeline，包括：
1. 数据加载（Qlib + Akshare 基本面）
2. 因子计算（量价、基本面、另类因子）
3. 因子预处理（缩尾、标准化、中性化、正交化）
4. 特征选择（IC 预筛选）
5. Walk-Forward 模型训练（LightGBM + Optuna）
6. 组合优化（CVXPY）
7. 回测与绩效评估
8. 模型可解释性分析（SHAP、重要性跟踪）
9. 蒙特卡洛稳健性检验

执行方式：
    python scripts/run_pipeline.py --config config/config.yaml

默认使用 config/config.yaml 配置文件。
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import qlib
from config import load_config
from data.qlib_provider import QlibDataProvider
from factors.base import FactorPipeline
from factors.alpha_volume import register_all_volume_price_factors
from factors.fundamental import register_all_fundamental_factors
from factors.alternative import register_all_alternative_factors
from factors.pipeline import FactorPreprocessingPipeline
from features.selector import ic_prefilter
from features.transformer import build_momentum_features, build_ic_time_series, build_cross_sectional_rank_features, build_reversal_features
from features.label import compute_labels
from model.trainer import WalkForwardTrainer
import akshare as ak
from model.optimizer import OptunaTuner
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsReport
from risk.monte_carlo import MonteCarloSimulator
from risk.crowding import CrowdingMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

try:
    from interpretation.shap_analyzer import SHAPAnalyzer
    _has_shap = True
except ImportError:
    _has_shap = False
    logger.warning("[主流程] SHAP不可用，跳过SHAP分析")

from interpretation.importance_tracker import ImportanceTracker
from interpretation.attribution import BrinsonAttribution
from scripts.analyze_results import ResultAnalyzer


def run_pipeline(config_path="config/config.yaml"):
    """
    执行完整的多因子策略 pipeline

    步骤详情：
    1. 加载配置
    2. 初始化 Qlib
    3. 加载数据（行情 + 基本面）
    4. 计算因子
    5. 预处理因子
    6. 特征选择
    7. Walk-Forward 训练
    8. 回测
    9. 模型可解释性
    10. 蒙特卡洛检验

    Args:
        config_path: 配置文件路径

    Returns:
        (metrics, mc_results, trainer) - 绩效指标、蒙特卡洛结果、训练器
    """
    logger.info("=" * 60)
    logger.info("[主流程] Pipeline开始...")
    logger.info("=" * 60)

    # Step 1: 加载配置
    config = load_config(config_path)
    logger.info(f"[主流程] 配置已加载: 市场={config['data']['market']}, 开始={config['data']['start_date']}, 结束={config['data']['end_date']}")

    # Step 2: 初始化 Qlib
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "cn_data"))
    logger.info("[主流程] Qlib初始化完成")

    # 创建数据提供者
    provider = QlibDataProvider(
        market=config["data"]["market"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Step 3: 加载数据
    logger.info("[1/8] 从Qlib+Akshare加载数据...")
    data = provider.get_augmented_data()
    trade_dates = provider.get_trade_dates()
    logger.info(f"[1/8] 数据加载完成: {len(data)} 行, {len(data.columns)} 列")

    # 加载行业分类并加入 data
    industry_path = Path(config["fundamental"]["cache_dir"]) / "industry.csv"
    if industry_path.exists():
        ind_df = pd.read_csv(industry_path)
        code_to_industry = ind_df.set_index("证券代码")["所属申万一级行业名称(2021)"].to_dict()
        idx = data.index
        if idx.nlevels == 2:
            inst_level = 0 if idx.names[0] in ("instrument", "Instrument", "code", "asset") else 1
            data["industry"] = idx.get_level_values(inst_level).map(code_to_industry)
            covered = data["industry"].notna().sum()
            logger.info(f"[1/8] 行业分类已加载: {len(code_to_industry)} 只基准股票, 覆盖 {covered}/{len(data)} 行")
        else:
            logger.warning("[1/8] 无法识别数据索引结构, 跳过行业分类")
    else:
        logger.warning(f"[1/8] 行业分类文件不存在: {industry_path}")

    # Step 4: 计算因子
    logger.info("[2/8] 计算因子...")
    pipeline = FactorPipeline()
    pipeline = register_all_volume_price_factors(pipeline)  # 注册量价因子
    pipeline = register_all_fundamental_factors(pipeline)   # 注册基本面因子
    pipeline = register_all_alternative_factors(pipeline)   # 注册另类因子
    factor_df = pipeline.compute_all(data)
    logger.info(f"[2/8] 因子计算完成: 生成 {len(factor_df.columns)} 个因子")

    # Step 5: 预处理因子
    logger.info("[3/8] 因子预处理...")
    if "market_cap" in data.columns:
        idx = data.index
        inst_level = 1 if idx.nlevels == 2 and (
            idx.names[0] in ("instrument", "Instrument", "code", "asset") or
            (idx.names[0] is None and np.issubdtype(idx.get_level_values(1).dtype, np.object_))
        ) else 0
        before = data["market_cap"].isna().sum()
        data["market_cap"] = data.groupby(level=inst_level)["market_cap"].ffill()
        after = data["market_cap"].isna().sum()
        logger.info(f"[3/8] 市值向前填充 (inst_level={inst_level}): 空值 {before} -> {after}")
    factor_df = FactorPreprocessingPipeline(config["factors"]["preprocessing"]).process(factor_df, data)
    # shift(1) 避免前视偏差：T日因子预测 T+1 日收益
    factor_df = factor_df.groupby(level=0).shift(1)
    factor_df = factor_df.dropna(how="all")
    logger.info(f"[3/8] 预处理完成: 剩余 {len(factor_df.columns)} 个因子")

    # Step 6: 特征选择 + 标签计算
    logger.info("[4/8] 特征选择+构建衍生特征...")
    close_wide = data["close"].unstack(level=0)
    label_periods = config["training"].get("predict_days", 20)
    labels_wide = compute_labels(close_wide, periods=label_periods, skip=1)  # T+1到T+label_periods累计收益率作为标签
    labels = labels_wide.stack().swaplevel().sort_index()
    factor_df = factor_df.dropna(how="all")
    factor_df = factor_df.sort_index()
    labels = labels.reindex(factor_df.index)

    # 内存优化：在 IC 预筛选前将股票池减少到约 800 只，减少数据量约 30%
    inst_level = 0
    n_instruments = len(factor_df.index.get_level_values(inst_level).unique())
    if n_instruments > 800:
        non_nan_per_instrument = factor_df.notna().sum(axis=1).groupby(level=inst_level).sum()
        top_instruments = non_nan_per_instrument.nlargest(800).index
        keep_mask = factor_df.index.get_level_values(inst_level).isin(top_instruments)
        factor_df = factor_df.loc[keep_mask]
        labels = labels.reindex(factor_df.index)
        logger.info(f"[4/8] 股票池缩减: {n_instruments} -> {len(top_instruments)} 只, 行数: {len(factor_df)}")

    # IC 预筛选（过滤低预测力因子 —— 放宽阈值以保留更多有意义的因子）
    selected = ic_prefilter(factor_df, labels, min_ic=0.02, min_icir=0.10)
    if len(selected) == 0:
        logger.warning("[4/8] 无特征通过IC筛选，使用全部特征")
        selected = list(factor_df.columns)
    elif len(selected) < 5:
        logger.warning(f"[4/8] 仅 {len(selected)} 个特征通过IC筛选，保留全部")
        selected = list(factor_df.columns)
    factor_df = factor_df[selected]
    logger.info(f"[4/8] 特征选择完成: 选出 {len(selected)} 个特征")

    # 因子拥挤度监控（基于因子收益与资金流相关性，需要 flow_data 支持）
    try:
        from scipy.stats import spearmanr
        factor_returns = {}
        labels_unstacked = labels.unstack() if isinstance(labels, pd.DataFrame) else labels
        dates_feat = sorted(factor_df.index.get_level_values(1).unique())
        for d in dates_feat:
            try:
                fac_row = factor_df.xs(d, level=1, dropna=False)
                ret_row = labels_unstacked.xs(d, level=0, dropna=False)
                common = fac_row.index.intersection(ret_row.index)
                if len(common) < 20:
                    continue
                fc = fac_row.loc[common]
                rc = ret_row.loc[common]
                for col in fc.columns:
                    r, _ = spearmanr(fc[col].values, rc.values, nan_policy='omit')
                    if not np.isnan(r):
                        factor_returns.setdefault(col, []).append(r)
            except Exception:
                pass
        if factor_returns:
            fr_df = pd.DataFrame(factor_returns)
            crowding_monitor = CrowdingMonitor(lookback=60)
            flow_data = data.get("money_flow", None)
            scores = crowding_monitor.compute_crowding(fr_df, flow_data)
            remaining_factors = crowding_monitor.filter_crowded(list(factor_df.columns), scores, threshold=0.6)
            removed = set(factor_df.columns) - set(remaining_factors)
            if removed:
                logger.info(f"[4/8] 因子拥挤监控: 剔除 {len(removed)} 个拥挤因子: {list(removed)[:10]}")
                factor_df = factor_df[remaining_factors]
                logger.info(f"[4/8] 拥挤度得分（前10）: {scores.sort_values(ascending=False).head(10).to_dict()}")
            else:
                logger.info(f"[4/8] 因子拥挤监控: 无因子被剔除（阈值=0.6），得分范围=[{scores.min():.3f}, {scores.max():.3f}]")
        else:
            logger.info("[4/8] 因子拥挤监控: 因子收益计算失败，跳过")
    except Exception as e:
        logger.info(f"[4/8] 因子拥挤监控: 初始化失败 ({e})，跳过")


    # 构建衍生特征
    factor_df = build_momentum_features(factor_df)
    factor_df = build_cross_sectional_rank_features(factor_df)
    factor_df = build_reversal_features(factor_df)
    factor_df = build_ic_time_series(factor_df, labels)
    logger.info(f"[4/8] 特征工程完成: 共 {len(factor_df.columns)} 个特征")

    # Step 7: Walk-Forward 训练
    logger.info("[5/8] 训练模型 (Walk-Forward)...")
    dates = sorted(factor_df.index.get_level_values(1).unique())
    logger.info(f"[5/8] 训练数据: {len(factor_df)} 行, {len(factor_df.columns)} 个特征, {len(dates)} 个交易日, 范围={dates[0]} 到 {dates[-1]}")
    tuner = OptunaTuner(config["training"].get("optuna_trials", 50))
    trainer = WalkForwardTrainer(config, tuner=tuner)
    market_cap_for_trainer = data["market_cap"] if "market_cap" in data.columns else None
    predictions = trainer.train(factor_df, labels, dates, market_cap=market_cap_for_trainer)
    logger.info(f"[5/8] 训练完成: {len(predictions)} 个预测")
    if len(predictions) == 0:
        logger.error("[5/8] 生成了零个预测。请检查训练窗口大小与可用交易日数量，以及因子/标签的有效性。")
        logger.error(f"[5/8]   factor_df 形状={factor_df.shape}, 空值占比={factor_df.isna().sum().sum() / factor_df.size * 100:.1f}")
        logger.error(f"[5/8]   labels 形状={labels.shape}, 空值占比={labels.isna().sum() / len(labels) * 100:.1f}")

    # Step 8: 回测
    logger.info("[6/8] 回测...")
    prices = data["close"].unstack(level=0)  # (date, instrument) DataFrame

    # 从 akshare 获取真实的沪深300指数基准收益
    try:
        bench_df = ak.stock_zh_index_daily(symbol="sh000300")
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bench_csi300 = bench_df.set_index("date").sort_index()["close"].pct_change().dropna()
        # 对齐到策略交易日历
        bench_csi300 = bench_csi300.reindex(prices.index).fillna(0.0)
        logger.info(f"[6/8] CSI300 基准已加载: {len(bench_csi300)} 个交易日, "
                    f"总收益率={bench_csi300.sum():.4%}")
    except Exception as e:
        logger.warning(f"[6/8] 无法从 akshare 获取 CSI300 指数: {e}，使用等权替代")
        bench_csi300 = prices.ffill().pct_change(fill_method=None).mean(axis=1).dropna()
    benchmark_ret = bench_csi300

    if len(predictions) == 0:
        logger.warning("[6/8] 无预测结果，跳过回测")
        metrics = {"annualized_return": 0.0, "annualized_vol": 0.0, "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0, "calmar_ratio": 0.0, "total_return": 0.0}
        weights = pd.DataFrame()
    else:
        engine = BacktestEngine(config)
        market_cap = data["market_cap"] if "market_cap" in data else None
        portfolio_returns, benchmark_returns, weights, ann_periods_per_year = engine.run(predictions, prices, benchmark_ret, market_cap=market_cap)

        # 生成绩效报告
        labels_unstacked = labels.unstack() if isinstance(labels, pd.DataFrame) else labels
        report = MetricsReport(portfolio_returns, benchmark_returns, predictions, labels_unstacked, weights,
                             holding_period=engine.holding_period, ann_periods_per_year=ann_periods_per_year)
        metrics = report.generate()
        logger.info("\n=== 绩效指标 ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # 生成可视化报告（净值曲线、回撤曲线、IC时序图）
        try:
            analyzer = ResultAnalyzer(output_dir="results")
            analyzer.generate_report(
                returns=portfolio_returns,
                benchmark=benchmark_returns,
                predictions=predictions,
                actuals=labels_unstacked,
                weights=weights,
                metrics=metrics,
            )
        except Exception as e:
            logger.warning(f"[6/8] 可视化报告生成失败: {e}")

    # Step 9: 模型可解释性（可选）
    if config.get("run", {}).get("interpretability", True):
        logger.info("[7/8] 模型可解释性分析...")
        if trainer.models:
            date_level = 0 if factor_df.index.nlevels >= 2 and np.issubdtype(
                factor_df.index.get_level_values(0).dtype, np.datetime64
            ) else 1
            if _has_shap:
                try:
                    shap_analyzer = SHAPAnalyzer()
                    first_model = trainer.models[0]
                    first_date = trainer.dates[0]
                    X_sample = factor_df.loc[factor_df.index.get_level_values(date_level) == first_date]
                    if len(X_sample) > 0:
                        shap_analyzer.analyze(first_model, X_sample.iloc[:min(100, len(X_sample))])
                        top_features = shap_analyzer.get_top_features(10)
                        logger.info("  SHAP重要性前10特征: %s", [f[0] for f in top_features])
                        try:
                            shap_analyzer.plot_global_importance(top_n=20)
                            shap_analyzer.plot_beeswarm()
                            shap_analyzer.plot_waterfall(idx=0)
                            logger.info("  SHAP图表已保存至 results/")
                        except Exception as e:
                            logger.warning(f"  SHAP图表绘制失败: {e}")
                except Exception as e:
                    logger.warning(f"  SHAP分析失败: {e}")
            else:
                logger.info("  SHAP不可用，跳过SHAP分析")

            try:
                imp_tracker = ImportanceTracker()
                for model, date in zip(trainer.models, trainer.dates):
                    imp_tracker.record(model, date, list(factor_df.columns))
                imp_tracker.plot_heatmap(15)
                decaying = imp_tracker.get_decaying_features()
                if decaying:
                    logger.info(f"  关注衰减特征: {decaying}")
            except Exception as e:
                logger.warning(f"  重要性跟踪不可用: {e}")
    else:
        logger.info("[7/8] 模型可解释性分析已跳过 (config.run.interpretability=false)")

    # Step 10: 蒙特卡洛稳健性检验（可选）
    if config.get("run", {}).get("monte_carlo", True):
        if not trainer.models:
            logger.info("[8/8] 无可用模型，跳过蒙特卡洛模拟")
            mc_results = {}
        else:
            logger.info("[8/8] 运行蒙特卡洛模拟...")
            mc = MonteCarloSimulator(config)
            mc_results = mc.run(factor_df, prices, benchmark_ret,
                                 models=trainer.models, dates=trainer.dates)
            if mc_results:
                logger.info(mc.report(mc_results))
    else:
        logger.info("[8/8] 蒙特卡洛模拟已跳过 (config.run.monte_carlo=false)")
        mc_results = {}

    logger.info("\n=== Pipeline完成! ===")
    return metrics, mc_results, trainer


if __name__ == "__main__":
    run_pipeline()