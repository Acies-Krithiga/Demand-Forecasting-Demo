import logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.Eda.eda import run_eda
from src.seg_rule.segmentation import Segmentation
from src.seg_rule.rules import ForecastRuleAssigner
from config.config import ML_MODELS_CONFIG
# from src.seg_rule.segmentation import Segmentation
# from src.seg_rule.rules import ForecastRuleAssigner
# from src.forecast.statistical.stat_forecast import DemandForecastingSystem
# from src.forecast.statistical.bestfit import BestFitAnalyzer
# from src.forecast.statistical.generate_bestfit import ForecastGenerator
# from src.forecast.ml.Feature_engineering.data_merge import FeatureEngineering
# from src.forecast.ml.model.bestfit_future import BestFitMLForecaster
# from src.forecast.ml.model.finding_bestfit import MLBestFitAnalyzer
# from src.forecast.ml.model.generate_ml_forecast import MLForecastGenerator
# from src.Eda.eda import run_eda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
DATA_OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


def _save_placeholder_csv(path: Path, columns: list[str]) -> None:
    """Write an empty CSV with headers so downstream pages can load it safely."""
    if path.exists() and path.stat().st_size > 0:
        return
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _ensure_baseline_placeholders() -> None:
    """Create baseline output files even when forecasting cannot produce rows."""
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "base_forecast_df.csv",
        ["date", "actual", "store_id", "item_id", "Moving Average", "Weighted Snaive"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "mape_baseline_df.csv",
        ["store_id", "item_id", "ma_mape", "sn_mape"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "base_future_df.csv",
        ["date", "store_id", "item_id", "Moving Average", "Weighted Snaive"],
    )


def _ensure_ml_placeholders() -> None:
    """Create ML output files so the dashboard can load safely after partial failures."""
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "df_feat_selected.csv",
        ["date", "store_id", "item_id", "units_sold"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "feature_importance.csv",
        ["feature", "importance"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "ml_forecast_df.csv",
        [
            "date",
            "actual",
            "store_id",
            "item_id",
            *ML_MODELS_CONFIG["available_algorithms"],
        ],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "best_fit_ml_df.csv",
        ["store_id", "item_id", "best_fit_algorithm", "best_mape"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "ml_bestfit_predictions_future.csv",
        ["date", "store_id", "item_id", "algorithm", "prediction"],
    )


def _ensure_probability_placeholders() -> None:
    """Create probability output files so the dashboard can load safely after partial failures."""
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "prob_residuals.csv",
        [
            "date",
            "store_id",
            "item_id",
            "point_forecast",
            "q05",
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
            "q95",
        ],
    )


def _ensure_statistical_placeholders() -> None:
    """Create statistical output files so the dashboard can load safely after partial failures."""
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "stat_forecasting.csv",
        ["date", "store_id", "item_id", "actual"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "future_statforecast.csv",
        ["date", "store_id", "item_id", "algorithm", "forecast"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "best_fit_df.csv",
        ["store_id", "item_id", "best_fit_algorithm", "best_mape"],
    )
    _save_placeholder_csv(
        DATA_OUTPUTS_DIR / "future_probability_forecasts.csv",
        [
            "date",
            "store_id",
            "item_id",
            "point_forecast",
            "q05",
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
            "q95",
        ],
    )


def read_csv_or_fail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def read_csv_or_warn(path: Path, step_name: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("[%s] Skipping. Missing file: %s", step_name, path)
        return None
    return pd.read_csv(path)


def run_uplift_forecast() -> None:
    from src.Feature_effect.feature_effect import UpliftDemandForecaster

    step_name = "Uplift"
    df1 = read_csv_or_warn(DATA_OUTPUTS_DIR / "df_feat_selected.csv", step_name)
    if df1 is None:
        return

    df = df1[(df1["store_id"] == "STORE_02") & (df1["item_id"] == "ITEM_002")]
    if df.empty:
        logger.warning("[%s] Skipping. No rows found for STORE_02 / ITEM_002.", step_name)
        return

    forecaster = UpliftDemandForecaster(df)
    forecast_results = forecaster.run_all_forecasts()
    print(forecast_results.head())
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast_results.to_csv(DATA_OUTPUTS_DIR / "uplift_effect.csv", index=False)
    logger.info("[%s] Saved: %s", step_name, DATA_OUTPUTS_DIR / "uplift_effect.csv")


def run_probability_forecast() -> None:
    from src.forecast.probability.prob import ProbabilityForecasting
    import platform

    if platform.system() == 'Windows':
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    step_name = "Probability"
    hist_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    future_df = read_csv_or_warn(DATA_OUTPUTS_DIR / "future_statforecast.csv", step_name)
    if hist_df is None or future_df is None:
        _ensure_probability_placeholders()
        return

    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        prob = ProbabilityForecasting(hist_df, future_df)
        print("Calculating residuals...")
        residuals_df = prob.generate_probability_forecasts()
        if residuals_df is None or residuals_df.empty:
            logger.warning("[%s] Probability forecasting returned no rows.", step_name)
            _ensure_probability_placeholders()
            return

        residuals_df.to_csv(DATA_OUTPUTS_DIR / "prob_residuals.csv", index=False)
        residuals_df.to_csv(DATA_OUTPUTS_DIR / "future_probability_forecasts.csv", index=False)
        logger.info("[%s] Saved: %s, %s", step_name, DATA_OUTPUTS_DIR / "prob_residuals.csv", DATA_OUTPUTS_DIR / "future_probability_forecasts.csv")
    except Exception as e:
        logger.warning("[%s] Skipping due to runtime error: %s", step_name, e)
        _ensure_probability_placeholders()


def run_segmentation_pipeline() -> None:
    step_name = "Segmentation"
    sales_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    if sales_df is None:
        return

    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seg_df = Segmentation(sales_df).run()
    seg_df.to_csv(DATA_OUTPUTS_DIR / "seg_df.csv", index=False)

    rules_df = ForecastRuleAssigner(seg_df).assign_rules()
    rules_df.to_csv(DATA_OUTPUTS_DIR / "rules_df.csv", index=False)
    logger.info("[%s] Saved: %s, %s", step_name, DATA_OUTPUTS_DIR / "seg_df.csv", DATA_OUTPUTS_DIR / "rules_df.csv")


def run_outlier_correction_pipeline() -> None:
    step_name = "Outlier"
    sales_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    if sales_df is None:
        return

    required_cols = ["date", "store_id", "item_id", "units_sold"]
    missing = [c for c in required_cols if c not in sales_df.columns]
    if missing:
        logger.warning("[%s] Skipping. Missing columns in sales_fact.csv: %s", step_name, missing)
        return

    df = sales_df.copy()
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")

    group_cols = ["store_id", "item_id"]
    q1 = df.groupby(group_cols)["units_sold"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby(group_cols)["units_sold"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df["units_sold_corrected"] = df["units_sold"].clip(lower=lower, upper=upper)
    df["units_sold_corrected"] = df["units_sold_corrected"].fillna(df["units_sold"])

    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_OUTPUTS_DIR / "units_sold_corrected.csv", index=False)
    logger.info("[%s] Saved: %s", step_name, DATA_OUTPUTS_DIR / "units_sold_corrected.csv")


def run_baseline_pipeline() -> None:
    step_name = "Baseline"
    sales_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    if sales_df is None:
        _ensure_baseline_placeholders()
        return

    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = DATA_OUTPUTS_DIR / "base_forecast_df.csv"
    mape_path = DATA_OUTPUTS_DIR / "mape_baseline_df.csv"
    max_combos = int(os.getenv("BASELINE_MAX_COMBINATIONS", "50"))

    baseline_exists = baseline_path.exists() and baseline_path.stat().st_size > 0
    mape_exists = mape_path.exists() and mape_path.stat().st_size > 0

    # Reuse existing baseline outputs when available to avoid very long recomputation.
    if not (baseline_exists and mape_exists):
        try:
            from src.forecast.baseline.base import BaselineForecastingSystem
            from src.forecast.baseline.base_mape import BaselineMAPE
        except Exception as e:
            logger.warning("[%s] Skipping. Baseline dependencies unavailable: %s", step_name, e)
            _ensure_baseline_placeholders()
            return

        baseline_system = BaselineForecastingSystem(sales_df)
        if len(baseline_system.valid_combinations) > max_combos:
            logger.warning(
                "[%s] Limiting baseline combinations from %d to %d for faster dashboard generation. Set BASELINE_MAX_COMBINATIONS to override.",
                step_name,
                len(baseline_system.valid_combinations),
                max_combos,
            )
            baseline_system.valid_combinations = baseline_system.valid_combinations[:max_combos]

        baseline_df = baseline_system.run_forecasting()
        if baseline_df is None or baseline_df.empty:
            logger.warning("[%s] Skipping. Baseline forecasting returned no rows.", step_name)
            _ensure_baseline_placeholders()
            return
        baseline_df.to_csv(baseline_path, index=False)

        mape_df = BaselineMAPE(baseline_df).run()
        mape_df.to_csv(mape_path, index=False)
    else:
        logger.info("[%s] Reusing existing baseline files: %s, %s", step_name, baseline_path, mape_path)

    # Generate future baseline forecasts for dashboard section
    try:
        from src.forecast.baseline.base_future import BaseFuture

        base_future = BaseFuture(sales_df)
        if len(base_future.valid_combinations) > max_combos:
            base_future.valid_combinations = base_future.valid_combinations[:max_combos]
        base_future_df = base_future.generate_future_forecast()
        if base_future_df is not None and not base_future_df.empty:
            base_future_path = DATA_OUTPUTS_DIR / "base_future_df.csv"
            base_future_df.to_csv(base_future_path, index=False)
            logger.info(
                "[%s] Saved: %s, %s, %s",
                step_name,
                baseline_path,
                mape_path,
                base_future_path,
            )
        else:
            logger.info("[%s] Saved: %s, %s", step_name, baseline_path, mape_path)
            logger.warning("[%s] Base future forecasting returned no rows.", step_name)
    except Exception as e:
        logger.info("[%s] Saved: %s, %s", step_name, baseline_path, mape_path)
        logger.warning("[%s] Skipping base future forecasting: %s", step_name, e)
        if not baseline_path.exists() or not mape_path.exists():
            _ensure_baseline_placeholders()


def safe_run(step_name: str, step_fn) -> None:
    """Run a pipeline step without letting one unexpected failure stop the rest."""
    try:
        step_fn()
    except Exception as e:
        logger.exception("[%s] Unhandled pipeline error: %s", step_name, e)


def run_statistical_pipeline() -> None:
    step_name = "Statistical"
    sales_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    rules_df = read_csv_or_warn(DATA_OUTPUTS_DIR / "rules_df.csv", step_name)
    if sales_df is None or rules_df is None:
        _ensure_statistical_placeholders()
        return

    max_rules = int(os.getenv("STAT_MAX_RULES", "25"))
    if len(rules_df) > max_rules:
        logger.warning(
            "[%s] Limiting rules from %d to %d for faster dashboard generation. Set STAT_MAX_RULES to override.",
            step_name,
            len(rules_df),
            max_rules,
        )
        rules_df = rules_df.head(max_rules).copy()

    try:
        from src.forecast.statistical.stat_forecast import DemandForecastingSystem
        from src.forecast.statistical.bestfit import BestFitAnalyzer
        from src.forecast.statistical.generate_bestfit import ForecastGenerator
    except Exception as e:
        logger.warning("[%s] Skipping. Dependencies unavailable: %s", step_name, e)
        _ensure_statistical_placeholders()
        return

    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    stat_df = DemandForecastingSystem(rules_df, sales_df).run_forecasting()
    if stat_df is None or stat_df.empty:
        logger.warning("[%s] Skipping. Validation forecasting returned no rows.", step_name)
        _ensure_statistical_placeholders()
        return
    stat_path = DATA_OUTPUTS_DIR / "stat_forecasting.csv"
    stat_df.to_csv(stat_path, index=False)

    best_fit_df = BestFitAnalyzer(stat_df).analyze_all_intersections()
    if best_fit_df is None or best_fit_df.empty:
        logger.warning("[%s] Skipping. Best-fit analysis returned no rows.", step_name)
        _ensure_statistical_placeholders()
        return
    best_fit_path = DATA_OUTPUTS_DIR / "best_fit_df.csv"
    best_fit_df.to_csv(best_fit_path, index=False)

    future_df = ForecastGenerator(best_fit_df, sales_df).generate_all_forecasts()
    if future_df is None or future_df.empty:
        logger.warning("[%s] Skipping. Future forecasting returned no rows.", step_name)
        _ensure_statistical_placeholders()
        return
    future_path = DATA_OUTPUTS_DIR / "future_statforecast.csv"
    future_df.to_csv(future_path, index=False)
    logger.info("[%s] Saved: %s, %s, %s", step_name, stat_path, best_fit_path, future_path)
    _save_placeholder_csv(stat_path, ["date", "store_id", "item_id", "actual"])
    _save_placeholder_csv(best_fit_path, ["store_id", "item_id", "best_fit_algorithm", "best_mape"])
    _save_placeholder_csv(future_path, ["date", "store_id", "item_id", "algorithm", "forecast"])


def run_ml_pipeline() -> None:
    step_name = "ML"
    sales_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    if sales_df is None:
        return

    # Make sure the dashboard has readable outputs even if the ML step aborts partway through.
    _ensure_ml_placeholders()

    try:
        from src.forecast.ml.Feature_engineering.create_features import CreateFeatures
        from src.forecast.ml.Feature_engineering.feature_selection import GenericFeatureSelector
        from src.forecast.ml.model.generate_ml_forecast import MLForecastGenerator
        from src.forecast.ml.model.finding_bestfit import MLBestFitAnalyzer
        from src.forecast.ml.model.bestfit_arml_forecast import BestFitARMLForecaster
    except Exception as e:
        logger.warning("[%s] Skipping. Dependencies unavailable: %s", step_name, e)
        return

    required_cols = ["date", "store_id", "item_id", "units_sold"]
    missing = [c for c in required_cols if c not in sales_df.columns]
    if missing:
        logger.warning("[%s] Skipping. Missing required columns in sales_fact.csv: %s", step_name, missing)
        return

    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep ML runtime bounded for dashboard use.
    max_combos = int(os.getenv("ML_MAX_COMBINATIONS", "10"))
    combos = sales_df[["store_id", "item_id"]].drop_duplicates()
    if len(combos) > max_combos:
        logger.warning(
            "[%s] Limiting ML combinations from %d to %d for faster dashboard generation. Set ML_MAX_COMBINATIONS to override.",
            step_name,
            len(combos),
            max_combos,
        )
        combos = combos.head(max_combos).copy()
        sales_df = sales_df.merge(combos, on=["store_id", "item_id"], how="inner")

    try:
        # 1) Create features from sales history.
        feat_df = CreateFeatures(sales_df).create_all_features()

        # 2) Feature selection: use RF by default here for stable runtime.
        fs_cfg = {
            "method": os.getenv("ML_FEATURE_SELECTION_METHOD", "random_forest"),
            "level_columns": ["store_id", "item_id"],
            "date_column": "date",
            "target_column": "units_sold",
            "feature_importance_threshold": 0.01,
            "random_state": 42,
            "random_forest": {"n_estimators": 100, "max_depth": None, "n_jobs": -1},
            "output": {"save_feature_importance": False, "output_dir": str(DATA_OUTPUTS_DIR)},
        }
        selector = GenericFeatureSelector(config=fs_cfg)
        df_feat_selected = selector.fit_transform(feat_df)
        df_feat_selected.to_csv(DATA_OUTPUTS_DIR / "df_feat_selected.csv", index=False)

        feat_imp_df = selector.get_feature_importance_summary()
        feat_imp_df.to_csv(DATA_OUTPUTS_DIR / "feature_importance.csv", index=False)

        # 3) Validation forecasts for each ML algorithm.
        ml_forecast_df = MLForecastGenerator(df_feat_selected).generate_all_forecasts()
        if ml_forecast_df is None or ml_forecast_df.empty:
            logger.warning("[%s] Skipping. ML forecast generation returned no rows.", step_name)
            return
        ml_forecast_df.to_csv(DATA_OUTPUTS_DIR / "ml_forecast_df.csv", index=False)

        # 4) Best-fit model by intersection.
        best_fit_ml_df = MLBestFitAnalyzer(ml_forecast_df).analyze_all_intersections()
        if best_fit_ml_df is None or best_fit_ml_df.empty:
            logger.warning("[%s] Skipping. ML best-fit analysis returned no rows.", step_name)
            return
        best_fit_ml_df.to_csv(DATA_OUTPUTS_DIR / "best_fit_ml_df.csv", index=False)

        # 5) Predictions using each intersection's best ML model.
        future_ml_df = BestFitARMLForecaster(best_fit_ml_df, df_feat_selected).generate_all_predictions()
        if future_ml_df is None or future_ml_df.empty:
            logger.warning("[%s] Best-fit ML predictions returned no rows.", step_name)
            return
        future_ml_df.to_csv(DATA_OUTPUTS_DIR / "ml_bestfit_predictions_future.csv", index=False)

        logger.info(
            "[%s] Saved: %s, %s, %s",
            step_name,
            DATA_OUTPUTS_DIR / "best_fit_ml_df.csv",
            DATA_OUTPUTS_DIR / "ml_bestfit_predictions_future.csv",
            DATA_OUTPUTS_DIR / "feature_importance.csv",
        )
    except Exception as e:
        logger.warning("[%s] Skipping due to runtime error: %s", step_name, e)


def run_eda_pipeline() -> None:
    step_name = "EDA"
    sales_df = read_csv_or_warn(DATA_INPUTS_DIR / "sales_fact.csv", step_name)
    if sales_df is None:
        return

    results = run_eda(sales_df)
    eda_dir = DATA_OUTPUTS_DIR / "EDA"
    eda_dir.mkdir(parents=True, exist_ok=True)

    results["df_sales"].to_csv(eda_dir / "df_sales.csv", index=False)
    def _json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        return str(value)

    with open(eda_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results["overall_metrics"], f, ensure_ascii=False, indent=2, default=_json_default)
    results["store_breakdown_metrics"].to_csv(eda_dir / "store_breakdown_metrics.csv", index=False)
    results["category_daily_sales"].to_csv(eda_dir / "category_daily_sales.csv", index=False)
    results["category_sales_treemap"].to_csv(eda_dir / "category_sales_treemap.csv", index=False)
    with open(eda_dir / "sku_count_by_cat.json", "w", encoding="utf-8") as f:
        json.dump(results["sku_count_by_cat"], f, ensure_ascii=False, indent=2, default=_json_default)

    logger.info("[%s] Saved outputs in: %s", step_name, eda_dir)


#segmentation = Segmentation(sales_fact_df)
#seg = segmentation.run()
#seg.to_csv(r'C:\Users\Pavan\DemandForecasting\data\seg_df.csv', index=False)

#rules = ForecastRuleAssigner(seg)
#rules_df = rules.assign_rules()
#rules_df.to_csv(r'C:\Users\Pavan\DemandForecasting\data\rules_df.csv', index=False)

#demand = DemandForecastingSystem(rules_df, sales_fact_df)
#stat = demand.run_forecasting()
#stat.to_csv(r'C:\Users\Pavan\DemandForecasting\data\stat_forecasting.csv', index=False)

#stat_df = pd.read_csv(r'C:\Users\Pavan\DemandForecasting\data\stat_forecasting.csv')
#best_fit = BestFitAnalyzer(stat_df)
#best_fit_df = best_fit.analyze_all_intersections()
#best_fit_df.to_csv(r'C:\Users\Pavan\DemandForecasting\data\best_fit_df.csv', index=False)


#forecast_generator = ForecastGenerator(best_fit_df, sales_fact_df)
#forecast_df = forecast_generator.generate_all_forecasts()
#forecast_df.to_csv(r'C:\Users\Pavan\DemandForecasting\data\forecast_df.csv', index=False)
#print('Forecast generation completed')



#sales = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\sales_fact.csv")
#price = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\price_fact.csv")
#calendar = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\calendar_dim.csv")
#location = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\location_dim.csv")
#external = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\external_data.csv")
#promotion = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\promotion_fact.csv")


#feature_engineering = FeatureEngineering()
#df_feat = feature_engineering.prepare_master_dataset(sales_df=sales, price_df=price, calendar_df=calendar, location_df=location, external_df=external, promotion_df=promotion)

#df_feat.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\df_feat.csv", index=False)



# df= pd.read_csv(r"C:\Users\Nithish\Downloads\DemandForecasting-main\df_feat.csv")
# df1 = df[(df["store_id"] == "STORE_02") & (df["item_id"] == "ITEM_002")]
# # Feature Selection
# print("Starting Feature Selection...")
# feature_selector = GenericFeatureSelector()
# df_feat_selected = feature_selector.fit_transform(df1)
# print(df_feat_selected.head())



# ML Forecasting
#print("Starting ML Forecasting...")
#ml_forecast_generator = MLForecastGenerator(df_feat_selected)
#ml_forecasts = ml_forecast_generator.generate_all_forecasts()


# Save ML forecasts
#ml_forecasts.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\ml_forecast_df.csv", index=False)

#ml_analyzer = MLBestFitAnalyzer(df_feat_selected)
#ml_best_fit_df = ml_analyzer.analyze_all_intersections()
#ml_best_fit_df.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\ml_best_fit_df.csv", index=False)

# Future ML Forecast using BestFitMLForecaster
#df_feat_selected = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\df_train_ml.csv")
#ml_best_fit_df = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\best_fit_ml_df.csv")
#future_features_df = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\df_futureml.csv")

#best_fit_ml_forecaster = BestFitMLForecaster(ml_best_fit_df, df_feat_selected, future_features_df)
#ml_predictions = best_fit_ml_forecaster.generate_all_predictions()
#ml_predictions.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\ml_bestfit_predictions.csv", index=False)


# Future ML Forecast using BestFitARMLForecaster

# df_feat_selected = pd.read_csv(r"C:\Users\Nithish\Documents\GitHub\DemandForecasting\data\outputs\df_feat_selected.csv")
# ml_best_fit_df = pd.read_csv(r"C:\Users\Nithish\Documents\GitHub\DemandForecasting\data\outputs\best_fit_ml_df.csv")
# # Initialize the BestFitMLForecaster with both best_fit_df and feature_df
# df1 = df_feat_selected[(df_feat_selected["store_id"] == "STORE_02") & (df_feat_selected["item_id"] == "ITEM_002")]
# df2 = ml_best_fit_df[(ml_best_fit_df["store_id"] == "STORE_02") & (ml_best_fit_df["item_id"] == "ITEM_002")]

# best_fit_arml_forecaster = BestFitARMLForecaster(df2, df1)

# # Generate predictions using the best fit models
# ml_predictions = best_fit_arml_forecaster.generate_all_predictions()
# print(ml_predictions.head())


#sales_fact_df = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\sales_fact.csv")

#from src.forecast.baseline.base import BaselineForecastingSystem
#print("Starting baseline forecasting...")
#base_forecast = BaselineForecastingSystem(sales_fact_df)
#base_forecast_df = base_forecast.run_forecasting()
#base_forecast_df.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\base_forecast_df.csv", index=False)
#print("Baseline forecasting completed")
#from src.forecast.baseline.base_mape import BaselineMAPE
#df = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\base_forecast_df.csv")

#mapecalculator = BaselineMAPE(df)
#mape_df = mapecalculator.run()
#mape_df.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\mape_baseline_df.csv", index=False)
#print("MAPE calculation completed")




#from src.forecast.ml.Feature_engineering.create_features import CreateFeatures

#df = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\df_feat.csv")

#feature_creator = CreateFeatures(df)
#df_with_lag_features = feature_creator.create_all_features()
#df_with_lag_features.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\df_with_lag_features.csv", index=False)


#from src.forecast.baseline.base_future import BaseFuture
#sales_fact_df = pd.read_csv(r"C:\Users\Pavan\DemandForecasting\data\inputs\sales_fact.csv")
#print("Starting base future forecasting...")
#base_future = BaseFuture(sales_fact_df)
#base_future_df = base_future.generate_future_forecast()
#base_future_df.to_csv(r"C:\Users\Pavan\DemandForecasting\data\outputs\base_future_df.csv", index=False)
#print("Base future forecasting completed")

if __name__ == '__main__':
    pipeline_steps = [
        ("Segmentation", run_segmentation_pipeline),
        ("Baseline", run_baseline_pipeline),
        ("Statistical", run_statistical_pipeline),
        ("ML", run_ml_pipeline),
        ("Probability", run_probability_forecast),
        ("Outlier", run_outlier_correction_pipeline),
        ("EDA", run_eda_pipeline),
        ("Uplift", run_uplift_forecast),
    ]
    for step_name, step_fn in pipeline_steps:
        safe_run(step_name, step_fn)


