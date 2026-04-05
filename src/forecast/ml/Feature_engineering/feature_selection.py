import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import shap
except Exception:  # pragma: no cover
    shap = None
import logging
import warnings
import os
import matplotlib.pyplot as plt
from config.config import FEATURE_SELECTION_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GenericFeatureSelector:

    def __init__(self, config: Dict = None):
        if config is None:
            config = FEATURE_SELECTION_CONFIG

        self.config = config
        self.method = config.get("method", "random_forest").lower()

        # General configs
        self.date_column = config.get('date_column', 'date')
        self.target_column = config.get('target_column', 'units_sold')
        self.level_columns = config.get('level_columns', ['store_id', 'item_id'])
        self.feature_importance_threshold = config.get('feature_importance_threshold', 0.01)
        self.random_state = config.get('random_state', 42)

        # Random Forest configs
        rf_config = config.get('random_forest', {})
        self.rf_params = {
            'n_estimators': rf_config.get('n_estimators', 100),
            'max_depth': rf_config.get('max_depth', None),
            'n_jobs': rf_config.get('n_jobs', -1),
            'random_state': self.random_state
        }

        # LASSO configs
        lasso_config = config.get('lasso', {})
        self.lasso_alpha = lasso_config.get('alpha', None)
        self.lasso_cv = lasso_config.get('cv', 5)

        # Output configs
        self.output_config = config.get('output', {})
        
        # Encoders / Models
        self.label_encoders = {}
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.feature_importance_df = None
        self.selected_features = None


    # ========== Common Data Preparation Methods ==========

    def _filter_intersection_data(self, df: pd.DataFrame, intersection_values: Dict) -> pd.DataFrame:
        filter_conditions = [df[col] == val for col, val in intersection_values.items()]
        return df[np.logical_and.reduce(filter_conditions)].copy()

    def _identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        exclude_cols = [self.target_column, self.date_column,'revenue'] + self.level_columns
        categorical, numerical = [], []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype == 'object':
                categorical.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() <= 20:
                    categorical.append(col)
                else:
                    numerical.append(col)
        return {'categorical': categorical, 'numerical': numerical, 'exclude': exclude_cols}

    def _encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str], fit=True):
        df_encoded = df.copy()
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    unseen = set(df_encoded[col].astype(str)) - set(le.classes_)
                    if unseen:
                        logger.warning(f"Unseen categories in {col}: {unseen}")
                        df_encoded[col] = df_encoded[col].astype(str).replace(list(unseen), le.classes_[0])
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))
        return df_encoded

    def _prepare_features(self, df, column_types, fit=True):
        df_processed = self._encode_categorical_features(df, column_types['categorical'], fit)
        feature_cols = column_types['numerical'] + column_types['categorical']
        for col in feature_cols:
            if df_processed[col].isnull().any():
                if col in column_types['categorical']:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
        return df_processed, feature_cols

    # ========== Feature Selection Methods ==========

    def _train_random_forest(self, X, y, feature_cols):
        self.rf_model.fit(X[feature_cols], y)
        importance = self.rf_model.feature_importances_
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importance}).sort_values('importance', ascending=False)
        selected = importance_df[importance_df['importance'] >= self.feature_importance_threshold]['feature'].tolist()
        return importance_df, selected

    def _train_lasso(self, X, y, feature_cols):
        X_scaled = self.scaler.fit_transform(X[feature_cols])
        model = LassoCV(alphas=None if self.lasso_alpha is None else [self.lasso_alpha],
                        cv=self.lasso_cv, random_state=self.random_state)
        model.fit(X_scaled, y)
        coef_importance = np.abs(model.coef_)
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': coef_importance}).sort_values('importance', ascending=False)
        selected = importance_df[importance_df['importance'] >= self.feature_importance_threshold]['feature'].tolist()
        return importance_df, selected

    def _train_shap(self, X, y, feature_cols):
        if shap is None:
            raise ImportError("shap is not installed")

         # Fit model
        self.rf_model.fit(X[feature_cols], y)
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(X[feature_cols])

        # Handle classification case (list of arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # take first class for binary

        # Compute mean absolute SHAP for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        # Select important features
        selected = importance_df[
            importance_df['importance'] >= self.feature_importance_threshold    
        ]['feature'].tolist()

        # # --- Visualization ---
        # shap.summary_plot(shap_values, X[feature_cols], plot_type="bar", show=False)
        # plt.title("Global Feature Importance (Mean |SHAP|)")
        # plt.tight_layout()
        # plt.show()

        # for f in importance_df['feature'].head(3):  # top 3 features
        #     shap.dependence_plot(f, shap_values, X[feature_cols], show=False)
        #     plt.title(f"SHAP Dependence Plot: {f}")
        #     plt.tight_layout()
        #     plt.show()

        
        return importance_df, selected

    # ========== Main Workflow ==========

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature selection using method: {self.method.upper()}")

        col_types = self._identify_column_types(df)
        df_processed, feature_cols = self._prepare_features(df, col_types, fit=True)
        y = df_processed[self.target_column]

        # Choose method
        if self.method == "random_forest":
            imp_df, selected = self._train_random_forest(df_processed, y, feature_cols)
        elif self.method == "lasso":
            imp_df, selected = self._train_lasso(df_processed, y, feature_cols)
        elif self.method == "shap":
            imp_df, selected = self._train_shap(df_processed, y, feature_cols)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

        self.feature_importance_df = imp_df
        self.selected_features = selected

        # Create final dataframe
        output_cols = [self.date_column, self.target_column] + self.level_columns + selected
        output_df = df_processed[[c for c in output_cols if c in df_processed.columns]]

        if self.output_config.get('save_feature_importance', False):
            self.save_feature_importance()

        logger.info(f"Feature selection completed using {self.method.upper()}. {len(selected)} features selected.")
        return output_df

    # ========== Utility Methods ==========

    def get_feature_importance_summary(self):
        if self.feature_importance_df is None:
            raise ValueError("Feature selector not fitted yet.")
        return self.feature_importance_df.copy()

    def save_feature_importance(self, output_dir=None):
        output_dir = output_dir or self.output_config.get('output_dir', 'data/outputs')
        os.makedirs(output_dir, exist_ok=True)
        if self.feature_importance_df is not None:
            path = os.path.join(output_dir, f"feature_importance_{self.method}.csv")
            self.feature_importance_df.to_csv(path, index=False)
            logger.info(f"Feature importance saved to {path}")
