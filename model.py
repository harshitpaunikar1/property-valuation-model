"""
Property valuation model for real estate price estimation.
Combines hedonic regression, gradient boosting, and location-based features.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class PropertyFeatureEngineer:
    """Adds domain-specific property features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "total_area_sqft" in df.columns and "plot_area_sqft" in df.columns:
            df["floor_area_ratio"] = df["total_area_sqft"] / df["plot_area_sqft"].replace(0, np.nan)
        if "built_year" in df.columns:
            df["property_age"] = 2024 - df["built_year"]
        if "bedrooms" in df.columns and "bathrooms" in df.columns:
            df["room_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
        if "total_area_sqft" in df.columns and "bedrooms" in df.columns:
            df["sqft_per_bedroom"] = df["total_area_sqft"] / df["bedrooms"].replace(0, np.nan)
        if "lat" in df.columns and "lng" in df.columns:
            df["lat_lng_interaction"] = df["lat"] * df["lng"]
        return df


class PropertyValuationModel:
    """
    Hedonic property valuation model with log-transformed target.
    Supports multiple regressors and provides MAPE, RMSE, and R2.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "price", log_transform: bool = True):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.log_transform = log_transform
        self.engineer = PropertyFeatureEngineer()
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                 self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict:
        models = {
            "Ridge": Ridge(alpha=10.0),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05,
                                                  max_depth=5, random_state=42,
                                                  tree_method="hist", verbosity=0)
        return models

    def mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        df = df[num_cols + cat_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in cat_cols:
            df[col] = df[col].fillna("unknown")

        X = df[num_cols + cat_cols]
        y_raw = df[self.target_col].values
        y = np.log1p(y_raw) if self.log_transform else y_raw

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        y_test_raw = np.expm1(y_test) if self.log_transform else y_test

        preprocessor = self._preprocessor()
        self.results = []
        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            preds_raw = np.maximum(np.expm1(preds) if self.log_transform else preds, 0)
            rmse = float(np.sqrt(mean_squared_error(y_test_raw, preds_raw)))
            mae = float(mean_absolute_error(y_test_raw, preds_raw))
            r2 = float(r2_score(y_test, preds))
            mape = self.mape(y_test_raw, preds_raw)
            self.models[name] = pipe
            self.results.append({"model": name, "rmse": round(rmse, 0),
                                 "mae": round(mae, 0), "r2": round(r2, 4),
                                 "mape_pct": round(mape, 2)})

        results_df = pd.DataFrame(self.results).sort_values("mape_pct").reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model_name not in self.models:
            raise RuntimeError("Call fit() first.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        preds = self.models[self.best_model_name].predict(df[num_cols + cat_cols])
        return np.expm1(preds).astype(int) if self.log_transform else preds

    def valuation_band(self, price: float, city: str = "metro") -> str:
        thresholds = {
            "metro": [(5_000_000, "budget"), (10_000_000, "mid"), (25_000_000, "premium"),
                      (float("inf"), "luxury")],
            "tier2": [(2_000_000, "budget"), (5_000_000, "mid"), (12_000_000, "premium"),
                      (float("inf"), "luxury")],
        }
        bands = thresholds.get(city, thresholds["metro"])
        for threshold, label in bands:
            if price < threshold:
                return label
        return "luxury"

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        prep = pipe.named_steps["preprocessor"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
        except Exception:
            cat_names = []
        names = self.numeric_features + cat_names
        return pd.DataFrame({
            "feature": names[:len(est.feature_importances_)],
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame({
        "total_area_sqft": np.random.uniform(500, 8000, n),
        "plot_area_sqft": np.random.uniform(1000, 15000, n),
        "bedrooms": np.random.randint(1, 7, n).astype(float),
        "bathrooms": np.random.randint(1, 5, n).astype(float),
        "floors": np.random.randint(1, 5, n).astype(float),
        "built_year": np.random.randint(1980, 2023, n).astype(float),
        "parking": np.random.randint(0, 3, n).astype(float),
        "facing": np.random.choice(["North", "South", "East", "West"], n),
        "locality_tier": np.random.choice(["premium", "mid", "budget"], n),
        "property_type": np.random.choice(["apartment", "villa", "rowhouse", "plot"], n),
        "price": np.abs(np.random.lognormal(15.5, 0.7, n)),
    })

    model = PropertyValuationModel(
        numeric_features=["total_area_sqft", "plot_area_sqft", "bedrooms", "bathrooms",
                           "floors", "built_year", "parking"],
        categorical_features=["facing", "locality_tier", "property_type"],
    )
    results = model.fit(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {model.best_model_name}")

    sample_preds = model.predict(df.head(5))
    for p in sample_preds:
        band = model.valuation_band(p, "metro")
        print(f"  Predicted value: Rs {p:,.0f} [{band}]")
