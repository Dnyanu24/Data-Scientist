"""Adaptive Preprocessing Pipeline Builder."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Adaptive preprocessing that builds and applies transformation pipelines."""

    def __init__(self):
        self.transformers = {}
        self.feature_columns = []
        self.target_column = None
        self.label_encoder = None
        self.steps_applied = []

    def build_pipeline(self, df: pd.DataFrame, target_column: str,
                       problem_type: str, fingerprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze dataset and build adaptive preprocessing steps."""
        steps = []

        # Analyze columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        # Drop high-cardinality text and constant columns
        cols_to_drop = []
        for col in df.columns:
            if col == target_column:
                continue
            if df[col].nunique() <= 1:
                cols_to_drop.append(col)
            elif col in categorical_cols and df[col].nunique() > 100:
                cols_to_drop.append(col)

        if cols_to_drop:
            steps.append({
                "step": "drop_columns",
                "columns": cols_to_drop,
                "reason": "Constant or high-cardinality columns"
            })
            numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]
            categorical_cols = [c for c in categorical_cols if c not in cols_to_drop]

        # Handle missing values - numeric
        missing_numeric = [c for c in numeric_cols if df[c].isnull().any()]
        if missing_numeric:
            # Use median for skewed data, mean for normal
            avg_skew = df[missing_numeric].skew().abs().mean() if len(missing_numeric) > 0 else 0
            strategy = "median" if avg_skew > 1 else "mean"
            steps.append({
                "step": "impute_numeric",
                "strategy": strategy,
                "columns": missing_numeric,
            })

        # Handle missing values - categorical
        missing_cat = [c for c in categorical_cols if df[c].isnull().any()]
        if missing_cat:
            steps.append({
                "step": "impute_categorical",
                "strategy": "most_frequent",
                "columns": missing_cat,
            })

        # Handle outliers
        outlier_ratio = fingerprint.get("outlier_ratio", 0)
        if outlier_ratio > 0.01 and numeric_cols:
            steps.append({
                "step": "handle_outliers",
                "strategy": "clip",  # clip to 1st/99th percentile
                "columns": numeric_cols,
            })

        # Encode categorical
        if categorical_cols:
            # Use target encoding for high cardinality, onehot for low
            low_card = [c for c in categorical_cols if df[c].nunique() <= 10]
            high_card = [c for c in categorical_cols if df[c].nunique() > 10]

            if low_card:
                steps.append({
                    "step": "encode_categorical",
                    "strategy": "onehot",
                    "columns": low_card,
                })
            if high_card:
                steps.append({
                    "step": "encode_categorical",
                    "strategy": "label",
                    "columns": high_card,
                })

        # Scale numeric features
        if numeric_cols:
            if outlier_ratio > 0.05:
                scale_strategy = "robust"
            else:
                scale_strategy = "standard"
            steps.append({
                "step": "scale_numeric",
                "strategy": scale_strategy,
                "columns": numeric_cols,
            })

        # Feature selection if many features
        feature_ratio = fingerprint.get("feature_to_sample_ratio", 0)
        total_features = len(numeric_cols) + len(categorical_cols)
        if total_features > 50 or feature_ratio > 0.1:
            k = min(30, total_features)
            steps.append({
                "step": "feature_selection",
                "strategy": "mutual_info",
                "k": k,
            })

        # Encode target for classification
        if problem_type in ["binary_classification", "multiclass_classification"]:
            if target_column in df.columns and df[target_column].dtype == "object":
                steps.append({
                    "step": "encode_target",
                    "strategy": "label",
                    "column": target_column,
                })

        return steps

    def apply_pipeline(self, df: pd.DataFrame, steps: List[Dict[str, Any]],
                       target_column: str, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply preprocessing pipeline to dataframe."""
        df = df.copy()
        self.target_column = target_column
        self.steps_applied = []

        for step_config in steps:
            step = step_config["step"]

            try:
                if step == "drop_columns":
                    cols = [c for c in step_config["columns"] if c in df.columns]
                    df = df.drop(columns=cols)

                elif step == "impute_numeric":
                    cols = [c for c in step_config["columns"] if c in df.columns]
                    if cols:
                        if fit:
                            imputer = SimpleImputer(strategy=step_config["strategy"])
                            df[cols] = imputer.fit_transform(df[cols])
                            self.transformers[f"impute_numeric"] = imputer
                        else:
                            imputer = self.transformers.get("impute_numeric")
                            if imputer:
                                df[cols] = imputer.transform(df[cols])

                elif step == "impute_categorical":
                    cols = [c for c in step_config["columns"] if c in df.columns]
                    if cols:
                        if fit:
                            imputer = SimpleImputer(strategy="most_frequent")
                            df[cols] = imputer.fit_transform(df[cols])
                            self.transformers["impute_cat"] = imputer
                        else:
                            imputer = self.transformers.get("impute_cat")
                            if imputer:
                                df[cols] = imputer.transform(df[cols])

                elif step == "handle_outliers":
                    cols = [c for c in step_config["columns"] if c in df.columns]
                    if fit:
                        bounds = {}
                        for col in cols:
                            q01 = df[col].quantile(0.01)
                            q99 = df[col].quantile(0.99)
                            bounds[col] = (q01, q99)
                            df[col] = df[col].clip(q01, q99)
                        self.transformers["outlier_bounds"] = bounds
                    else:
                        bounds = self.transformers.get("outlier_bounds", {})
                        for col in cols:
                            if col in bounds:
                                df[col] = df[col].clip(bounds[col][0], bounds[col][1])

                elif step == "encode_categorical":
                    cols = [c for c in step_config["columns"] if c in df.columns]
                    strategy = step_config["strategy"]
                    if strategy == "onehot":
                        if fit:
                            for col in cols:
                                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                            self.transformers[f"onehot_cols"] = cols
                        else:
                            for col in cols:
                                if col in df.columns:
                                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    elif strategy == "label":
                        for col in cols:
                            if col in df.columns:
                                if fit:
                                    le = LabelEncoder()
                                    df[col] = le.fit_transform(df[col].astype(str))
                                    self.transformers[f"le_{col}"] = le
                                else:
                                    le = self.transformers.get(f"le_{col}")
                                    if le:
                                        # Handle unseen labels
                                        known = set(le.classes_)
                                        df[col] = df[col].astype(str).map(
                                            lambda x: le.transform([x])[0] if x in known else -1
                                        )

                elif step == "scale_numeric":
                    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
                    if cols:
                        strategy = step_config["strategy"]
                        if fit:
                            if strategy == "standard":
                                scaler = StandardScaler()
                            elif strategy == "robust":
                                scaler = RobustScaler()
                            else:
                                scaler = MinMaxScaler()
                            df[cols] = scaler.fit_transform(df[cols])
                            self.transformers["scaler"] = scaler
                            self.transformers["scaler_cols"] = cols
                        else:
                            scaler = self.transformers.get("scaler")
                            scaler_cols = self.transformers.get("scaler_cols", [])
                            common_cols = [c for c in scaler_cols if c in df.columns]
                            if scaler and common_cols:
                                df[common_cols] = scaler.transform(df[common_cols])

                elif step == "encode_target":
                    if target_column in df.columns:
                        if fit:
                            le = LabelEncoder()
                            df[target_column] = le.fit_transform(df[target_column].astype(str))
                            self.label_encoder = le
                            self.transformers["target_encoder"] = le
                        else:
                            le = self.transformers.get("target_encoder")
                            if le:
                                df[target_column] = le.transform(df[target_column].astype(str))

                elif step == "feature_selection":
                    if fit and target_column in df.columns:
                        X_cols = [c for c in df.columns if c != target_column]
                        numeric_X = df[X_cols].select_dtypes(include=[np.number])
                        if len(numeric_X.columns) > step_config.get("k", 30):
                            target_series = df[target_column]
                            try:
                                if target_series.nunique() <= 20:
                                    mi = mutual_info_classif(numeric_X, target_series, random_state=42)
                                else:
                                    mi = mutual_info_regression(numeric_X, target_series, random_state=42)
                                k = step_config.get("k", 30)
                                top_indices = np.argsort(mi)[-k:]
                                selected = numeric_X.columns[top_indices].tolist()
                                # Keep non-numeric columns
                                non_numeric = [c for c in X_cols if c not in numeric_X.columns]
                                keep_cols = selected + non_numeric + [target_column]
                                df = df[[c for c in keep_cols if c in df.columns]]
                                self.transformers["selected_features"] = selected + non_numeric
                            except Exception as e:
                                logger.warning(f"Feature selection failed: {e}")

                self.steps_applied.append(step)

            except Exception as e:
                logger.error(f"Error in preprocessing step '{step}': {e}")

        # Split X and y
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            y = pd.Series(dtype=float)
            X = df

        if fit:
            self.steps_config = steps
            self.feature_columns = X.columns.tolist()
        elif self.feature_columns:
            # Align features with training columns
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]

        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform new inference data using the fitted pipeline."""
        df = df.copy()
        steps = getattr(self, "steps_config", [])
        if steps:
            X, y = self.apply_pipeline(df, steps, self.target_column or "", fit=False)
        else:
            X = df
            y = None

        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]

        return X, y

    def transform_prediction_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Transform single prediction input using fitted pipeline."""
        df = pd.DataFrame([data])
        X, _ = self.transform(df)
        return X

    def save(self, path: str):
        """Save pipeline to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load pipeline from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, PreprocessingPipeline):
                self.transformers = data.transformers
                self.feature_columns = data.feature_columns
                self.target_column = data.target_column
                self.label_encoder = data.label_encoder
                self.steps_applied = data.steps_applied
                self.steps_config = getattr(data, "steps_config", [])
            else:
                self.transformers = data.get("transformers", {})
                self.feature_columns = data.get("feature_columns", [])
                self.target_column = data.get("target_column")
                self.label_encoder = data.get("label_encoder")
                self.steps_applied = data.get("steps_applied", [])
                self.steps_config = data.get("steps_config", [])

