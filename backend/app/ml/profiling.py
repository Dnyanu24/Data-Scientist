"""Dataset profiling, schema detection, and data quality analysis service."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from scipy import stats

logger = logging.getLogger(__name__)


class DataProfiler:
    """Performs comprehensive dataset analysis."""

    def profile_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Generate a full profile of the dataset."""
        profile = {
            "overview": self._get_overview(df),
            "columns": self._get_columns_info(df),
            "schema": self._detect_schema(df),
            "data_quality": self._assess_quality(df),
            "statistics": self._get_statistics(df),
            "correlations": self._get_correlations(df),
        }

        if target_column and target_column in df.columns:
            profile["target_analysis"] = self._analyze_target(df, target_column)

        return profile

    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset overview statistics."""
        memory_usage = df.memory_usage(deep=True).sum()
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "memory_usage_bytes": int(memory_usage),
            "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round(df.duplicated().mean() * 100, 2),
            "total_missing": int(df.isnull().sum().sum()),
            "total_missing_percentage": round(df.isnull().mean().mean() * 100, 2),
        }

    def _get_columns_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get detailed info per column."""
        columns = []
        for col in df.columns:
            series = df[col]
            info = {
                "name": col,
                "dtype": str(series.dtype),
                "detected_type": self._detect_column_type(series),
                "count": int(series.count()),
                "missing": int(series.isnull().sum()),
                "missing_pct": round(series.isnull().mean() * 100, 2),
                "unique": int(series.nunique()),
                "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 2),
            }

            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                info.update({
                    "mean": round(float(desc.get("mean", 0)), 4) if not pd.isna(desc.get("mean")) else None,
                    "std": round(float(desc.get("std", 0)), 4) if not pd.isna(desc.get("std")) else None,
                    "min": float(desc.get("min", 0)) if not pd.isna(desc.get("min")) else None,
                    "max": float(desc.get("max", 0)) if not pd.isna(desc.get("max")) else None,
                    "median": round(float(series.median()), 4) if not series.isnull().all() else None,
                    "skewness": round(float(series.skew()), 4) if not series.isnull().all() else None,
                    "kurtosis": round(float(series.kurtosis()), 4) if not series.isnull().all() else None,
                    "zeros": int((series == 0).sum()),
                    "negative": int((series < 0).sum()),
                })
            else:
                top_values = series.value_counts().head(10).to_dict()
                info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                info["sample_values"] = [str(v) for v in series.dropna().head(5).tolist()]

            columns.append(info)
        return columns

    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect semantic type of column."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 2:
                return "binary"
            if series.nunique() <= 20 and series.nunique() / max(len(series), 1) < 0.05:
                return "categorical_numeric"
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            return "float"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Try parsing as datetime
        try:
            sample = series.dropna().head(100)
            if len(sample) > 0:
                pd.to_datetime(sample, infer_datetime_format=True)
                return "datetime"
        except (ValueError, TypeError):
            pass

        # Check for text vs categorical
        if series.nunique() / max(len(series), 1) > 0.5:
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return "text"
            return "high_cardinality"

        return "categorical"

    def _detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overall schema."""
        type_counts = {}
        column_types = {}
        for col in df.columns:
            ctype = self._detect_column_type(df[col])
            column_types[col] = ctype
            base_type = ctype.split("_")[0] if "_" in ctype else ctype
            type_counts[base_type] = type_counts.get(base_type, 0) + 1

        return {
            "column_types": column_types,
            "type_distribution": type_counts,
            "num_numeric": sum(1 for t in column_types.values() if t in ["integer", "float", "binary", "categorical_numeric"]),
            "num_categorical": sum(1 for t in column_types.values() if t in ["categorical", "boolean"]),
            "num_text": sum(1 for t in column_types.values() if t in ["text", "high_cardinality"]),
            "num_datetime": sum(1 for t in column_types.values() if t == "datetime"),
        }

    def _assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality."""
        issues = []
        scores = {}

        # Completeness
        completeness = 1 - df.isnull().mean().mean()
        scores["completeness"] = round(completeness * 100, 1)

        # Uniqueness
        dup_ratio = df.duplicated().mean()
        scores["uniqueness"] = round((1 - dup_ratio) * 100, 1)
        if dup_ratio > 0.05:
            issues.append({
                "type": "duplicates",
                "severity": "warning",
                "message": f"{round(dup_ratio*100, 1)}% duplicate rows found"
            })

        # Missing values check per column
        high_missing = []
        for col in df.columns:
            miss_pct = df[col].isnull().mean()
            if miss_pct > 0.5:
                high_missing.append(col)
                issues.append({
                    "type": "high_missing",
                    "severity": "critical",
                    "column": col,
                    "message": f"Column '{col}' has {round(miss_pct*100, 1)}% missing values"
                })
            elif miss_pct > 0.1:
                issues.append({
                    "type": "missing",
                    "severity": "warning",
                    "column": col,
                    "message": f"Column '{col}' has {round(miss_pct*100, 1)}% missing values"
                })

        # Outliers in numeric columns
        outlier_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_mask = (df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)
                outlier_pct = outlier_mask.mean()
                if outlier_pct > 0.05:
                    outlier_cols.append(col)
                    issues.append({
                        "type": "outliers",
                        "severity": "info",
                        "column": col,
                        "message": f"Column '{col}' has {round(outlier_pct*100, 1)}% extreme outliers"
                    })

        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues.append({
                    "type": "constant",
                    "severity": "warning",
                    "column": col,
                    "message": f"Column '{col}' is constant (single value)"
                })

        # Overall score
        issue_penalty = len([i for i in issues if i["severity"] == "critical"]) * 10
        issue_penalty += len([i for i in issues if i["severity"] == "warning"]) * 3
        overall_score = max(0, min(100, round(
            (scores["completeness"] * 0.4 + scores["uniqueness"] * 0.3 + max(0, 100 - issue_penalty) * 0.3), 1
        )))
        scores["overall"] = overall_score

        return {
            "scores": scores,
            "issues": issues,
            "high_missing_columns": high_missing,
            "outlier_columns": outlier_cols,
            "recommendations": self._quality_recommendations(issues, df),
        }

    def _quality_recommendations(self, issues: List, df: pd.DataFrame) -> List[str]:
        """Generate data quality recommendations."""
        recs = []
        issue_types = {i["type"] for i in issues}
        if "high_missing" in issue_types:
            recs.append("Consider dropping columns with >50% missing data")
        if "missing" in issue_types:
            recs.append("Use imputation (median for numeric, mode for categorical)")
        if "duplicates" in issue_types:
            recs.append("Review and remove duplicate rows")
        if "outliers" in issue_types:
            recs.append("Consider outlier treatment (clipping or removal)")
        if "constant" in issue_types:
            recs.append("Remove constant columns (zero information)")
        return recs

    def _get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get aggregate statistics."""
        numeric_df = df.select_dtypes(include=[np.number])
        result = {
            "numeric_summary": {},
            "categorical_summary": {},
        }
        if len(numeric_df.columns) > 0:
            desc = numeric_df.describe().to_dict()
            result["numeric_summary"] = {
                col: {k: round(float(v), 4) if not pd.isna(v) else None for k, v in stats.items()}
                for col, stats in desc.items()
            }

        cat_df = df.select_dtypes(include=["object", "category"])
        for col in cat_df.columns:
            result["categorical_summary"][col] = {
                "unique": int(cat_df[col].nunique()),
                "top": str(cat_df[col].mode().iloc[0]) if len(cat_df[col].mode()) > 0 else None,
                "freq": int(cat_df[col].value_counts().iloc[0]) if len(cat_df[col].value_counts()) > 0 else 0,
            }

        return result

    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation matrix for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {"matrix": {}, "high_correlations": []}

        corr = numeric_df.corr()
        # Find high correlations
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8 and not pd.isna(val):
                    high_corr.append({
                        "feature_1": corr.columns[i],
                        "feature_2": corr.columns[j],
                        "correlation": round(float(val), 4),
                    })

        # Limit correlation matrix to top 30 columns for response size
        cols = numeric_df.columns[:30].tolist()
        matrix = {}
        for col in cols:
            matrix[col] = {
                c: round(float(corr.loc[col, c]), 4) if not pd.isna(corr.loc[col, c]) else 0
                for c in cols
            }

        return {
            "matrix": matrix,
            "columns": cols,
            "high_correlations": high_corr,
            "avg_correlation": round(float(corr.values[np.triu_indices_from(corr.values, k=1)].mean()), 4)
            if len(corr) > 1 else 0,
        }

    def _analyze_target(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze the target column."""
        target = df[target_column]
        analysis = {
            "column": target_column,
            "dtype": str(target.dtype),
            "unique_values": int(target.nunique()),
            "missing": int(target.isnull().sum()),
            "missing_pct": round(target.isnull().mean() * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
            analysis["problem_type"] = "regression"
            analysis["distribution"] = {
                "mean": round(float(target.mean()), 4),
                "std": round(float(target.std()), 4),
                "min": float(target.min()),
                "max": float(target.max()),
                "skewness": round(float(target.skew()), 4),
            }
        else:
            n_unique = target.nunique()
            analysis["problem_type"] = "binary_classification" if n_unique == 2 else "multiclass_classification"
            vc = target.value_counts()
            analysis["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            if len(vc) > 1:
                analysis["class_imbalance_ratio"] = round(float(vc.max() / vc.min()), 2)
            analysis["is_imbalanced"] = analysis.get("class_imbalance_ratio", 1) > 3

        return analysis

    def detect_target_column(self, df: pd.DataFrame, business_goal: Optional[str] = None) -> Optional[str]:
        """Auto-detect the most likely target column."""
        candidates = []

        for col in df.columns:
            score = 0
            series = df[col]

            # Skip ID-like columns
            if col.lower() in ["id", "index", "row_id", "row_num"]:
                continue
            if series.nunique() == len(df):
                continue

            # Prefer columns at the end (common convention)
            position_score = df.columns.get_loc(col) / max(len(df.columns) - 1, 1)
            score += position_score * 0.2

            # Prefer columns named target/label/class/y
            target_keywords = ["target", "label", "class", "output", "y", "result", "outcome", "status", "category"]
            if any(kw in col.lower() for kw in target_keywords):
                score += 0.4

            # Prefer categorical with few classes
            if series.nunique() <= 10:
                score += 0.2
            elif series.nunique() <= 50:
                score += 0.1

            # Penalize high cardinality text
            if series.dtype == "object" and series.nunique() > 100:
                score -= 0.3

            # Prefer columns with no missing values
            if series.isnull().sum() == 0:
                score += 0.1

            candidates.append((col, score))

        if not candidates:
            return df.columns[-1] if len(df.columns) > 0 else None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def detect_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Detect problem type from target column."""
        target = df[target_column]
        if pd.api.types.is_numeric_dtype(target):
            if target.nunique() <= 2:
                return "binary_classification"
            elif target.nunique() <= 20 and target.nunique() / len(df) < 0.05:
                return "multiclass_classification"
            else:
                return "regression"
        else:
            if target.nunique() <= 2:
                return "binary_classification"
            return "multiclass_classification"


data_profiler = DataProfiler()
