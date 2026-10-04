"""Dataset fingerprinting and similarity search service."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DatasetFingerprinter:
    """Generate dataset fingerprints and compute similarity."""

    EMBEDDING_DIM = 128

    def generate_fingerprint(self, df: pd.DataFrame, target_column: Optional[str] = None,
                              problem_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset fingerprint."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # Remove target from feature analysis
        feature_numeric = [c for c in numeric_cols if c != target_column]
        feature_categorical = [c for c in categorical_cols if c != target_column]

        fingerprint = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "num_numeric": len(feature_numeric),
            "num_categorical": len(feature_categorical),
            "num_text": sum(1 for c in categorical_cols if df[c].nunique() / max(len(df), 1) > 0.5),
            "num_datetime": len(datetime_cols),
            "missing_ratio": round(float(df.isnull().mean().mean()), 4),
            "feature_to_sample_ratio": round(len(df.columns) / max(len(df), 1), 6),
            "problem_type": problem_type,
        }

        # Numeric feature statistics
        if feature_numeric:
            numeric_data = df[feature_numeric]
            fingerprint["numeric_skewness_mean"] = round(float(numeric_data.skew().mean()), 4)
            fingerprint["numeric_kurtosis_mean"] = round(float(numeric_data.kurtosis().mean()), 4)

            # Outlier ratio
            outlier_count = 0
            total_count = 0
            for col in feature_numeric:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
                    outlier_count += outliers
                    total_count += len(df[col].dropna())
            fingerprint["outlier_ratio"] = round(outlier_count / max(total_count, 1), 4)

            # Correlation stats
            if len(feature_numeric) > 1:
                corr = numeric_data.corr().values
                upper = corr[np.triu_indices_from(corr, k=1)]
                fingerprint["avg_correlation"] = round(float(np.nanmean(np.abs(upper))), 4)
                fingerprint["max_correlation"] = round(float(np.nanmax(np.abs(upper))), 4)
            else:
                fingerprint["avg_correlation"] = 0.0
                fingerprint["max_correlation"] = 0.0
        else:
            fingerprint.update({
                "numeric_skewness_mean": 0.0,
                "numeric_kurtosis_mean": 0.0,
                "outlier_ratio": 0.0,
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
            })

        # Categorical stats
        if feature_categorical:
            avg_cardinality = np.mean([df[c].nunique() for c in feature_categorical])
            fingerprint["avg_categorical_cardinality"] = round(float(avg_cardinality), 2)
        else:
            fingerprint["avg_categorical_cardinality"] = 0.0

        # Class imbalance
        if target_column and target_column in df.columns:
            target = df[target_column]
            if target.nunique() <= 50:  # classification
                vc = target.value_counts()
                if len(vc) > 1:
                    fingerprint["class_imbalance_ratio"] = round(float(vc.max() / vc.min()), 2)
                    fingerprint["num_classes"] = int(target.nunique())
                else:
                    fingerprint["class_imbalance_ratio"] = 1.0
                    fingerprint["num_classes"] = 1
            else:
                fingerprint["class_imbalance_ratio"] = None
                fingerprint["num_classes"] = None
        else:
            fingerprint["class_imbalance_ratio"] = None
            fingerprint["num_classes"] = None

        # Size category
        n = len(df)
        if n < 1000:
            fingerprint["size_category"] = "small"
        elif n < 100000:
            fingerprint["size_category"] = "medium"
        else:
            fingerprint["size_category"] = "large"

        return fingerprint

    def fingerprint_to_embedding(self, fingerprint: Dict[str, Any]) -> List[float]:
        """Convert fingerprint to fixed-size embedding vector."""
        features = []

        # Numerical features (normalized)
        features.append(np.log1p(fingerprint.get("num_rows", 0)) / 20.0)
        features.append(np.log1p(fingerprint.get("num_columns", 0)) / 10.0)
        features.append(fingerprint.get("num_numeric", 0) / max(fingerprint.get("num_columns", 1), 1))
        features.append(fingerprint.get("num_categorical", 0) / max(fingerprint.get("num_columns", 1), 1))
        features.append(fingerprint.get("num_text", 0) / max(fingerprint.get("num_columns", 1), 1))
        features.append(fingerprint.get("missing_ratio", 0))
        features.append(fingerprint.get("feature_to_sample_ratio", 0))
        features.append(min(fingerprint.get("outlier_ratio", 0), 1.0))
        features.append(fingerprint.get("avg_correlation", 0))
        features.append(fingerprint.get("max_correlation", 0))
        features.append(min(fingerprint.get("numeric_skewness_mean", 0), 10.0) / 10.0)
        features.append(min(fingerprint.get("avg_categorical_cardinality", 0), 100.0) / 100.0)

        # Class imbalance
        cir = fingerprint.get("class_imbalance_ratio")
        features.append(min(cir, 100.0) / 100.0 if cir is not None else 0.5)

        # Problem type encoding
        pt = fingerprint.get("problem_type", "")
        features.append(1.0 if pt == "binary_classification" else 0.0)
        features.append(1.0 if pt == "multiclass_classification" else 0.0)
        features.append(1.0 if pt == "regression" else 0.0)

        # Size category encoding
        sc = fingerprint.get("size_category", "medium")
        features.append(1.0 if sc == "small" else 0.0)
        features.append(1.0 if sc == "medium" else 0.0)
        features.append(1.0 if sc == "large" else 0.0)

        # Pad to EMBEDDING_DIM with hashed features
        base_features = np.array(features, dtype=np.float32)

        # Create repeating pattern to fill embedding
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding[:len(base_features)] = base_features

        # Fill remaining with derived features
        for i in range(len(base_features), self.EMBEDDING_DIM):
            idx1 = i % len(base_features)
            idx2 = (i * 7 + 3) % len(base_features)
            embedding[i] = np.sin(base_features[idx1] * np.pi + base_features[idx2])

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a = np.array(embedding1)
        b = np.array(embedding2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


fingerprinter = DatasetFingerprinter()
