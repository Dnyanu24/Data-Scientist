"""Drift monitoring service."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor for data and concept drift."""

    def detect_data_drift(self, reference_data: pd.DataFrame,
                          current_data: pd.DataFrame,
                          feature_columns: List[str]) -> Dict[str, Any]:
        """Detect data drift using statistical tests."""
        results = {"features": {}, "overall_drift": False, "drift_score": 0.0}
        drift_count = 0

        for col in feature_columns:
            if col not in reference_data.columns or col not in current_data.columns:
                continue

            ref = reference_data[col].dropna()
            cur = current_data[col].dropna()

            if len(ref) == 0 or len(cur) == 0:
                continue

            if pd.api.types.is_numeric_dtype(ref):
                # KS test for numeric
                statistic, p_value = stats.ks_2samp(ref, cur)
                is_drifted = p_value < 0.05
                results["features"][col] = {
                    "test": "ks_test",
                    "statistic": round(float(statistic), 4),
                    "p_value": round(float(p_value), 4),
                    "is_drifted": is_drifted,
                    "drift_type": "data_drift",
                }
            else:
                # Chi-squared for categorical
                ref_counts = ref.value_counts(normalize=True)
                cur_counts = cur.value_counts(normalize=True)
                all_cats = set(ref_counts.index) | set(cur_counts.index)
                ref_freq = [ref_counts.get(c, 0) for c in all_cats]
                cur_freq = [cur_counts.get(c, 0) for c in all_cats]
                if sum(cur_freq) > 0:
                    try:
                        statistic, p_value = stats.chisquare(
                            [max(x, 1e-10) for x in cur_freq],
                            [max(x, 1e-10) for x in ref_freq]
                        )
                        is_drifted = p_value < 0.05
                    except Exception:
                        statistic, p_value, is_drifted = 0, 1, False
                else:
                    statistic, p_value, is_drifted = 0, 1, False

                results["features"][col] = {
                    "test": "chi_squared",
                    "statistic": round(float(statistic), 4),
                    "p_value": round(float(p_value), 4),
                    "is_drifted": is_drifted,
                    "drift_type": "data_drift",
                }

            if is_drifted:
                drift_count += 1

        total_features = len(results["features"])
        results["drift_score"] = round(drift_count / max(total_features, 1), 4)
        results["overall_drift"] = results["drift_score"] > 0.3
        results["drifted_features"] = [
            col for col, info in results["features"].items() if info["is_drifted"]
        ]

        return results

    def detect_prediction_drift(self, reference_predictions: List,
                                 current_predictions: List) -> Dict[str, Any]:
        """Detect drift in prediction distribution."""
        ref = np.array(reference_predictions)
        cur = np.array(current_predictions)

        if len(ref) == 0 or len(cur) == 0:
            return {"drift_score": 0, "is_drifted": False}

        # Check if numeric
        try:
            ref = ref.astype(float)
            cur = cur.astype(float)
            statistic, p_value = stats.ks_2samp(ref, cur)
            return {
                "test": "ks_test",
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "is_drifted": p_value < 0.05,
                "drift_score": round(float(statistic), 4),
                "drift_type": "prediction_drift",
            }
        except (ValueError, TypeError):
            # Categorical predictions
            from collections import Counter
            ref_dist = Counter(ref.tolist())
            cur_dist = Counter(cur.tolist())
            all_keys = set(ref_dist.keys()) | set(cur_dist.keys())

            total_ref = sum(ref_dist.values())
            total_cur = sum(cur_dist.values())

            diff = sum(
                abs(ref_dist.get(k, 0) / max(total_ref, 1) - cur_dist.get(k, 0) / max(total_cur, 1))
                for k in all_keys
            ) / 2

            return {
                "test": "distribution_diff",
                "drift_score": round(float(diff), 4),
                "is_drifted": diff > 0.1,
                "drift_type": "prediction_drift",
            }


drift_monitor = DriftMonitor()
