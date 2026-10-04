"""Model evaluation and explainability service."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and explainability."""

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                 problem_type: str) -> Dict[str, Any]:
        """Full evaluation with all metrics."""
        y_pred = model.predict(X_test)
        result = {"predictions": y_pred.tolist()[:100]}  # store sample

        if "classification" in problem_type:
            result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            result["classification_report"] = report

            # Per-class metrics
            classes = sorted(y_test.unique())
            result["per_class"] = {}
            for cls in classes:
                cls_str = str(cls)
                if cls_str in report:
                    result["per_class"][cls_str] = report[cls_str]

            # Probability-based metrics
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                result["prediction_probabilities"] = y_proba[:100].tolist()

        else:
            result["residuals"] = {
                "mean": round(float(np.mean(y_test - y_pred)), 4),
                "std": round(float(np.std(y_test - y_pred)), 4),
                "max_error": round(float(np.max(np.abs(y_test - y_pred))), 4),
            }
            result["actual_vs_predicted"] = {
                "actual": y_test.tolist()[:100],
                "predicted": y_pred.tolist()[:100],
            }

        return result

    def explain_model(self, model, X_sample: pd.DataFrame, feature_names: List[str],
                      problem_type: str) -> Dict[str, Any]:
        """Generate model explanations."""
        explanations = {}

        # Feature importance (from model)
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            n = min(len(imp), len(feature_names))
            explanations["feature_importance"] = {
                feature_names[i]: round(float(imp[i]), 6) for i in range(n)
            }
            # Top features
            sorted_imp = sorted(explanations["feature_importance"].items(), key=lambda x: x[1], reverse=True)
            explanations["top_features"] = [{"feature": k, "importance": v} for k, v in sorted_imp[:20]]
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)
            n = min(len(importance), len(feature_names))
            explanations["feature_importance"] = {
                feature_names[i]: round(float(importance[i]), 6) for i in range(n)
            }
            sorted_imp = sorted(explanations["feature_importance"].items(), key=lambda x: x[1], reverse=True)
            explanations["top_features"] = [{"feature": k, "importance": v} for k, v in sorted_imp[:20]]

        # Try SHAP (lightweight - only for small samples)
        try:
            import shap
            sample_size = min(100, len(X_sample))
            X_shap = X_sample.iloc[:sample_size]

            if hasattr(model, "predict_proba"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                mean_shap = np.abs(shap_values).mean(axis=0)
                n = min(len(mean_shap), len(feature_names))
                explanations["shap_importance"] = {
                    feature_names[i]: round(float(mean_shap[i]), 6) for i in range(n)
                }
            elif hasattr(model, "predict"):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap)
                    mean_shap = np.abs(shap_values).mean(axis=0)
                    n = min(len(mean_shap), len(feature_names))
                    explanations["shap_importance"] = {
                        feature_names[i]: round(float(mean_shap[i]), 6) for i in range(n)
                    }
                except Exception:
                    pass
        except ImportError:
            logger.info("SHAP not available")
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

        return explanations


evaluator = ModelEvaluator()
