"""Intelligent Training & Hyperparameter Optimization Engine."""
import time
import logging
import importlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, log_loss
)
import optuna
import pickle
import os
import json

from app.ml.recommendation import ALGORITHM_REGISTRY

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrainingEngine:
    """Progressive training with Optuna hyperparameter optimization."""

    def __init__(self, model_store_dir: str = "./model_store"):
        self.model_store_dir = model_store_dir

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        algorithm_name: str,
        problem_type: str,
        n_trials: int = 15,
        training_stage: str = "stage_1",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Train a single model with Optuna optimization."""
        start_time = time.time()
        algo_info = ALGORITHM_REGISTRY.get(algorithm_name)

        if not algo_info:
            return {"status": "failed", "error": f"Unknown algorithm: {algorithm_name}"}

        try:
            # Create Optuna study
            is_classification = problem_type in ["binary_classification", "multiclass_classification"]
            direction = "maximize" if is_classification else "minimize"
            metric_name = "f1_weighted" if is_classification else "neg_mse"

            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
            )

            def objective(trial):
                params = self._suggest_params(trial, algorithm_name, algo_info, problem_type, training_stage)
                model = self._create_model(algorithm_name, algo_info, problem_type, params)

                try:
                    model.fit(X_train, y_train)
                    if is_classification:
                        y_pred = model.predict(X_val)
                        score = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                    else:
                        y_pred = model.predict(X_val)
                        score = -mean_squared_error(y_val, y_pred)
                    return score
                except Exception as e:
                    logger.warning(f"Trial failed for {algorithm_name}: {e}")
                    return float("-inf") if is_classification else float("inf")

            # Run optimization
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            # Train final model with best params
            best_params = study.best_params
            final_model = self._create_model(algorithm_name, algo_info, problem_type, best_params)
            final_model.fit(X_train, y_train)

            # Evaluate
            metrics = self._evaluate_model(final_model, X_val, y_val, problem_type)

            # Feature importance
            feature_importance = self._get_feature_importance(final_model, X_train.columns.tolist())

            training_time = time.time() - start_time

            return {
                "status": "completed",
                "algorithm_name": algorithm_name,
                "best_params": best_params,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "training_time": round(training_time, 2),
                "n_trials": n_trials,
                "best_trial_value": float(study.best_value),
                "training_stage": training_stage,
                "model": final_model,
            }

        except Exception as e:
            logger.error(f"Training failed for {algorithm_name}: {e}")
            return {
                "status": "failed",
                "algorithm_name": algorithm_name,
                "error": str(e),
                "training_time": time.time() - start_time,
            }

    def _suggest_params(self, trial, algo_name: str, algo_info: Dict,
                        problem_type: str, stage: str) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        search_space = algo_info.get("search_space", {})
        params = {}

        for param_name, config in search_space.items():
            ptype = config["type"]

            # Handle classification/regression specific choices
            if param_name == "criterion" and "choices_clf" in config:
                choices = config["choices_clf"] if "classification" in problem_type else config["choices_reg"]
                params[param_name] = trial.suggest_categorical(param_name, choices)
                continue

            if ptype == "int":
                low, high = config["low"], config["high"]
                # Narrow search space for early stages
                if stage == "stage_1":
                    high = low + (high - low) // 2
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif ptype == "uniform":
                params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
            elif ptype == "loguniform":
                params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=True)
            elif ptype == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, config["choices"])

        # Add fixed params
        if algo_name == "xgboost":
            params["use_label_encoder"] = False
            params["eval_metric"] = "logloss" if "classification" in problem_type else "rmse"
            params["verbosity"] = 0
        elif algo_name == "lightgbm":
            params["verbose"] = -1
            params["force_row_wise"] = True
        elif algo_name == "catboost":
            params["verbose"] = 0
            params["allow_writing_files"] = False
        elif algo_name == "logistic_regression":
            if params.get("penalty") == "l1":
                params["solver"] = "liblinear"

        return params

    def _create_model(self, algo_name: str, algo_info: Dict,
                      problem_type: str, params: Dict) -> Any:
        """Create model instance from algorithm name and params."""
        is_clf = "classification" in problem_type

        # Determine class path
        if is_clf:
            class_path = algo_info.get("class_path_clf", algo_info.get("class_path"))
        else:
            class_path = algo_info.get("class_path_reg", algo_info.get("class_path"))

        # Import class
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Handle special cases
        clean_params = {k: v for k, v in params.items()}

        # SVC needs probability for AUC
        if algo_name == "svm" and is_clf:
            clean_params["probability"] = True

        return model_class(**clean_params)

    def _evaluate_model(self, model, X_val, y_val, problem_type: str) -> Dict[str, Any]:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X_val)
        metrics = {}

        if "classification" in problem_type:
            metrics["accuracy"] = round(float(accuracy_score(y_val, y_pred)), 4)
            avg = "binary" if len(np.unique(y_val)) == 2 else "weighted"
            metrics["f1_score"] = round(float(f1_score(y_val, y_pred, average=avg, zero_division=0)), 4)
            metrics["precision"] = round(float(precision_score(y_val, y_pred, average=avg, zero_division=0)), 4)
            metrics["recall"] = round(float(recall_score(y_val, y_pred, average=avg, zero_division=0)), 4)

            # AUC
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_val)
                    if len(np.unique(y_val)) == 2:
                        metrics["auc_roc"] = round(float(roc_auc_score(y_val, y_proba[:, 1])), 4)
                    else:
                        metrics["auc_roc"] = round(float(roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")), 4)
            except Exception:
                pass

            # Log loss
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_val)
                    metrics["log_loss"] = round(float(log_loss(y_val, y_proba)), 4)
            except Exception:
                pass
        else:
            metrics["mse"] = round(float(mean_squared_error(y_val, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4)
            metrics["mae"] = round(float(mean_absolute_error(y_val, y_pred)), 4)
            metrics["r2"] = round(float(r2_score(y_val, y_pred)), 4)

        return metrics

    def _get_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)

        if importance is not None:
            n = min(len(importance), len(feature_names))
            imp_dict = {feature_names[i]: round(float(importance[i]), 6) for i in range(n)}
            # Sort by importance
            imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
            return imp_dict

        return None

    def save_model(self, model, project_id: int, model_name: str) -> str:
        """Save trained model to disk."""
        model_dir = os.path.join(self.model_store_dir, str(project_id))
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return model_path

    def load_model(self, model_path: str):
        """Load model from disk."""
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def create_ensemble(self, models: List[Any], ensemble_type: str,
                        X_train, y_train, X_val, y_val,
                        problem_type: str) -> Dict[str, Any]:
        """Create ensemble from multiple models."""
        from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
        start_time = time.time()
        is_clf = "classification" in problem_type

        try:
            estimators = [(f"model_{i}", m) for i, m in enumerate(models)]

            if ensemble_type == "voting":
                if is_clf:
                    ensemble = VotingClassifier(estimators=estimators, voting="soft")
                else:
                    ensemble = VotingRegressor(estimators=estimators)
            elif ensemble_type == "weighted_voting":
                # Use validation scores as weights
                weights = []
                for _, m in estimators:
                    y_pred = m.predict(X_val)
                    if is_clf:
                        w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                    else:
                        w = max(0.01, r2_score(y_val, y_pred))
                    weights.append(w)
                if is_clf:
                    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
                else:
                    ensemble = VotingRegressor(estimators=estimators, weights=weights)
            elif ensemble_type == "stacking":
                if is_clf:
                    from sklearn.linear_model import LogisticRegression
                    ensemble = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(max_iter=1000),
                        cv=3,
                    )
                else:
                    from sklearn.linear_model import Ridge
                    ensemble = StackingRegressor(
                        estimators=estimators,
                        final_estimator=Ridge(),
                        cv=3,
                    )
            else:
                return {"status": "failed", "error": f"Unknown ensemble type: {ensemble_type}"}

            ensemble.fit(X_train, y_train)
            metrics = self._evaluate_model(ensemble, X_val, y_val, problem_type)

            return {
                "status": "completed",
                "algorithm_name": f"ensemble_{ensemble_type}",
                "ensemble_type": ensemble_type,
                "metrics": metrics,
                "training_time": round(time.time() - start_time, 2),
                "model": ensemble,
                "n_members": len(models),
            }

        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return {"status": "failed", "error": str(e)}


training_engine = TrainingEngine()
