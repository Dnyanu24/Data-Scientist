"""Hybrid AI Algorithm Recommendation Engine."""
import logging
from typing import Dict, Any, List, Optional
from app.models.models import ProblemType

logger = logging.getLogger(__name__)

# ─── Algorithm Registry ─────────────────────────────────────────────────────

ALGORITHM_REGISTRY = {
    # Linear models
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "class_path": "sklearn.linear_model.LogisticRegression",
        "category": "linear",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["linear_relationships", "interpretability", "baseline"],
        "search_space": {
            "C": {"type": "loguniform", "low": 0.001, "high": 100},
            "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
            "solver": {"type": "categorical", "choices": ["liblinear", "saga"]},
            "max_iter": {"type": "int", "low": 100, "high": 1000},
        }
    },
    "ridge_regression": {
        "display_name": "Ridge Regression",
        "class_path": "sklearn.linear_model.Ridge",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["linear_relationships", "multicollinearity"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 0.001, "high": 100},
        }
    },
    "lasso_regression": {
        "display_name": "Lasso Regression",
        "class_path": "sklearn.linear_model.Lasso",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["feature_selection", "sparse_features"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 0.0001, "high": 10},
        }
    },
    "elastic_net": {
        "display_name": "Elastic Net",
        "class_path": "sklearn.linear_model.ElasticNet",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["feature_selection", "correlated_features"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 0.0001, "high": 10},
            "l1_ratio": {"type": "uniform", "low": 0.0, "high": 1.0},
        }
    },
    # Tree models
    "decision_tree": {
        "display_name": "Decision Tree",
        "class_path_clf": "sklearn.tree.DecisionTreeClassifier",
        "class_path_reg": "sklearn.tree.DecisionTreeRegressor",
        "category": "tree",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["interpretability", "non_linear"],
        "search_space": {
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "criterion": {"type": "categorical", "choices_clf": ["gini", "entropy"], "choices_reg": ["squared_error", "absolute_error"]},
        }
    },
    "random_forest": {
        "display_name": "Random Forest",
        "class_path_clf": "sklearn.ensemble.RandomForestClassifier",
        "class_path_reg": "sklearn.ensemble.RandomForestRegressor",
        "category": "ensemble",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["general_purpose", "robust", "feature_importance"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 5, "high": 50},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        }
    },
    "extra_trees": {
        "display_name": "Extra Trees",
        "class_path_clf": "sklearn.ensemble.ExtraTreesClassifier",
        "class_path_reg": "sklearn.ensemble.ExtraTreesRegressor",
        "category": "ensemble",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["general_purpose", "variance_reduction"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 5, "high": 50},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        }
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "class_path_clf": "sklearn.ensemble.GradientBoostingClassifier",
        "class_path_reg": "sklearn.ensemble.GradientBoostingRegressor",
        "category": "boosting",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "medium",
        "best_for": ["accuracy", "structured_data"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        }
    },
    "xgboost": {
        "display_name": "XGBoost",
        "class_path_clf": "xgboost.XGBClassifier",
        "class_path_reg": "xgboost.XGBRegressor",
        "category": "boosting",
        "library": "xgboost",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["tabular_data", "competitions", "accuracy"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 1000},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "loguniform", "low": 1e-8, "high": 10},
            "reg_lambda": {"type": "loguniform", "low": 1e-8, "high": 10},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
        }
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "class_path_clf": "lightgbm.LGBMClassifier",
        "class_path_reg": "lightgbm.LGBMRegressor",
        "category": "boosting",
        "library": "lightgbm",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": True,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["large_datasets", "speed", "accuracy"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 1000},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "num_leaves": {"type": "int", "low": 20, "high": 150},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "loguniform", "low": 1e-8, "high": 10},
            "reg_lambda": {"type": "loguniform", "low": 1e-8, "high": 10},
            "min_child_samples": {"type": "int", "low": 5, "high": 50},
        }
    },
    "catboost": {
        "display_name": "CatBoost",
        "class_path_clf": "catboost.CatBoostClassifier",
        "class_path_reg": "catboost.CatBoostRegressor",
        "category": "boosting",
        "library": "catboost",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": True,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["categorical_features", "robust", "accuracy"],
        "search_space": {
            "iterations": {"type": "int", "low": 100, "high": 1000},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "depth": {"type": "int", "low": 4, "high": 10},
            "l2_leaf_reg": {"type": "loguniform", "low": 1e-8, "high": 10},
            "border_count": {"type": "int", "low": 32, "high": 255},
        }
    },
    # SVM
    "svm": {
        "display_name": "Support Vector Machine",
        "class_path_clf": "sklearn.svm.SVC",
        "class_path_reg": "sklearn.svm.SVR",
        "category": "svm",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "medium",
        "memory_usage": "high",
        "scalability": "small",
        "best_for": ["small_datasets", "high_dimensional"],
        "search_space": {
            "C": {"type": "loguniform", "low": 0.01, "high": 100},
            "kernel": {"type": "categorical", "choices": ["rbf", "poly", "linear"]},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        }
    },
    # KNN
    "knn": {
        "display_name": "K-Nearest Neighbors",
        "class_path_clf": "sklearn.neighbors.KNeighborsClassifier",
        "class_path_reg": "sklearn.neighbors.KNeighborsRegressor",
        "category": "instance",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "slow",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["small_datasets", "non_parametric"],
        "search_space": {
            "n_neighbors": {"type": "int", "low": 3, "high": 30},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "metric": {"type": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]},
        }
    },
    # AdaBoost
    "adaboost": {
        "display_name": "AdaBoost",
        "class_path_clf": "sklearn.ensemble.AdaBoostClassifier",
        "class_path_reg": "sklearn.ensemble.AdaBoostRegressor",
        "category": "boosting",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "medium",
        "best_for": ["weak_learner_boosting"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 1.0},
        }
    },
    # Naive Bayes
    "naive_bayes": {
        "display_name": "Naive Bayes",
        "class_path": "sklearn.naive_bayes.GaussianNB",
        "category": "probabilistic",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["baseline", "text_classification"],
        "search_space": {
            "var_smoothing": {"type": "loguniform", "low": 1e-12, "high": 1e-6},
        }
    },
}


class AlgorithmRecommender:
    """Hybrid recommendation engine combining meta-learning, rules, and history."""

    def recommend(
        self,
        fingerprint: Dict[str, Any],
        problem_type: str,
        historical_results: Optional[List[Dict[str, Any]]] = None,
        user_requirements: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate ranked algorithm recommendations."""

        # Step 1: Filter feasible candidates
        candidates = self._filter_feasible(problem_type, fingerprint)

        # Step 2: Score each candidate
        scored = []
        for algo_name in candidates:
            algo_info = ALGORITHM_REGISTRY[algo_name]
            scores = {
                "rule_based": self._rule_based_score(algo_name, algo_info, fingerprint, problem_type),
                "compatibility": self._compatibility_score(algo_name, algo_info, fingerprint),
                "meta_learning": self._meta_learning_score(algo_name, historical_results),
                "historical": self._historical_score(algo_name, historical_results),
                "user_requirement": self._user_requirement_score(algo_name, algo_info, user_requirements),
            }

            # Weighted combination
            weights = {
                "rule_based": 0.25,
                "compatibility": 0.20,
                "meta_learning": 0.25,
                "historical": 0.20,
                "user_requirement": 0.10,
            }

            total_score = sum(scores[k] * weights[k] for k in weights)
            confidence = min(1.0, total_score * 1.2)  # scale confidence

            scored.append({
                "algorithm_name": algo_name,
                "display_name": algo_info["display_name"],
                "recommendation_score": round(total_score, 4),
                "confidence": round(confidence, 4),
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "category": algo_info["category"],
                "library": algo_info["library"],
                "training_speed": algo_info["training_speed"],
                "reasoning": self._generate_reasoning(algo_name, algo_info, scores, fingerprint),
            })

        # Step 3: Sort by score
        scored.sort(key=lambda x: x["recommendation_score"], reverse=True)

        # Step 4: Assign training stages
        for i, item in enumerate(scored):
            if i < 10:
                item["training_stage"] = "stage_1"
                item["training_priority"] = i + 1
            elif i < 16:
                item["training_stage"] = "stage_2"
                item["training_priority"] = i + 1
            else:
                item["training_stage"] = "stage_3"
                item["training_priority"] = i + 1

        return scored

    def _filter_feasible(self, problem_type: str, fingerprint: Dict[str, Any]) -> List[str]:
        """Filter algorithms that support the problem type and dataset."""
        feasible = []
        for name, info in ALGORITHM_REGISTRY.items():
            if problem_type not in info["supports"]:
                continue

            # Skip SVM/KNN for large datasets
            n_rows = fingerprint.get("num_rows", 0)
            if name in ["svm", "knn"] and n_rows > 50000:
                continue

            feasible.append(name)
        return feasible

    def _rule_based_score(self, algo_name: str, info: Dict, fingerprint: Dict, problem_type: str) -> float:
        """Score based on dataset characteristics matching algorithm strengths."""
        score = 0.5  # baseline

        n_rows = fingerprint.get("num_rows", 0)
        n_cols = fingerprint.get("num_columns", 0)
        n_numeric = fingerprint.get("num_numeric", 0)
        n_categorical = fingerprint.get("num_categorical", 0)
        missing_ratio = fingerprint.get("missing_ratio", 0)
        imbalance = fingerprint.get("class_imbalance_ratio")

        # Boosting models excel on tabular data
        if info["category"] == "boosting":
            score += 0.15

        # Tree-based models handle mixed types better
        if info["category"] in ["tree", "ensemble", "boosting"] and n_categorical > 0:
            score += 0.1

        # Linear models preferred for high-dimensional sparse data
        if info["category"] == "linear" and n_cols > n_rows * 0.5:
            score += 0.15

        # Missing data handling
        if missing_ratio > 0.05 and info["handles_missing"]:
            score += 0.1

        # Categorical handling
        if n_categorical > n_numeric and info["handles_categorical"]:
            score += 0.1

        # Large dataset bonuses
        if n_rows > 100000:
            if info["training_speed"] == "fast":
                score += 0.1
            if info["scalability"] == "large":
                score += 0.05
        elif n_rows < 1000:
            if info["category"] == "linear":
                score += 0.05
            if algo_name in ["svm", "knn"]:
                score += 0.1

        return min(1.0, max(0.0, score))

    def _compatibility_score(self, algo_name: str, info: Dict, fingerprint: Dict) -> float:
        """Score based on general dataset-algorithm compatibility."""
        score = 0.5
        n_rows = fingerprint.get("num_rows", 0)

        # Memory/scalability
        if info["scalability"] == "large":
            score += 0.1
        elif info["scalability"] == "small" and n_rows > 10000:
            score -= 0.2

        # Speed
        speed_bonus = {"fast": 0.1, "medium": 0.0, "slow": -0.1}
        score += speed_bonus.get(info["training_speed"], 0)

        return min(1.0, max(0.0, score))

    def _meta_learning_score(self, algo_name: str, historical_results: Optional[List[Dict]] = None) -> float:
        """Score based on similar dataset performance (meta-learning)."""
        if not historical_results:
            return 0.5

        scores = []
        for result in historical_results:
            if result.get("best_algorithm") == algo_name:
                scores.append(0.9)
            elif algo_name in [r.get("algo") for r in result.get("all_results", [])]:
                for r in result["all_results"]:
                    if r.get("algo") == algo_name:
                        scores.append(0.5 + 0.4 * (1 - r.get("rank", 5) / 10))
                        break
        return sum(scores) / max(len(scores), 1) if scores else 0.5

    def _historical_score(self, algo_name: str, historical_results: Optional[List[Dict]] = None) -> float:
        """Score based on historical win rate."""
        if not historical_results:
            return 0.5

        wins = sum(1 for r in historical_results if r.get("best_algorithm") == algo_name)
        total = len(historical_results)
        return min(1.0, 0.3 + 0.7 * (wins / max(total, 1)))

    def _user_requirement_score(self, algo_name: str, info: Dict,
                                 requirements: Optional[Dict] = None) -> float:
        """Score based on user requirements."""
        if not requirements:
            return 0.5

        score = 0.5
        if requirements.get("prefer_fast") and info["training_speed"] == "fast":
            score += 0.3
        if requirements.get("prefer_interpretable") and info["category"] in ["linear", "tree"]:
            score += 0.3
        if requirements.get("prefer_accuracy") and info["category"] == "boosting":
            score += 0.2
        return min(1.0, score)

    def _generate_reasoning(self, algo_name: str, info: Dict, scores: Dict, fingerprint: Dict) -> str:
        """Generate human-readable reasoning."""
        parts = []
        top_score_key = max(scores, key=scores.get)

        if top_score_key == "rule_based":
            parts.append(f"{info['display_name']} is well-suited for this dataset's characteristics")
        elif top_score_key == "meta_learning":
            parts.append(f"{info['display_name']} performed well on similar datasets")
        elif top_score_key == "historical":
            parts.append(f"{info['display_name']} has strong historical performance")
        elif top_score_key == "compatibility":
            parts.append(f"{info['display_name']} is highly compatible with this dataset")

        if info["handles_missing"] and fingerprint.get("missing_ratio", 0) > 0.05:
            parts.append("handles missing values natively")
        if info["handles_categorical"] and fingerprint.get("num_categorical", 0) > 0:
            parts.append("handles categorical features natively")
        if info["training_speed"] == "fast":
            parts.append("fast training speed")

        return ". ".join(parts) + "."


recommender = AlgorithmRecommender()
