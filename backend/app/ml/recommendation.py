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
    # Histogram Gradient Boosting
    "hist_gradient_boosting": {
        "display_name": "Histogram Gradient Boosting",
        "class_path_clf": "sklearn.ensemble.HistGradientBoostingClassifier",
        "class_path_reg": "sklearn.ensemble.HistGradientBoostingRegressor",
        "category": "boosting",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": True,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["large_datasets", "speed", "accuracy", "native_missing_values"],
        "search_space": {
            "max_iter": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 127},
            "min_samples_leaf": {"type": "int", "low": 10, "high": 100},
            "l2_regularization": {"type": "loguniform", "low": 1e-6, "high": 10.0},
        }
    },
    # Multi-Layer Perceptron (Neural Network)
    "mlp": {
        "display_name": "Multi-Layer Perceptron (Neural Net)",
        "class_path_clf": "sklearn.neural_network.MLPClassifier",
        "class_path_reg": "sklearn.neural_network.MLPRegressor",
        "category": "neural_network",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "medium",
        "best_for": ["complex_patterns", "non_linear", "deep_representations"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "learning_rate_init": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "max_iter": {"type": "int", "low": 100, "high": 400},
        }
    },
    # Linear Regression (OLS)
    "linear_regression": {
        "display_name": "Linear Regression (OLS)",
        "class_path": "sklearn.linear_model.LinearRegression",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["baseline", "interpretability", "linear_relationships"],
        "search_space": {
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
        }
    },
    # Huber Robust Regressor
    "huber_regressor": {
        "display_name": "Huber Robust Regressor",
        "class_path": "sklearn.linear_model.HuberRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["outlier_resilient", "robust_linear"],
        "search_space": {
            "epsilon": {"type": "uniform", "low": 1.1, "high": 2.0},
            "alpha": {"type": "loguniform", "low": 1e-4, "high": 10.0},
        }
    },
    # SGD Classifier
    "sgd_classifier": {
        "display_name": "SGD Classifier",
        "class_path": "sklearn.linear_model.SGDClassifier",
        "category": "linear",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["massive_datasets", "online_learning", "streaming"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "penalty": {"type": "categorical", "choices": ["l2", "l1", "elasticnet"]},
        }
    },
    # SGD Regressor
    "sgd_regressor": {
        "display_name": "SGD Regressor",
        "class_path": "sklearn.linear_model.SGDRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["massive_datasets", "online_regression"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "penalty": {"type": "categorical", "choices": ["l2", "l1", "elasticnet"]},
        }
    },
    # Linear Support Vector Machine
    "linear_svm": {
        "display_name": "Linear SVM",
        "class_path_clf": "sklearn.svm.LinearSVC",
        "class_path_reg": "sklearn.svm.LinearSVR",
        "category": "svm",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["text_classification", "sparse_high_dimensional"],
        "search_space": {
            "C": {"type": "loguniform", "low": 0.01, "high": 100.0},
        }
    },
    # Multinomial Naive Bayes
    "multinomial_nb": {
        "display_name": "Multinomial Naive Bayes",
        "class_path": "sklearn.naive_bayes.MultinomialNB",
        "category": "probabilistic",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["text_counts", "nlp", "document_classification"],
        "search_space": {
            "alpha": {"type": "uniform", "low": 0.01, "high": 2.0},
        }
    },
    # Bernoulli Naive Bayes
    "bernoulli_nb": {
        "display_name": "Bernoulli Naive Bayes",
        "class_path": "sklearn.naive_bayes.BernoulliNB",
        "category": "probabilistic",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["binary_features", "text_boolean_features"],
        "search_space": {
            "alpha": {"type": "uniform", "low": 0.01, "high": 2.0},
        }
    },
    # Linear Discriminant Analysis
    "lda": {
        "display_name": "Linear Discriminant Analysis (LDA)",
        "class_path": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        "category": "probabilistic",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["gaussian_distributions", "dimensionality_reduction", "classification"],
        "search_space": {
            "solver": {"type": "categorical", "choices": ["svd", "lsqr"]},
        }
    },
    # Quadratic Discriminant Analysis
    "qda": {
        "display_name": "Quadratic Discriminant Analysis (QDA)",
        "class_path": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        "category": "probabilistic",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["quadratic_boundaries", "differing_class_covariances"],
        "search_space": {
            "reg_param": {"type": "uniform", "low": 0.0, "high": 1.0},
        }
    },
    # Gaussian Process
    "gaussian_process": {
        "display_name": "Gaussian Process",
        "class_path_clf": "sklearn.gaussian_process.GaussianProcessClassifier",
        "class_path_reg": "sklearn.gaussian_process.GaussianProcessRegressor",
        "category": "probabilistic",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "slow",
        "memory_usage": "high",
        "scalability": "small",
        "best_for": ["small_data", "uncertainty_quantification", "bayesian_modeling"],
        "search_space": {
            "max_iter_predict": {"type": "int", "low": 50, "high": 200},
        }
    },
    # Bagging Meta-Estimator
    "bagging": {
        "display_name": "Bagging Ensemble",
        "class_path_clf": "sklearn.ensemble.BaggingClassifier",
        "class_path_reg": "sklearn.ensemble.BaggingRegressor",
        "category": "ensemble",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["variance_reduction", "stability", "general_purpose"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 10, "high": 100},
            "max_samples": {"type": "uniform", "low": 0.5, "high": 1.0},
            "max_features": {"type": "uniform", "low": 0.5, "high": 1.0},
        }
    },
    # Isolation Forest (Outlier & Anomaly Detection)
    "isolation_forest": {
        "display_name": "Isolation Forest (Anomaly)",
        "class_path": "sklearn.ensemble.IsolationForest",
        "category": "unsupervised",
        "library": "sklearn",
        "supports": ["anomaly_detection", "unsupervised"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["anomaly_detection", "outlier_identification", "fraud_detection"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "contamination": {"type": "uniform", "low": 0.01, "high": 0.2},
        }
    },
    # K-Means Clustering
    "kmeans": {
        "display_name": "K-Means Clustering",
        "class_path": "sklearn.cluster.KMeans",
        "category": "unsupervised",
        "library": "sklearn",
        "supports": ["clustering", "unsupervised"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["customer_segmentation", "grouping", "unsupervised_clustering"],
        "search_space": {
            "n_clusters": {"type": "int", "low": 2, "high": 15},
            "init": {"type": "categorical", "choices": ["k-means++", "random"]},
        }
    },
    # Principal Component Analysis
    "pca": {
        "display_name": "Principal Component Analysis (PCA)",
        "class_path": "sklearn.decomposition.PCA",
        "category": "unsupervised",
        "library": "sklearn",
        "supports": ["dimensionality_reduction", "unsupervised"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["dimensionality_reduction", "data_compression", "visualization"],
        "search_space": {
            "n_components": {"type": "int", "low": 2, "high": 10},
        }
    },

    # ─── Extended Linear & Robust Models ──────────────────────────────────────
    "passive_aggressive_classifier": {
        "display_name": "Passive-Aggressive Classifier",
        "class_path": "sklearn.linear_model.PassiveAggressiveClassifier",
        "category": "linear",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "very_fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["large_scale", "online_learning", "streaming_data"],
        "search_space": {
            "C": {"type": "loguniform", "low": 0.001, "high": 100},
            "max_iter": {"type": "int", "low": 500, "high": 2000},
        }
    },
    "passive_aggressive_regressor": {
        "display_name": "Passive-Aggressive Regressor",
        "class_path": "sklearn.linear_model.PassiveAggressiveRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "very_fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["large_scale", "online_learning", "streaming_data"],
        "search_space": {
            "C": {"type": "loguniform", "low": 0.001, "high": 100},
            "max_iter": {"type": "int", "low": 500, "high": 2000},
        }
    },
    "perceptron": {
        "display_name": "Perceptron",
        "class_path": "sklearn.linear_model.Perceptron",
        "category": "linear",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "very_fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["linearly_separable", "fast_baseline", "streaming"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 0.00001, "high": 0.1},
            "max_iter": {"type": "int", "low": 500, "high": 2000},
        }
    },
    "bayesian_ridge": {
        "display_name": "Bayesian Ridge Regression",
        "class_path": "sklearn.linear_model.BayesianRidge",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["probabilistic_regression", "small_data", "uncertainty_estimation"],
        "search_space": {
            "n_iter": {"type": "int", "low": 100, "high": 1000},
            "alpha_1": {"type": "loguniform", "low": 1e-7, "high": 1e-3},
            "lambda_1": {"type": "loguniform", "low": 1e-7, "high": 1e-3},
        }
    },
    "ard_regression": {
        "display_name": "ARD Regression (Automatic Relevance Determination)",
        "class_path": "sklearn.linear_model.ARDRegression",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["sparse_features", "feature_pruning", "bayesian"],
        "search_space": {
            "n_iter": {"type": "int", "low": 100, "high": 600},
            "alpha_1": {"type": "loguniform", "low": 1e-7, "high": 1e-3},
        }
    },
    "theil_sen": {
        "display_name": "Theil-Sen Robust Regressor",
        "class_path": "sklearn.linear_model.TheilSenRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["outlier_robust", "breakdown_point_robustness"],
        "search_space": {
            "max_subpopulation": {"type": "int", "low": 100, "high": 1000},
        }
    },
    "ransac": {
        "display_name": "RANSAC Regressor (Random Sample Consensus)",
        "class_path": "sklearn.linear_model.RANSACRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["extreme_outliers", "robust_estimation"],
        "search_space": {
            "min_samples": {"type": "int", "low": 5, "high": 50},
        }
    },
    "huber_regressor": {
        "display_name": "Huber Robust Regressor",
        "class_path": "sklearn.linear_model.HuberRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["heavy_tailed_noise", "linear_outlier_robust"],
        "search_space": {
            "epsilon": {"type": "uniform", "low": 1.1, "high": 2.0},
            "alpha": {"type": "loguniform", "low": 0.0001, "high": 10.0},
        }
    },
    "quantile_regressor": {
        "display_name": "Quantile Regressor",
        "class_path": "sklearn.linear_model.QuantileRegressor",
        "category": "linear",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["median_regression", "quantile_forecasting", "non_normal_errors"],
        "search_space": {
            "quantile": {"type": "uniform", "low": 0.1, "high": 0.9},
            "alpha": {"type": "loguniform", "low": 0.001, "high": 10.0},
        }
    },

    # ─── Extended Support Vector Machines ─────────────────────────────────────
    "nu_svc": {
        "display_name": "Nu-Support Vector Classifier",
        "class_path": "sklearn.svm.NuSVC",
        "category": "svm",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["error_rate_control", "high_dimensional", "margin_control"],
        "search_space": {
            "nu": {"type": "uniform", "low": 0.1, "high": 0.8},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
        }
    },
    "nu_svr": {
        "display_name": "Nu-Support Vector Regressor",
        "class_path": "sklearn.svm.NuSVR",
        "category": "svm",
        "library": "sklearn",
        "supports": ["regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["error_rate_control", "non_linear_regression"],
        "search_space": {
            "nu": {"type": "uniform", "low": 0.1, "high": 0.8},
            "C": {"type": "loguniform", "low": 0.1, "high": 100},
        }
    },

    # ─── Extended Nearest Neighbors & Centroids ───────────────────────────────
    "nearest_centroid": {
        "display_name": "Nearest Centroid Classifier",
        "class_path": "sklearn.neighbors.NearestCentroid",
        "category": "nearest_neighbors",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "very_fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["text_classification", "fast_prototype", "high_dimensional_sparse"],
        "search_space": {
            "metric": {"type": "categorical", "choices": ["euclidean", "manhattan"]},
        }
    },
    "radius_neighbors": {
        "display_name": "Radius Neighbors",
        "class_path_clf": "sklearn.neighbors.RadiusNeighborsClassifier",
        "class_path_reg": "sklearn.neighbors.RadiusNeighborsRegressor",
        "category": "nearest_neighbors",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "slow",
        "memory_usage": "medium",
        "scalability": "medium",
        "best_for": ["fixed_radius_clusters", "spatial_data"],
        "search_space": {
            "radius": {"type": "uniform", "low": 1.0, "high": 10.0},
        }
    },

    # ─── Extended Naive Bayes ────────────────────────────────────────────────
    "bernoulli_nb": {
        "display_name": "Bernoulli Naive Bayes",
        "class_path": "sklearn.naive_bayes.BernoulliNB",
        "category": "naive_bayes",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "very_fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["binary_features", "text_bag_of_words", "boolean_data"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 0.01, "high": 10.0},
            "binarize": {"type": "uniform", "low": 0.0, "high": 1.0},
        }
    },
    "complement_nb": {
        "display_name": "Complement Naive Bayes (Imbalanced Data)",
        "class_path": "sklearn.naive_bayes.ComplementNB",
        "category": "naive_bayes",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "very_fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["imbalanced_data", "text_classification", "fast_training"],
        "search_space": {
            "alpha": {"type": "loguniform", "low": 0.01, "high": 10.0},
            "norm": {"type": "categorical", "choices": [True, False]},
        }
    },

    # ─── Clustering & Density Estimators ─────────────────────────────────────
    "dbscan": {
        "display_name": "DBSCAN (Density-Based Spatial Clustering)",
        "class_path": "sklearn.cluster.DBSCAN",
        "category": "clustering",
        "library": "sklearn",
        "supports": ["clustering"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "medium",
        "best_for": ["arbitrary_shapes", "noise_filtering", "spatial_clustering"],
        "search_space": {
            "eps": {"type": "uniform", "low": 0.1, "high": 5.0},
            "min_samples": {"type": "int", "low": 3, "high": 20},
        }
    },
    "birch": {
        "display_name": "BIRCH Clustering",
        "class_path": "sklearn.cluster.Birch",
        "category": "clustering",
        "library": "sklearn",
        "supports": ["clustering"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["large_datasets", "hierarchical_compression"],
        "search_space": {
            "threshold": {"type": "uniform", "low": 0.1, "high": 1.0},
            "n_clusters": {"type": "int", "low": 2, "high": 10},
        }
    },
    "agglomerative_clustering": {
        "display_name": "Agglomerative Hierarchical Clustering",
        "class_path": "sklearn.cluster.AgglomerativeClustering",
        "category": "clustering",
        "library": "sklearn",
        "supports": ["clustering"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "slow",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["dendrograms", "nested_clusters", "hierarchical_data"],
        "search_space": {
            "n_clusters": {"type": "int", "low": 2, "high": 10},
            "linkage": {"type": "categorical", "choices": ["ward", "complete", "average"]},
        }
    },
    "mean_shift": {
        "display_name": "Mean Shift Clustering",
        "class_path": "sklearn.cluster.MeanShift",
        "category": "clustering",
        "library": "sklearn",
        "supports": ["clustering"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["automatic_cluster_count", "mode_seeking"],
        "search_space": {
            "max_iter": {"type": "int", "low": 100, "high": 500},
        }
    },
    "gaussian_mixture": {
        "display_name": "Gaussian Mixture Models (GMM)",
        "class_path": "sklearn.mixture.GaussianMixture",
        "category": "clustering",
        "library": "sklearn",
        "supports": ["clustering", "density_estimation"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["soft_clustering", "ellipsoidal_clusters", "density_estimation"],
        "search_space": {
            "n_components": {"type": "int", "low": 2, "high": 10},
            "covariance_type": {"type": "categorical", "choices": ["full", "tied", "diag"]},
        }
    },
    "spectral_clustering": {
        "display_name": "Spectral Clustering (Graph Laplacian)",
        "class_path": "sklearn.cluster.SpectralClustering",
        "category": "clustering",
        "library": "sklearn",
        "supports": ["clustering"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "slow",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["non_convex_clusters", "manifold_structure", "graph_cut"],
        "search_space": {
            "n_clusters": {"type": "int", "low": 2, "high": 8},
        }
    },

    # ─── Decomposition & Manifold Learning ────────────────────────────────────
    "truncated_svd": {
        "display_name": "Truncated SVD (Latent Semantic Analysis)",
        "class_path": "sklearn.decomposition.TruncatedSVD",
        "category": "dimensionality_reduction",
        "library": "sklearn",
        "supports": ["dimensionality_reduction"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "very_fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["sparse_matrices", "text_lsa", "dimensionality_reduction"],
        "search_space": {
            "n_components": {"type": "int", "low": 2, "high": 10},
        }
    },
    "factor_analysis": {
        "display_name": "Factor Analysis",
        "class_path": "sklearn.decomposition.FactorAnalysis",
        "category": "dimensionality_reduction",
        "library": "sklearn",
        "supports": ["dimensionality_reduction"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["latent_variables", "correlated_features", "probabilistic"],
        "search_space": {
            "n_components": {"type": "int", "low": 2, "high": 8},
        }
    },
    "fast_ica": {
        "display_name": "FastICA (Independent Component Analysis)",
        "class_path": "sklearn.decomposition.FastICA",
        "category": "dimensionality_reduction",
        "library": "sklearn",
        "supports": ["dimensionality_reduction"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "large",
        "best_for": ["signal_separation", "non_gaussian_components"],
        "search_space": {
            "n_components": {"type": "int", "low": 2, "high": 8},
        }
    },
    "kernel_pca": {
        "display_name": "Kernel PCA (Non-Linear Reduction)",
        "class_path": "sklearn.decomposition.KernelPCA",
        "category": "dimensionality_reduction",
        "library": "sklearn",
        "supports": ["dimensionality_reduction"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["non_linear_manifolds", "complex_geometry"],
        "search_space": {
            "n_components": {"type": "int", "low": 2, "high": 8},
            "kernel": {"type": "categorical", "choices": ["linear", "poly", "rbf", "sigmoid"]},
        }
    },

    # ─── Anomaly & Outlier Detection ──────────────────────────────────────────
    "isolation_forest": {
        "display_name": "Isolation Forest (Anomaly Detection)",
        "class_path": "sklearn.ensemble.IsolationForest",
        "category": "anomaly_detection",
        "library": "sklearn",
        "supports": ["clustering", "anomaly_detection"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "fast",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "very_large",
        "best_for": ["outlier_detection", "unsupervised_anomaly", "high_dimensional"],
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 250},
            "contamination": {"type": "uniform", "low": 0.01, "high": 0.2},
        }
    },
    "elliptic_envelope": {
        "display_name": "Elliptic Envelope (Gaussian Outlier Detection)",
        "class_path": "sklearn.covariance.EllipticEnvelope",
        "category": "anomaly_detection",
        "library": "sklearn",
        "supports": ["clustering", "anomaly_detection"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "low",
        "scalability": "medium",
        "best_for": ["gaussian_distributed_outliers", "robust_covariance"],
        "search_space": {
            "contamination": {"type": "uniform", "low": 0.01, "high": 0.2},
        }
    },
    "local_outlier_factor": {
        "display_name": "Local Outlier Factor (LOF)",
        "class_path": "sklearn.neighbors.LocalOutlierFactor",
        "category": "anomaly_detection",
        "library": "sklearn",
        "supports": ["clustering", "anomaly_detection"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "slow",
        "memory_usage": "medium",
        "scalability": "medium",
        "best_for": ["density_based_outliers", "local_anomalies"],
        "search_space": {
            "n_neighbors": {"type": "int", "low": 10, "high": 40},
        }
    },
    "one_class_svm": {
        "display_name": "One-Class SVM (Novelty Detection)",
        "class_path": "sklearn.svm.OneClassSVM",
        "category": "anomaly_detection",
        "library": "sklearn",
        "supports": ["clustering", "anomaly_detection"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "slow",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["novelty_detection", "high_dimensional_boundaries"],
        "search_space": {
            "nu": {"type": "uniform", "low": 0.01, "high": 0.5},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
        }
    },

    # ─── Tabular Ensembles & Foundation Architectures ─────────────────────────
    "voting_ensemble": {
        "display_name": "Voting Ensemble (Multi-Algorithm Consensus)",
        "class_path_clf": "sklearn.ensemble.VotingClassifier",
        "class_path_reg": "sklearn.ensemble.VotingRegressor",
        "category": "ensemble",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["variance_reduction", "maximum_robustness", "meta_ensembling"],
        "search_space": {
            "voting": {"type": "categorical", "choices": ["soft", "hard"]},
        }
    },
    "stacking_ensemble": {
        "display_name": "Stacking Ensemble (Meta-Learner Blending)",
        "class_path_clf": "sklearn.ensemble.StackingClassifier",
        "class_path_reg": "sklearn.ensemble.StackingRegressor",
        "category": "ensemble",
        "library": "sklearn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": False,
        "handles_categorical": False,
        "training_speed": "medium",
        "prediction_speed": "medium",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["kaggle_winning_patterns", "highest_accuracy", "multi_model_stacking"],
        "search_space": {
            "cv": {"type": "int", "low": 3, "high": 5},
        }
    },
    "tabpfn": {
        "display_name": "TabPFN (Prior-Data Fitted Tabular Transformer)",
        "class_path_clf": "tabpfn.TabPFNClassifier",
        "class_path_reg": "tabpfn.TabPFNRegressor",
        "category": "foundation_model",
        "library": "tabpfn",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": True,
        "training_speed": "very_fast",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "small",
        "best_for": ["zero_shot_tabular", "small_data_champions", "instant_inference"],
        "search_space": {
            "N_ensemble_configurations": {"type": "int", "low": 4, "high": 32},
        }
    },
    "tabm": {
        "display_name": "TabM (Tabular Batch Ensembled Architecture)",
        "class_path_clf": "tabm.TabMClassifier",
        "class_path_reg": "tabm.TabMRegressor",
        "category": "deep_learning",
        "library": "tabm",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": True,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["deep_tabular_leader", "batch_ensembling", "modern_benchmarks"],
        "search_space": {
            "k": {"type": "int", "low": 8, "high": 32},
        }
    },

    # TabNet (Attentive Interpretable Tabular Deep Learning)
    "tabnet": {
        "display_name": "TabNet (Deep Tabular Attention)",
        "class_path_clf": "pytorch_tabnet.tab_model.TabNetClassifier",
        "class_path_reg": "pytorch_tabnet.tab_model.TabNetRegressor",
        "category": "deep_learning",
        "library": "pytorch_tabnet",
        "supports": ["binary_classification", "multiclass_classification", "regression"],
        "handles_missing": True,
        "handles_categorical": True,
        "training_speed": "medium",
        "prediction_speed": "fast",
        "memory_usage": "medium",
        "scalability": "large",
        "best_for": ["deep_tabular", "feature_selection_masks", "state_of_the_art_tabular"],
        "search_space": {
            "n_d": {"type": "int", "low": 8, "high": 64},
            "n_a": {"type": "int", "low": 8, "high": 64},
            "n_steps": {"type": "int", "low": 3, "high": 10},
            "gamma": {"type": "uniform", "low": 1.0, "high": 2.0},
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

            # Meta-ensembles are created and tuned in Stage 3 from top-performing trained models
            if name in ["voting_ensemble", "stacking_ensemble"]:
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
