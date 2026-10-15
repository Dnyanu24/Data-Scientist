"""Script to inject all extended algorithms into ALGORITHM_REGISTRY in app/ml/recommendation.py."""
import sys
import os

target_path = os.path.join(os.path.dirname(__file__), "app", "ml", "recommendation.py")

with open(target_path, "r", encoding="utf-8") as f:
    content = f.read()

# Marker right before the closing brace of ALGORITHM_REGISTRY
target_marker = '    # TabNet (Attentive Interpretable Tabular Deep Learning)'

if "passive_aggressive_classifier" in content:
    print("Extended algorithms already present in recommendation.py!")
    sys.exit(0)

additional_algos = '''
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
'''

if target_marker in content:
    new_content = content.replace(target_marker, additional_algos + "\n" + target_marker)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully expanded ALGORITHM_REGISTRY with 28+ new algorithms!")
else:
    print(f"Error: Target marker '{target_marker}' not found in recommendation.py")
    sys.exit(1)
