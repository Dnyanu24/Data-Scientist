"""Seed data for Algorithm Knowledge Base and Historical Meta-Learning Experiences."""
import logging
from datetime import datetime
from app.database.session import SyncSessionLocal
from app.models.models import AlgorithmKnowledge, ExperienceRecord, ProblemType
from app.ml.recommendation import ALGORITHM_REGISTRY
from app.ml.fingerprinting import fingerprinter

logger = logging.getLogger(__name__)


def seed_algorithm_knowledge(db):
    """Seed the algorithm knowledge base from ALGORITHM_REGISTRY."""
    existing_count = db.query(AlgorithmKnowledge).count()
    if existing_count > 0:
        return

    logger.info("Seeding Algorithm Knowledge Base...")
    for name, info in ALGORITHM_REGISTRY.items():
        supports = info.get("supports", [])
        algo = AlgorithmKnowledge(
            name=name,
            display_name=info.get("display_name", name),
            category=info.get("category", "general"),
            library=info.get("library", "sklearn"),
            class_path=info.get("class_path") or info.get("class_path_clf") or "",
            supports_classification="binary_classification" in supports or "multiclass_classification" in supports,
            supports_regression="regression" in supports,
            supports_multiclass="multiclass_classification" in supports,
            handles_missing=info.get("handles_missing", False),
            handles_categorical=info.get("handles_categorical", False),
            handles_text=info.get("handles_text", False),
            handles_imbalanced=info.get("handles_imbalanced", False),
            supports_feature_importance="feature_importance" in info.get("best_for", []),
            is_ensemble=info.get("category") == "ensemble",
            training_speed=info.get("training_speed", "medium"),
            prediction_speed=info.get("prediction_speed", "fast"),
            memory_usage=info.get("memory_usage", "medium"),
            scalability=info.get("scalability", "medium"),
            search_space=info.get("search_space", {}),
            description=f"Standard ML algorithm {info.get('display_name')} from {info.get('library')}",
            strengths=info.get("best_for", []),
            weaknesses=["May require feature scaling" if not info.get("category") in ["tree", "ensemble"] else "Higher compute overhead"],
        )
        db.add(algo)
    db.commit()
    logger.info(f"Seeded {len(ALGORITHM_REGISTRY)} algorithms.")


def seed_historical_experiences(db):
    """Seed historical dataset experiences for meta-learning similarity retrieval."""
    existing_count = db.query(ExperienceRecord).count()
    if existing_count > 0:
        return

    logger.info("Seeding Historical Meta-Learning Experiences...")

    historical_datasets = [
        {
            "problem_type": ProblemType.BINARY_CLASSIFICATION,
            "dataset_size": "medium",
            "num_features": 20,
            "feature_types": {"numeric": 15, "categorical": 5},
            "data_quality_score": 0.92,
            "best_algorithm": "random_forest",
            "best_metrics": {"accuracy": 0.942, "f1_score": 0.938, "precision": 0.935, "recall": 0.941, "roc_auc": 0.981},
            "best_hyperparameters": {"n_estimators": 250, "max_depth": 14, "min_samples_split": 4},
            "preprocessing_steps": [
                {"step": "impute_numeric", "strategy": "median"},
                {"step": "encode_categorical", "strategy": "target_encoding"},
                {"step": "scale_numeric", "strategy": "standard"}
            ],
            "all_results": [
                {"algo": "random_forest", "metrics": {"f1_score": 0.938}, "rank": 1},
                {"algo": "gradient_boosting", "metrics": {"f1_score": 0.932}, "rank": 2},
                {"algo": "logistic_regression", "metrics": {"f1_score": 0.865}, "rank": 3},
            ],
            "mock_fp": {
                "num_rows": 5000, "num_columns": 21, "num_numeric": 15, "num_categorical": 5,
                "missing_ratio": 0.02, "avg_correlation": 0.28, "feature_to_sample_ratio": 0.004,
                "problem_type": "binary_classification", "size_category": "medium"
            }
        },
        {
            "problem_type": ProblemType.BINARY_CLASSIFICATION,
            "dataset_size": "small",
            "num_features": 8,
            "feature_types": {"numeric": 8, "categorical": 0},
            "data_quality_score": 0.88,
            "best_algorithm": "gradient_boosting",
            "best_metrics": {"accuracy": 0.895, "f1_score": 0.887, "precision": 0.880, "recall": 0.894, "roc_auc": 0.942},
            "best_hyperparameters": {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 5},
            "preprocessing_steps": [
                {"step": "impute_numeric", "strategy": "mean"},
                {"step": "scale_numeric", "strategy": "robust"}
            ],
            "all_results": [
                {"algo": "gradient_boosting", "metrics": {"f1_score": 0.887}, "rank": 1},
                {"algo": "random_forest", "metrics": {"f1_score": 0.879}, "rank": 2},
                {"algo": "svm", "metrics": {"f1_score": 0.852}, "rank": 3},
            ],
            "mock_fp": {
                "num_rows": 768, "num_columns": 9, "num_numeric": 8, "num_categorical": 0,
                "missing_ratio": 0.05, "avg_correlation": 0.35, "feature_to_sample_ratio": 0.01,
                "problem_type": "binary_classification", "size_category": "small"
            }
        },
        {
            "problem_type": ProblemType.REGRESSION,
            "dataset_size": "medium",
            "num_features": 13,
            "feature_types": {"numeric": 11, "categorical": 2},
            "data_quality_score": 0.95,
            "best_algorithm": "random_forest",
            "best_metrics": {"r2_score": 0.912, "mse": 12.4, "rmse": 3.52, "mae": 2.31},
            "best_hyperparameters": {"n_estimators": 300, "max_depth": 16, "min_samples_split": 3},
            "preprocessing_steps": [
                {"step": "impute_numeric", "strategy": "median"},
                {"step": "encode_categorical", "strategy": "one_hot"},
                {"step": "scale_numeric", "strategy": "standard"}
            ],
            "all_results": [
                {"algo": "random_forest", "metrics": {"r2_score": 0.912}, "rank": 1},
                {"algo": "gradient_boosting", "metrics": {"r2_score": 0.905}, "rank": 2},
                {"algo": "ridge_regression", "metrics": {"r2_score": 0.785}, "rank": 3},
            ],
            "mock_fp": {
                "num_rows": 2500, "num_columns": 14, "num_numeric": 11, "num_categorical": 2,
                "missing_ratio": 0.01, "avg_correlation": 0.42, "feature_to_sample_ratio": 0.005,
                "problem_type": "regression", "size_category": "medium"
            }
        },
        {
            "problem_type": ProblemType.MULTICLASS_CLASSIFICATION,
            "dataset_size": "small",
            "num_features": 4,
            "feature_types": {"numeric": 4, "categorical": 0},
            "data_quality_score": 1.0,
            "best_algorithm": "svm",
            "best_metrics": {"accuracy": 0.980, "f1_score": 0.980, "precision": 0.981, "recall": 0.980},
            "best_hyperparameters": {"C": 1.5, "kernel": "rbf", "gamma": "scale"},
            "preprocessing_steps": [
                {"step": "scale_numeric", "strategy": "standard"}
            ],
            "all_results": [
                {"algo": "svm", "metrics": {"f1_score": 0.980}, "rank": 1},
                {"algo": "logistic_regression", "metrics": {"f1_score": 0.973}, "rank": 2},
                {"algo": "random_forest", "metrics": {"f1_score": 0.967}, "rank": 3},
            ],
            "mock_fp": {
                "num_rows": 150, "num_columns": 5, "num_numeric": 4, "num_categorical": 0,
                "missing_ratio": 0.0, "avg_correlation": 0.65, "feature_to_sample_ratio": 0.026,
                "problem_type": "multiclass_classification", "size_category": "small"
            }
        },
        {
            "problem_type": ProblemType.BINARY_CLASSIFICATION,
            "dataset_size": "large",
            "num_features": 30,
            "feature_types": {"numeric": 28, "categorical": 2},
            "data_quality_score": 0.96,
            "best_algorithm": "extra_trees",
            "best_metrics": {"accuracy": 0.968, "f1_score": 0.965, "precision": 0.970, "recall": 0.960, "roc_auc": 0.992},
            "best_hyperparameters": {"n_estimators": 350, "max_depth": 20, "min_samples_split": 2},
            "preprocessing_steps": [
                {"step": "impute_numeric", "strategy": "median"},
                {"step": "scale_numeric", "strategy": "robust"},
                {"step": "feature_selection", "strategy": "mutual_info", "k": 20}
            ],
            "all_results": [
                {"algo": "extra_trees", "metrics": {"f1_score": 0.965}, "rank": 1},
                {"algo": "random_forest", "metrics": {"f1_score": 0.961}, "rank": 2},
                {"algo": "gradient_boosting", "metrics": {"f1_score": 0.958}, "rank": 3},
            ],
            "mock_fp": {
                "num_rows": 15000, "num_columns": 31, "num_numeric": 28, "num_categorical": 2,
                "missing_ratio": 0.005, "avg_correlation": 0.31, "feature_to_sample_ratio": 0.002,
                "problem_type": "binary_classification", "size_category": "large"
            }
        }
    ]

    for item in historical_datasets:
        embedding = fingerprinter.fingerprint_to_embedding(item["mock_fp"])
        record = ExperienceRecord(
            problem_type=item["problem_type"],
            dataset_size=item["dataset_size"],
            num_features=item["num_features"],
            feature_types=item["feature_types"],
            data_quality_score=item["data_quality_score"],
            best_algorithm=item["best_algorithm"],
            best_metrics=item["best_metrics"],
            best_hyperparameters=item["best_hyperparameters"],
            preprocessing_steps=item["preprocessing_steps"],
            fingerprint_embedding=embedding,
            all_results=item["all_results"],
            created_at=datetime.utcnow(),
        )
        db.add(record)

    db.commit()
    logger.info(f"Seeded {len(historical_datasets)} historical experience records.")


def seed_all():
    """Seed both algorithm knowledge and meta-learning experiences."""
    db = SyncSessionLocal()
    try:
        seed_algorithm_knowledge(db)
        seed_historical_experiences(db)
    finally:
        db.close()
