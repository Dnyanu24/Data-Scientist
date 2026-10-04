"""Background Celery tasks for long-running ML operations."""
import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.workers.celery_app import celery_app
from app.database.session import SyncSessionLocal
from app.models.models import (
    Project, Dataset, DatasetFingerprint, AlgorithmRecommendation,
    PreprocessingConfig, Experiment, TrainedModel, BackgroundTask,
    ExperienceRecord, PipelinePhase, ModelStatus, TrainingStage
)
from app.ml.profiling import data_profiler
from app.ml.fingerprinting import fingerprinter
from app.ml.recommendation import recommender
from app.ml.preprocessing import PreprocessingPipeline
from app.ml.training import TrainingEngine
from app.ml.evaluation import evaluator
from app.core.config import settings

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def update_task_progress(task_id: int, progress: float, message: str, status: str = "running", session=None):
    """Update background task progress in DB."""
    close_db = False
    if session is not None:
        db = session
    else:
        db = SyncSessionLocal()
        close_db = True
    try:
        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.progress = progress
            task.message = message
            task.status = status
            task.updated_at = datetime.utcnow()
            db.commit()
    finally:
        if close_db:
            db.close()


def update_project_phase(project_id: int, phase: str, progress: float = 0.0, message: str = "", session=None):
    """Update project phase."""
    close_db = False
    if session is not None:
        db = session
    else:
        db = SyncSessionLocal()
        close_db = True
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.phase = phase
            project.phase_progress = progress
            project.phase_message = message
            project.updated_at = datetime.utcnow()
            db.commit()
    finally:
        if close_db:
            db.close()



@celery_app.task(bind=True, name="profile_dataset")
def profile_dataset_task(self, project_id: int, dataset_id: int, task_id: int):
    """Profile dataset: schema detection, quality analysis, target detection."""
    db = SyncSessionLocal()
    try:
        update_task_progress(task_id, 0.1, "Loading dataset...")
        update_project_phase(project_id, PipelinePhase.PROFILING.value, 0.1, "Loading dataset...")

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        project = db.query(Project).filter(Project.id == project_id).first()
        if not dataset or not project:
            raise ValueError("Dataset or project not found")

        # Load data
        file_path = dataset.file_path
        if dataset.file_type == "csv":
            df = pd.read_csv(file_path)
        elif dataset.file_type in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {dataset.file_type}")

        update_task_progress(task_id, 0.2, "Detecting target column...")
        update_project_phase(project_id, PipelinePhase.PROFILING.value, 0.2, "Detecting target column...")

        # Detect target column
        target_column = project.target_column
        if not target_column:
            target_column = data_profiler.detect_target_column(df, project.business_goal)
            project.target_column = target_column
            db.commit()

        # Detect problem type
        if target_column and target_column in df.columns:
            problem_type = data_profiler.detect_problem_type(df, target_column)
            project.problem_type = problem_type
            db.commit()

        update_task_progress(task_id, 0.4, "Generating profile...")
        update_project_phase(project_id, PipelinePhase.PROFILING.value, 0.4, "Generating profile...")

        # Generate profile
        profile = data_profiler.profile_dataset(df, target_column)

        update_task_progress(task_id, 0.7, "Analyzing data quality...")
        update_project_phase(project_id, PipelinePhase.PROFILING.value, 0.7, "Analyzing data quality...")

        # Update dataset
        dataset.num_rows = len(df)
        dataset.num_columns = len(df.columns)
        dataset.columns_info = profile.get("columns", [])
        dataset.schema_info = profile.get("schema", {})
        dataset.profile_report = profile
        dataset.data_quality = profile.get("data_quality", {})
        dataset.target_analysis = profile.get("target_analysis", {})

        update_task_progress(task_id, 0.9, "Finalizing...")
        update_project_phase(project_id, PipelinePhase.PROFILED.value, 1.0, "Profiling completed")

        db.commit()

        # Update task as completed
        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = 1.0
            task.message = "Profiling completed"
            task.completed_at = datetime.utcnow()
            task.result = {
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "target_column": target_column,
                "problem_type": project.problem_type,
                "quality_score": profile.get("data_quality", {}).get("scores", {}).get("overall", 0),
            }
            db.commit()

        return {"status": "completed", "project_id": project_id}

    except Exception as e:
        logger.error(f"Profile task failed: {e}")
        update_task_progress(task_id, 0, str(e), "failed")
        update_project_phase(project_id, PipelinePhase.FAILED.value, 0, str(e))
        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.error = str(e)
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="generate_fingerprint")
def generate_fingerprint_task(self, project_id: int, dataset_id: int, task_id: int):
    """Generate dataset fingerprint and store embedding."""
    db = SyncSessionLocal()
    try:
        update_task_progress(task_id, 0.1, "Loading dataset...")
        update_project_phase(project_id, PipelinePhase.FINGERPRINTING.value, 0.1)

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        project = db.query(Project).filter(Project.id == project_id).first()
        if not dataset or not project:
            raise ValueError("Dataset or project not found")

        df = pd.read_csv(dataset.file_path) if dataset.file_type == "csv" else pd.read_excel(dataset.file_path)

        update_task_progress(task_id, 0.3, "Computing fingerprint...")

        # Generate fingerprint
        fp_data = fingerprinter.generate_fingerprint(
            df, project.target_column, project.problem_type
        )

        update_task_progress(task_id, 0.6, "Computing embedding...")

        # Generate embedding
        embedding = fingerprinter.fingerprint_to_embedding(fp_data)

        # Store fingerprint
        existing = db.query(DatasetFingerprint).filter(
            DatasetFingerprint.dataset_id == dataset_id
        ).first()

        if existing:
            for key, value in fp_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.fingerprint_data = fp_data
            existing.embedding = embedding
        else:
            fp = DatasetFingerprint(
                dataset_id=dataset_id,
                project_id=project_id,
                num_rows=fp_data.get("num_rows"),
                num_columns=fp_data.get("num_columns"),
                num_numeric=fp_data.get("num_numeric"),
                num_categorical=fp_data.get("num_categorical"),
                num_text=fp_data.get("num_text"),
                num_datetime=fp_data.get("num_datetime"),
                missing_ratio=fp_data.get("missing_ratio"),
                class_imbalance_ratio=fp_data.get("class_imbalance_ratio"),
                avg_correlation=fp_data.get("avg_correlation"),
                feature_to_sample_ratio=fp_data.get("feature_to_sample_ratio"),
                outlier_ratio=fp_data.get("outlier_ratio"),
                numeric_skewness_mean=fp_data.get("numeric_skewness_mean"),
                problem_type=project.problem_type,
                fingerprint_data=fp_data,
                embedding=embedding,
            )
            db.add(fp)

        update_project_phase(project_id, PipelinePhase.FINGERPRINTED.value, 1.0, "Fingerprinting completed")

        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = 1.0
            task.message = "Fingerprinting completed"
            task.completed_at = datetime.utcnow()
            task.result = fp_data
        db.commit()

        return {"status": "completed", "project_id": project_id}

    except Exception as e:
        logger.error(f"Fingerprint task failed: {e}")
        update_task_progress(task_id, 0, str(e), "failed")
        update_project_phase(project_id, PipelinePhase.FAILED.value, 0, str(e))
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="recommend_algorithms")
def recommend_algorithms_task(self, project_id: int, task_id: int):
    """Generate algorithm recommendations."""
    db = SyncSessionLocal()
    try:
        update_task_progress(task_id, 0.1, "Analyzing dataset...")
        update_project_phase(project_id, PipelinePhase.RECOMMENDING.value, 0.1)

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError("Project not found")

        # Get fingerprint
        dataset = db.query(Dataset).filter(Dataset.project_id == project_id).first()
        fp = db.query(DatasetFingerprint).filter(DatasetFingerprint.project_id == project_id).first()

        if not fp:
            raise ValueError("Fingerprint not found. Run fingerprinting first.")

        fp_data = fp.fingerprint_data or {}

        update_task_progress(task_id, 0.3, "Finding similar experiments...")

        # Find similar historical experiments
        historical = []
        if fp.embedding:
            target_embedding = list(fp.embedding) if hasattr(fp.embedding, "__iter__") else json.loads(fp.embedding) if isinstance(fp.embedding, str) else []
            is_postgres = False
            try:
                bind_url = str(db.get_bind().url).lower()
                is_postgres = "postgres" in bind_url
            except Exception:
                pass

            if is_postgres:
                try:
                    from sqlalchemy import text
                    result = db.execute(
                        text("""
                            SELECT er.*, 1 - (er.fingerprint_embedding <=> :embedding) as similarity
                            FROM experience_records er
                            WHERE er.fingerprint_embedding IS NOT NULL
                            ORDER BY er.fingerprint_embedding <=> :embedding
                            LIMIT 10
                        """),
                        {"embedding": str(target_embedding)}
                    )
                    for row in result:
                        row_dict = dict(row._mapping)
                        if row_dict.get("similarity", 0) > 0.5:
                            historical.append({
                                "best_algorithm": row_dict.get("best_algorithm"),
                                "best_metrics": row_dict.get("best_metrics"),
                                "all_results": row_dict.get("all_results"),
                                "similarity": float(row_dict.get("similarity", 0)),
                            })
                except Exception as pg_exc:
                    logger.warning(f"Postgres pgvector query failed ({pg_exc}), falling back to in-memory cosine similarity.")
                    is_postgres = False

            if not is_postgres or not historical:
                experiences = db.query(ExperienceRecord).filter(ExperienceRecord.fingerprint_embedding.isnot(None)).all()
                for exp in experiences:
                    exp_emb = list(exp.fingerprint_embedding) if hasattr(exp.fingerprint_embedding, "__iter__") else json.loads(exp.fingerprint_embedding) if isinstance(exp.fingerprint_embedding, str) else []
                    if exp_emb and target_embedding:
                        sim = fingerprinter.compute_similarity(target_embedding, exp_emb)
                        if sim > 0.5:
                            historical.append({
                                "best_algorithm": exp.best_algorithm,
                                "best_metrics": exp.best_metrics,
                                "all_results": exp.all_results,
                                "similarity": round(float(sim), 4),
                            })

        update_task_progress(task_id, 0.5, "Computing recommendations...")

        # Generate recommendations
        recommendations = recommender.recommend(
            fingerprint=fp_data,
            problem_type=project.problem_type or "binary_classification",
            historical_results=historical,
        )

        # Clear old recommendations
        db.query(AlgorithmRecommendation).filter(
            AlgorithmRecommendation.project_id == project_id
        ).delete()

        # Store new recommendations
        for rec in recommendations:
            ar = AlgorithmRecommendation(
                project_id=project_id,
                algorithm_name=rec["algorithm_name"],
                recommendation_score=rec["recommendation_score"],
                confidence=rec["confidence"],
                training_priority=rec["training_priority"],
                meta_learning_score=rec["scores"]["meta_learning"],
                historical_score=rec["scores"]["historical"],
                rule_based_score=rec["scores"]["rule_based"],
                compatibility_score=rec["scores"]["compatibility"],
                user_requirement_score=rec["scores"]["user_requirement"],
                reasoning=rec.get("reasoning"),
                training_stage=rec.get("training_stage", "stage_1"),
            )
            db.add(ar)

        project.phase = PipelinePhase.RECOMMENDED.value
        project.phase_progress = 1.0
        project.phase_message = "Recommendations ready"
        project.updated_at = datetime.utcnow()

        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = 1.0
            task.message = f"Generated {len(recommendations)} recommendations"
            task.completed_at = datetime.utcnow()
            task.result = {"num_recommendations": len(recommendations)}
        db.commit()

        return {"status": "completed", "recommendations": len(recommendations)}

    except Exception as e:
        logger.error(f"Recommendation task failed: {e}")
        update_task_progress(task_id, 0, str(e), "failed")
        update_project_phase(project_id, PipelinePhase.FAILED.value, 0, str(e))
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="preprocess_dataset")
def preprocess_dataset_task(self, project_id: int, task_id: int):
    """Build and apply preprocessing pipeline."""
    db = SyncSessionLocal()
    try:
        update_task_progress(task_id, 0.1, "Loading dataset...")
        update_project_phase(project_id, PipelinePhase.PREPROCESSING.value, 0.1)

        project = db.query(Project).filter(Project.id == project_id).first()
        dataset = db.query(Dataset).filter(Dataset.project_id == project_id).first()
        fp = db.query(DatasetFingerprint).filter(DatasetFingerprint.project_id == project_id).first()

        if not project or not dataset or not fp:
            raise ValueError("Project, dataset, or fingerprint not found")

        df = pd.read_csv(dataset.file_path) if dataset.file_type == "csv" else pd.read_excel(dataset.file_path)

        update_task_progress(task_id, 0.3, "Building preprocessing pipeline...")

        # Build pipeline
        pipeline = PreprocessingPipeline()
        fp_data = fp.fingerprint_data or {}
        steps = pipeline.build_pipeline(
            df, project.target_column, project.problem_type or "binary_classification", fp_data
        )

        update_task_progress(task_id, 0.6, "Applying preprocessing...")

        # Apply pipeline
        X, y = pipeline.apply_pipeline(df, steps, project.target_column, fit=True)

        # Save pipeline
        pipeline_path = os.path.join(settings.MODEL_STORE_DIR, str(project_id), "pipeline.pkl")
        pipeline.save(pipeline_path)

        # Store config
        existing = db.query(PreprocessingConfig).filter(
            PreprocessingConfig.project_id == project_id
        ).first()

        if existing:
            existing.pipeline_steps = steps
            existing.feature_columns = X.columns.tolist()
            existing.target_column = project.target_column
            existing.applied = True
        else:
            config = PreprocessingConfig(
                project_id=project_id,
                pipeline_steps=steps,
                feature_columns=X.columns.tolist(),
                target_column=project.target_column,
                applied=True,
            )
            db.add(config)

        project.phase = PipelinePhase.PREPROCESSED.value
        project.phase_progress = 1.0
        project.phase_message = "Preprocessing completed"
        project.updated_at = datetime.utcnow()

        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = 1.0
            task.message = f"Applied {len(steps)} preprocessing steps"
            task.completed_at = datetime.utcnow()
            task.result = {
                "steps": steps,
                "num_features": len(X.columns),
                "num_samples": len(X),
            }
        db.commit()

        return {"status": "completed"}

    except Exception as e:
        logger.error(f"Preprocessing task failed: {e}")
        update_task_progress(task_id, 0, str(e), "failed")
        update_project_phase(project_id, PipelinePhase.FAILED.value, 0, str(e))
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="train_models")
def train_models_task(self, project_id: int, task_id: int,
                      max_trials_s1: int = 15, max_trials_s2: int = 40,
                      max_trials_s3: int = 100, create_ensembles: bool = True):
    """Progressive model training with Optuna optimization."""
    db = SyncSessionLocal()
    try:
        update_task_progress(task_id, 0.01, "Loading data...")
        update_project_phase(project_id, PipelinePhase.TRAINING.value, 0.01, "Loading data...")

        project = db.query(Project).filter(Project.id == project_id).first()
        dataset = db.query(Dataset).filter(Dataset.project_id == project_id).first()

        if not project or not dataset:
            raise ValueError("Project or dataset not found")

        # Load and preprocess data
        df = pd.read_csv(dataset.file_path) if dataset.file_type == "csv" else pd.read_excel(dataset.file_path)

        # Get fingerprint
        fp = db.query(DatasetFingerprint).filter(DatasetFingerprint.project_id == project_id).first()
        fp_data = fp.fingerprint_data if fp else {}

        # Build preprocessing pipeline
        pipeline = PreprocessingPipeline()
        steps_config = db.query(PreprocessingConfig).filter(
            PreprocessingConfig.project_id == project_id
        ).first()

        if steps_config and steps_config.pipeline_steps:
            X, y = pipeline.apply_pipeline(df, steps_config.pipeline_steps, project.target_column, fit=True)
        else:
            steps = pipeline.build_pipeline(
                df, project.target_column, project.problem_type or "binary_classification", fp_data
            )
            X, y = pipeline.apply_pipeline(df, steps, project.target_column, fit=True)

        # Save pipeline
        pipeline_path = os.path.join(settings.MODEL_STORE_DIR, str(project_id), "pipeline.pkl")
        pipeline.save(pipeline_path)

        # Split data
        problem_type = project.problem_type or "binary_classification"
        if "classification" in problem_type:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Get recommendations
        recs = db.query(AlgorithmRecommendation).filter(
            AlgorithmRecommendation.project_id == project_id
        ).order_by(AlgorithmRecommendation.training_priority).all()

        if not recs:
            # Generate on-the-fly
            recommendations = recommender.recommend(fp_data, problem_type)
            algo_names = [r["algorithm_name"] for r in recommendations[:10]]
        else:
            algo_names = [r.algorithm_name for r in recs]

        engine = TrainingEngine(settings.MODEL_STORE_DIR)
        all_results = []
        stage_1_algos = algo_names[:10]
        total_algos = len(stage_1_algos)

        # ─── STAGE 1: Quick scan ─────────────────────────────────────────
        update_task_progress(task_id, 0.05, "Stage 1: Quick evaluation...")
        update_project_phase(project_id, PipelinePhase.TRAINING.value, 0.05, "Stage 1: Quick evaluation")

        for i, algo in enumerate(stage_1_algos):
            progress = 0.05 + (i / total_algos) * 0.30
            update_task_progress(task_id, progress, f"Stage 1: Training {algo}...")
            update_project_phase(project_id, PipelinePhase.TRAINING.value, progress, f"Training {algo}")

            result = engine.train_model(
                X_train, y_train, X_val, y_val,
                algo, problem_type, n_trials=max_trials_s1,
                training_stage="stage_1"
            )

            if result["status"] == "completed":
                # Save model
                model_path = engine.save_model(result["model"], project_id, f"{algo}_s1")
                result["model_path"] = model_path

                # Store in DB
                tm = TrainedModel(
                    project_id=project_id,
                    name=f"{algo}_stage1",
                    algorithm_name=algo,
                    status=ModelStatus.TRAINED,
                    model_path=model_path,
                    pipeline_path=pipeline_path,
                    metrics=result["metrics"],
                    hyperparameters=result["best_params"],
                    feature_importance=result.get("feature_importance"),
                    training_stage=TrainingStage.STAGE_1,
                    training_time=result["training_time"],
                )
                db.add(tm)

                # Store experiment
                exp = Experiment(
                    project_id=project_id,
                    name=f"{algo}_stage1",
                    algorithm_name=algo,
                    hyperparameters=result["best_params"],
                    training_stage=TrainingStage.STAGE_1,
                    metrics=result["metrics"],
                    training_time=result["training_time"],
                    num_trials=max_trials_s1,
                    status="completed",
                    completed_at=datetime.utcnow(),
                    dataset_fingerprint_id=fp.id if fp else None,
                )
                db.add(exp)
                db.commit()

                all_results.append(result)
                del result["model"]  # Free memory

        # ─── STAGE 2: Deep dive top 6 ────────────────────────────────────
        # Sort by primary metric
        def get_metric(r):
            if "classification" in problem_type:
                return r.get("metrics", {}).get("f1_score", 0)
            return -r.get("metrics", {}).get("mse", float("inf"))

        all_results.sort(key=get_metric, reverse=True)
        stage_2_algos = [r["algorithm_name"] for r in all_results[:6]]

        update_task_progress(task_id, 0.40, "Stage 2: Deep optimization...")
        update_project_phase(project_id, PipelinePhase.TRAINING.value, 0.40, "Stage 2: Deep optimization")

        stage_2_results = []
        for i, algo in enumerate(stage_2_algos):
            progress = 0.40 + (i / len(stage_2_algos)) * 0.25
            update_task_progress(task_id, progress, f"Stage 2: Optimizing {algo}...")

            result = engine.train_model(
                X_train, y_train, X_val, y_val,
                algo, problem_type, n_trials=max_trials_s2,
                training_stage="stage_2"
            )

            if result["status"] == "completed":
                model_path = engine.save_model(result["model"], project_id, f"{algo}_s2")
                result["model_path"] = model_path

                tm = TrainedModel(
                    project_id=project_id,
                    name=f"{algo}_stage2",
                    algorithm_name=algo,
                    status=ModelStatus.TRAINED,
                    model_path=model_path,
                    pipeline_path=pipeline_path,
                    metrics=result["metrics"],
                    hyperparameters=result["best_params"],
                    feature_importance=result.get("feature_importance"),
                    training_stage=TrainingStage.STAGE_2,
                    training_time=result["training_time"],
                )
                db.add(tm)

                exp = Experiment(
                    project_id=project_id,
                    name=f"{algo}_stage2",
                    algorithm_name=algo,
                    hyperparameters=result["best_params"],
                    training_stage=TrainingStage.STAGE_2,
                    metrics=result["metrics"],
                    training_time=result["training_time"],
                    num_trials=max_trials_s2,
                    status="completed",
                    completed_at=datetime.utcnow(),
                )
                db.add(exp)
                db.commit()

                stage_2_results.append(result)
                del result["model"]

        # ─── STAGE 3: Final tuning top 3 ─────────────────────────────────
        stage_2_results.sort(key=get_metric, reverse=True)
        stage_3_algos = [r["algorithm_name"] for r in stage_2_results[:3]]

        update_task_progress(task_id, 0.70, "Stage 3: Final optimization...")
        update_project_phase(project_id, PipelinePhase.TRAINING.value, 0.70, "Stage 3: Final optimization")

        best_models = []
        stage_3_results = []
        for i, algo in enumerate(stage_3_algos):
            progress = 0.70 + (i / len(stage_3_algos)) * 0.15
            update_task_progress(task_id, progress, f"Stage 3: Fine-tuning {algo}...")

            result = engine.train_model(
                X_train, y_train, X_val, y_val,
                algo, problem_type, n_trials=max_trials_s3,
                training_stage="stage_3"
            )

            if result["status"] == "completed":
                model_path = engine.save_model(result["model"], project_id, f"{algo}_s3")

                # Evaluate fully
                eval_result = evaluator.evaluate(result["model"], X_val, y_val, problem_type)
                explain_result = evaluator.explain_model(
                    result["model"], X_val, X_train.columns.tolist(), problem_type
                )

                tm = TrainedModel(
                    project_id=project_id,
                    name=f"{algo}_stage3",
                    algorithm_name=algo,
                    status=ModelStatus.EVALUATED,
                    model_path=model_path,
                    pipeline_path=pipeline_path,
                    metrics=result["metrics"],
                    hyperparameters=result["best_params"],
                    feature_importance=result.get("feature_importance"),
                    explainability={**eval_result, **explain_result},
                    training_stage=TrainingStage.STAGE_3,
                    training_time=result["training_time"],
                )
                db.add(tm)

                exp = Experiment(
                    project_id=project_id,
                    name=f"{algo}_stage3",
                    algorithm_name=algo,
                    hyperparameters=result["best_params"],
                    training_stage=TrainingStage.STAGE_3,
                    metrics=result["metrics"],
                    training_time=result["training_time"],
                    num_trials=max_trials_s3,
                    status="completed",
                    completed_at=datetime.utcnow(),
                )
                db.add(exp)
                db.commit()

                best_models.append(result["model"])
                result["model_path"] = model_path
                stage_3_results.append(result)
                del result["model"]

        # ─── ENSEMBLES ───────────────────────────────────────────────────
        if create_ensembles and len(best_models) >= 2:
            update_task_progress(task_id, 0.88, "Creating ensembles...")
            update_project_phase(project_id, PipelinePhase.TRAINING.value, 0.88, "Creating ensembles")

            for ens_type in ["voting", "stacking"]:
                try:
                    ens_result = engine.create_ensemble(
                        best_models, ens_type,
                        X_train, y_train, X_val, y_val, problem_type
                    )
                    if ens_result["status"] == "completed":
                        model_path = engine.save_model(
                            ens_result["model"], project_id, f"ensemble_{ens_type}"
                        )
                        tm = TrainedModel(
                            project_id=project_id,
                            name=f"ensemble_{ens_type}",
                            algorithm_name=f"ensemble_{ens_type}",
                            status=ModelStatus.EVALUATED,
                            model_path=model_path,
                            pipeline_path=pipeline_path,
                            metrics=ens_result["metrics"],
                            is_ensemble=True,
                            ensemble_type=ens_type,
                            training_stage=TrainingStage.ENSEMBLE,
                            training_time=ens_result["training_time"],
                        )
                        db.add(tm)
                        db.commit()
                except Exception as e:
                    logger.warning(f"Ensemble {ens_type} failed: {e}")

        # Mark best model
        all_models = db.query(TrainedModel).filter(
            TrainedModel.project_id == project_id,
            TrainedModel.status.in_([ModelStatus.TRAINED, ModelStatus.EVALUATED]),
        ).all()

        if all_models:
            if "classification" in problem_type:
                best = max(all_models, key=lambda m: (m.metrics or {}).get("f1_score", 0))
            else:
                best = min(all_models, key=lambda m: (m.metrics or {}).get("mse", float("inf")))
            best.is_best = True
            db.commit()

        # Store experience
        best_model = db.query(TrainedModel).filter(
            TrainedModel.project_id == project_id,
            TrainedModel.is_best == True,
        ).first()

        if best_model and fp:
            exp_record = ExperienceRecord(
                project_id=project_id,
                problem_type=project.problem_type,
                dataset_size=fp_data.get("size_category", "medium") if fp_data else "medium",
                num_features=fp.num_columns,
                data_quality_score=dataset.data_quality.get("scores", {}).get("overall") if dataset.data_quality else None,
                best_algorithm=best_model.algorithm_name,
                best_metrics=best_model.metrics,
                best_hyperparameters=best_model.hyperparameters,
                fingerprint_embedding=fp.embedding,
                all_results=[
                    {"algo": m.algorithm_name, "metrics": m.metrics, "rank": i + 1}
                    for i, m in enumerate(sorted(
                        all_models,
                        key=lambda m: (m.metrics or {}).get("f1_score", 0)
                        if "classification" in problem_type
                        else -(m.metrics or {}).get("mse", float("inf")),
                        reverse=True
                    ))
                ],
            )
            db.add(exp_record)

        project.phase = PipelinePhase.TRAINED.value
        project.phase_progress = 1.0
        project.phase_message = "Training completed"
        project.updated_at = datetime.utcnow()

        task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = 1.0
            task.message = "Training completed"
            task.completed_at = datetime.utcnow()
            task.result = {
                "total_models": len(all_models) if all_models else 0,
                "best_model": best_model.algorithm_name if best_model else None,
                "best_metrics": best_model.metrics if best_model else None,
            }
        db.commit()

        return {"status": "completed"}

    except Exception as e:
        logger.error(f"Training task failed: {e}")
        update_task_progress(task_id, 0, str(e), "failed")
        update_project_phase(project_id, PipelinePhase.FAILED.value, 0, str(e))
        raise
    finally:
        db.close()
