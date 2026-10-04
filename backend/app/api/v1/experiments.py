"""Training, Progressive Optimization, Model Evaluation & Explainability API."""
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.models import (
    Project, Experiment, TrainedModel, BackgroundTask, User,
    PipelinePhase, ModelStatus
)
from app.schemas.schemas import ExperimentResponse, TrainedModelResponse
from app.workers.dispatcher import dispatch_task
from app.workers.tasks import train_models_task

router = APIRouter()


@router.post("/projects/{project_id}/train")
async def start_training(
    project_id: int,
    max_trials_s1: int = Body(5, embed=True),
    max_trials_s2: int = Body(15, embed=True),
    max_trials_s3: int = Body(30, embed=True),
    create_ensembles: bool = Body(True, embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 7: Start Intelligent 3-Stage Progressive Training & Hyperparameter Optimization.
    
    Stage 1: Quick exploration across all recommended candidate algorithms.
    Stage 2: Deep optimization of top 6 models using Optuna Bayesian TPE.
    Stage 3: Fine-tuning top 3 models + Voting & Stacking Ensembles.
    """
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    task = BackgroundTask(
        project_id=project_id,
        task_type="training",
        status="pending",
        progress=0.0,
        message="Initializing 3-stage training pipeline...",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    dispatch_task(
        train_models_task,
        project_id,
        task.id,
        max_trials_s1,
        max_trials_s2,
        max_trials_s3,
        create_ensembles,
    )

    return {
        "status": "started",
        "task_id": task.id,
        "message": "Intelligent 3-stage training pipeline initiated in background."
    }


@router.get("/projects/{project_id}/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """List all experiment runs for a project."""
    stmt = select(Experiment).filter(Experiment.project_id == project_id).order_by(Experiment.created_at.desc())
    result = await db.execute(stmt)
    experiments = result.scalars().all()
    return [ExperimentResponse.model_validate(e) for e in experiments]


@router.get("/projects/{project_id}/leaderboard", response_model=List[TrainedModelResponse])
async def get_model_leaderboard(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 8: Model Evaluation & Decision Leaderboard.
    
    Returns all trained and evaluated models ranked by primary metric,
    highlighting the recommended champion model and ensemble models.
    """
    stmt = select(TrainedModel).filter(
        TrainedModel.project_id == project_id,
        TrainedModel.status.in_([ModelStatus.TRAINED, ModelStatus.EVALUATED, ModelStatus.REGISTERED, ModelStatus.DEPLOYED])
    )
    result = await db.execute(stmt)
    models = result.scalars().all()

    # Determine problem type for sorting
    proj_stmt = select(Project).filter(Project.id == project_id)
    proj_res = await db.execute(proj_stmt)
    project = proj_res.scalars().first()
    is_classification = project and "classification" in str(project.problem_type or "")

    def get_sort_key(m):
        metrics = m.metrics or {}
        if is_classification:
            return metrics.get("f1_score", metrics.get("accuracy", 0))
        return -metrics.get("mse", metrics.get("rmse", float("inf")))

    sorted_models = sorted(models, key=get_sort_key, reverse=True)
    return [TrainedModelResponse.model_validate(m) for m in sorted_models]


@router.get("/models/{model_id}", response_model=TrainedModelResponse)
async def get_model_details(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get full details, hyperparameters, and metrics of a trained model."""
    stmt = select(TrainedModel).filter(TrainedModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return TrainedModelResponse.model_validate(model)


@router.get("/models/{model_id}/evaluation")
async def get_model_evaluation(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 8: Comprehensive Model Evaluation Report (Confusion Matrix, ROC, Classification Report)."""
    stmt = select(TrainedModel).filter(TrainedModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    explainability = model.explainability or {}
    return {
        "model_id": model.id,
        "name": model.name,
        "algorithm": model.algorithm_name,
        "metrics": model.metrics,
        "confusion_matrix": explainability.get("confusion_matrix"),
        "classification_report": explainability.get("classification_report"),
        "roc_curve": explainability.get("roc_curve"),
        "per_class": explainability.get("per_class"),
        "residuals": explainability.get("residuals"),
    }


@router.get("/models/{model_id}/explain")
async def get_model_explainability(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 8: Model Explainability (Feature Importance, SHAP Summary, Decision Insights)."""
    stmt = select(TrainedModel).filter(TrainedModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    explainability = model.explainability or {}
    feature_importance = model.feature_importance or explainability.get("feature_importance", {})

    return {
        "model_id": model.id,
        "name": model.name,
        "algorithm": model.algorithm_name,
        "feature_importance": feature_importance,
        "shap_summary": explainability.get("shap_summary", []),
        "decision_rules": explainability.get("decision_rules", []),
    }


@router.post("/models/{model_id}/set-best")
async def set_best_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Designate a specific model as the champion / best model for deployment."""
    stmt = select(TrainedModel).filter(TrainedModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Clear is_best from other models in project
    all_stmt = select(TrainedModel).filter(TrainedModel.project_id == model.project_id)
    all_res = await db.execute(all_stmt)
    for m in all_res.scalars().all():
        m.is_best = (m.id == model_id)

    await db.commit()
    return {"status": "success", "message": f"Model {model.name} designated as best model."}
