"""Hybrid Algorithm Recommendation & Adaptive Preprocessing API."""
from typing import Any, List, Dict
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.models import (
    Project, AlgorithmRecommendation, PreprocessingConfig,
    BackgroundTask, User, PipelinePhase
)
from app.schemas.schemas import (
    AlgorithmRecommendationResponse, PreprocessingConfigResponse
)
from app.workers.dispatcher import dispatch_task
from app.workers.tasks import recommend_algorithms_task, preprocess_dataset_task

router = APIRouter()


@router.post("/projects/{project_id}/recommend")
async def trigger_algorithm_recommendation(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 5: Trigger Hybrid AI Algorithm Recommendation Engine.
    
    Synthesizes meta-learning from similar past experiments, rule-based heuristics
    on dataset properties, and LLM reasoning to rank candidate ML models.
    """
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    task = BackgroundTask(
        project_id=project_id,
        task_type="recommendation",
        status="pending",
        progress=0.0,
        message="Running hybrid algorithm recommendation...",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    dispatch_task(recommend_algorithms_task, project_id, task.id)

    return {
        "status": "started",
        "task_id": task.id,
        "message": "Algorithm recommendation started in background."
    }


@router.get("/projects/{project_id}/recommendations", response_model=List[AlgorithmRecommendationResponse])
async def get_algorithm_recommendations(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get ranked list of recommended algorithms with score breakdown and reasoning."""
    stmt = select(AlgorithmRecommendation).filter(
        AlgorithmRecommendation.project_id == project_id
    ).order_by(AlgorithmRecommendation.recommendation_score.desc())
    result = await db.execute(stmt)
    recommendations = result.scalars().all()
    return [AlgorithmRecommendationResponse.model_validate(r) for r in recommendations]


@router.post("/projects/{project_id}/preprocessing/generate")
async def trigger_preprocessing_pipeline(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 6: Trigger Adaptive Preprocessing Pipeline Builder.
    
    Dynamically constructs and applies transformations (imputation, encoding,
    scaling, outlier clipping, feature selection) tailored to dataset traits.
    """
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    task = BackgroundTask(
        project_id=project_id,
        task_type="preprocessing",
        status="pending",
        progress=0.0,
        message="Generating and executing adaptive preprocessing pipeline...",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    dispatch_task(preprocess_dataset_task, project_id, task.id)

    return {
        "status": "started",
        "task_id": task.id,
        "message": "Preprocessing pipeline builder started in background."
    }


@router.get("/projects/{project_id}/preprocessing", response_model=PreprocessingConfigResponse)
async def get_preprocessing_config(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get the adaptive preprocessing configuration and generated transformation steps."""
    stmt = select(PreprocessingConfig).filter(PreprocessingConfig.project_id == project_id)
    result = await db.execute(stmt)
    config = result.scalars().first()
    if not config:
        raise HTTPException(status_code=404, detail="Preprocessing pipeline has not been generated yet.")
    return PreprocessingConfigResponse.model_validate(config)


@router.put("/projects/{project_id}/preprocessing", response_model=PreprocessingConfigResponse)
async def update_preprocessing_config(
    project_id: int,
    steps: List[Dict[str, Any]] = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Allow user customization of pipeline transformation steps."""
    stmt = select(PreprocessingConfig).filter(PreprocessingConfig.project_id == project_id)
    result = await db.execute(stmt)
    config = result.scalars().first()
    if not config:
        raise HTTPException(status_code=404, detail="Preprocessing pipeline not found.")

    config.pipeline_steps = steps
    await db.commit()
    await db.refresh(config)
    return PreprocessingConfigResponse.model_validate(config)
