"""Continuous Learning & Knowledge Base API."""
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api.deps import get_db, get_current_user
from app.models.models import ExperienceRecord, AlgorithmKnowledge, User, TrainedModel, Project
from app.services.seed_data import seed_all

router = APIRouter()


@router.get("/experiences")
async def list_experiences(
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 11: Explore Historical Meta-Learning Experiences & Learned Patterns."""
    stmt = select(ExperienceRecord).order_by(ExperienceRecord.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        {
            "id": r.id,
            "project_id": r.project_id,
            "problem_type": r.problem_type,
            "dataset_size": r.dataset_size,
            "num_features": r.num_features,
            "feature_types": r.feature_types,
            "data_quality_score": r.data_quality_score,
            "best_algorithm": r.best_algorithm,
            "best_metrics": r.best_metrics,
            "best_hyperparameters": r.best_hyperparameters,
            "preprocessing_steps": r.preprocessing_steps,
            "all_results": r.all_results,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@router.get("/algorithms")
async def list_algorithms(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Explore Algorithm Knowledge Base with capabilities and performance traits."""
    stmt = select(AlgorithmKnowledge).order_by(AlgorithmKnowledge.category, AlgorithmKnowledge.display_name)
    result = await db.execute(stmt)
    algos = result.scalars().all()
    return [
        {
            "id": a.id,
            "name": a.name,
            "display_name": a.display_name,
            "category": a.category,
            "library": a.library,
            "supports_classification": a.supports_classification,
            "supports_regression": a.supports_regression,
            "supports_multiclass": a.supports_multiclass,
            "handles_missing": a.handles_missing,
            "handles_categorical": a.handles_categorical,
            "supports_feature_importance": a.supports_feature_importance,
            "is_ensemble": a.is_ensemble,
            "training_speed": a.training_speed,
            "prediction_speed": a.prediction_speed,
            "memory_usage": a.memory_usage,
            "scalability": a.scalability,
            "description": a.description,
            "strengths": a.strengths,
            "weaknesses": a.weaknesses,
        }
        for a in algos
    ]


@router.get("/stats")
async def get_knowledge_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Continuous learning repository summary statistics."""
    exp_count = (await db.execute(select(func.count(ExperienceRecord.id)))).scalar_one()
    algo_count = (await db.execute(select(func.count(AlgorithmKnowledge.id)))).scalar_one()
    model_count = (await db.execute(select(func.count(TrainedModel.id)))).scalar_one()
    proj_count = (await db.execute(select(func.count(Project.id)))).scalar_one()

    return {
        "historical_experiments": exp_count,
        "known_algorithms": algo_count,
        "total_models_trained": model_count,
        "total_projects": proj_count,
        "learning_status": "active",
    }


@router.post("/seed")
async def trigger_seed(
    current_user: User = Depends(get_current_user),
) -> Any:
    """Manually trigger or refresh knowledge base seed experiences."""
    seed_all()
    return {"status": "success", "message": "Knowledge base seeded successfully."}
