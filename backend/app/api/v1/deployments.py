"""Model Deployment, Real-Time Prediction & Feedback API."""
import time
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, List, Dict
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.models import (
    Project, TrainedModel, Deployment, Prediction, PredictionFeedback,
    User, DeploymentStatus, ModelStatus
)
from app.schemas.schemas import (
    DeploymentResponse, PredictionCreate, PredictionResponse,
    PredictionFeedbackCreate, PredictionFeedbackResponse
)

from app.ml.preprocessing import PreprocessingPipeline

router = APIRouter()

# Cache loaded models in memory for high-speed inference
_MODEL_CACHE: Dict[int, Any] = {}
_PIPELINE_CACHE: Dict[int, Any] = {}


def _get_loaded_model_and_pipeline(model: TrainedModel):
    """Load model and pipeline from disk or return cached instance."""
    model_id = model.id
    if model_id not in _MODEL_CACHE:
        if not model.model_path or not os.path.exists(model.model_path):
            raise HTTPException(status_code=500, detail=f"Model artifact not found at {model.model_path}")
        with open(model.model_path, "rb") as f:
            _MODEL_CACHE[model_id] = pickle.load(f)

    if model_id not in _PIPELINE_CACHE and model.pipeline_path and os.path.exists(model.pipeline_path):
        with open(model.pipeline_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, PreprocessingPipeline):
                _PIPELINE_CACHE[model_id] = data
            elif isinstance(data, dict):
                p = PreprocessingPipeline()
                p.transformers = data.get("transformers", {})
                p.feature_columns = data.get("feature_columns", [])
                p.target_column = data.get("target_column")
                p.label_encoder = data.get("label_encoder")
                p.steps_applied = data.get("steps_applied", [])
                p.steps_config = data.get("steps_config", [])
                _PIPELINE_CACHE[model_id] = p
            else:
                _PIPELINE_CACHE[model_id] = data

    return _MODEL_CACHE[model_id], _PIPELINE_CACHE.get(model_id)



@router.post("/", response_model=DeploymentResponse)
async def deploy_model(
    model_id: int = Body(..., embed=True),
    endpoint_name: str = Body(None, embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 9: Deploy a trained model as an active inference endpoint."""
    stmt = select(TrainedModel).filter(TrainedModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check existing deployment
    existing_stmt = select(Deployment).filter(Deployment.model_id == model_id)
    existing_res = await db.execute(existing_stmt)
    existing = existing_res.scalars().first()
    if existing:
        existing.status = DeploymentStatus.ACTIVE
        await db.commit()
        await db.refresh(existing)
        return DeploymentResponse.model_validate(existing)

    endpoint = endpoint_name or f"endpoint-{model.algorithm_name}-{model.id}"
    deployment = Deployment(
        project_id=model.project_id,
        model_id=model.id,
        endpoint_name=endpoint,
        status=DeploymentStatus.ACTIVE,
        version=model.version or 1,
        config={"algorithm": model.algorithm_name, "deployed_by": current_user.username},
        request_count=0,
        avg_latency_ms=0.0,
    )
    db.add(deployment)
    model.status = ModelStatus.DEPLOYED
    await db.commit()
    await db.refresh(deployment)
    return DeploymentResponse.model_validate(deployment)


@router.get("/", response_model=List[DeploymentResponse])
async def list_deployments(
    project_id: int = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """List deployed endpoints."""
    stmt = select(Deployment)
    if project_id:
        stmt = stmt.filter(Deployment.project_id == project_id)
    result = await db.execute(stmt.order_by(Deployment.created_at.desc()))
    deployments = result.scalars().all()
    return [DeploymentResponse.model_validate(d) for d in deployments]


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get deployment details and serving metrics."""
    stmt = select(Deployment).filter(Deployment.id == deployment_id)
    result = await db.execute(stmt)
    deployment = result.scalars().first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return DeploymentResponse.model_validate(deployment)


@router.delete("/{deployment_id}")
async def deactivate_deployment(
    deployment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Deactivate a deployment endpoint."""
    stmt = select(Deployment).filter(Deployment.id == deployment_id)
    result = await db.execute(stmt)
    deployment = result.scalars().first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment.status = DeploymentStatus.INACTIVE
    await db.commit()
    return {"status": "success", "message": f"Deployment {deployment_id} deactivated."}


@router.post("/{deployment_id}/predict")
async def run_prediction(
    deployment_id: int,
    request: PredictionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 9: Real-time single or batch inference against deployed model."""
    stmt = select(Deployment).filter(Deployment.id == deployment_id)
    result = await db.execute(stmt)
    deployment = result.scalars().first()
    if not deployment or deployment.status != DeploymentStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Deployment is not active or does not exist.")

    # Fetch model
    model_stmt = select(TrainedModel).filter(TrainedModel.id == deployment.model_id)
    model_res = await db.execute(model_stmt)
    model = model_res.scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Associated trained model not found.")

    loaded_model, loaded_pipeline = _get_loaded_model_and_pipeline(model)

    start_time = time.time()
    input_data = request.features

    # Convert to DataFrame
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_input = pd.DataFrame(input_data)
    else:
        raise HTTPException(status_code=400, detail="Input features must be a dict or list of dicts.")

    # Apply preprocessing pipeline if present
    X = df_input
    if loaded_pipeline:
        try:
            if hasattr(loaded_pipeline, "transform"):
                X, _ = loaded_pipeline.transform(df_input)
            elif isinstance(loaded_pipeline, dict):
                p = PreprocessingPipeline()
                p.transformers = loaded_pipeline.get("transformers", {})
                p.feature_columns = loaded_pipeline.get("feature_columns", [])
                p.target_column = loaded_pipeline.get("target_column")
                p.steps_applied = loaded_pipeline.get("steps_applied", [])
                p.steps_config = loaded_pipeline.get("steps_config", [])
                X, _ = p.transform(df_input)
        except Exception:
            X = pd.get_dummies(df_input)
    else:
        X = pd.get_dummies(df_input)

    # Ensure all columns required by the model exist and match expected order
    if hasattr(loaded_model, "feature_names_in_"):
        expected_cols = list(loaded_model.feature_names_in_)
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]
    elif hasattr(loaded_pipeline, "feature_columns") and loaded_pipeline.feature_columns:
        expected_cols = loaded_pipeline.feature_columns
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]


    # Predict
    try:
        raw_pred = loaded_model.predict(X)
        if hasattr(raw_pred, "tolist"):
            pred_list = raw_pred.tolist()
        else:
            pred_list = list(raw_pred)

        probabilities = None
        if hasattr(loaded_model, "predict_proba"):
            try:
                proba = loaded_model.predict_proba(X)
                probabilities = proba.tolist() if hasattr(proba, "tolist") else list(proba)
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Format result
    single = isinstance(input_data, dict)
    final_pred = pred_list[0] if single else pred_list
    final_proba = probabilities[0] if (probabilities and single) else probabilities

    # Record prediction for drift monitoring and feedback
    pred_record = Prediction(
        project_id=deployment.project_id,
        deployment_id=deployment.id,
        input_data=input_data,
        prediction={"prediction": final_pred},
        probability={"probability": final_proba} if final_proba else None,
        latency_ms=latency_ms,
    )
    db.add(pred_record)

    # Update deployment serving metrics
    deployment.request_count = (deployment.request_count or 0) + 1
    prev_avg = deployment.avg_latency_ms or latency_ms
    deployment.avg_latency_ms = round((prev_avg * 0.9) + (latency_ms * 0.1), 2)
    deployment.last_prediction_at = datetime.utcnow()

    await db.commit()
    await db.refresh(pred_record)

    return {
        "prediction_id": pred_record.id,
        "deployment_id": deployment.id,
        "prediction": final_pred,
        "probability": final_proba,
        "latency_ms": latency_ms,
        "timestamp": pred_record.created_at.isoformat(),
    }


@router.get("/{deployment_id}/predictions")
async def list_prediction_history(
    deployment_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get recent prediction history for a deployment endpoint."""
    stmt = select(Prediction).filter(
        Prediction.deployment_id == deployment_id
    ).order_by(Prediction.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        {
            "id": p.id,
            "input_data": p.input_data,
            "prediction": p.prediction,
            "probability": p.probability,
            "latency_ms": p.latency_ms,
            "created_at": p.created_at.isoformat(),
            "has_feedback": bool(p.feedback),
        }
        for p in records
    ]


@router.post("/predictions/{prediction_id}/feedback", response_model=PredictionFeedbackResponse)
async def submit_prediction_feedback(
    prediction_id: int,
    feedback_in: PredictionFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 10 & 11: Continuous Learning User Feedback Loop."""
    stmt = select(Prediction).filter(Prediction.id == prediction_id)
    result = await db.execute(stmt)
    pred = result.scalars().first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction record not found")

    fb = PredictionFeedback(
        prediction_id=prediction_id,
        user_id=current_user.id,
        is_correct=feedback_in.is_correct,
        correct_value=feedback_in.correct_value,
        comment=feedback_in.comment,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return PredictionFeedbackResponse.model_validate(fb)
