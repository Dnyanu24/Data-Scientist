"""Drift Monitoring & Model Observability API."""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.models import (
    Project, Dataset, TrainedModel, Deployment, Prediction,
    DriftRecord, User
)
from app.ml.monitoring import drift_monitor

router = APIRouter()


@router.get("/deployments/{deployment_id}")
async def get_deployment_monitoring(
    deployment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 9 & 11: Real-time Drift Monitoring & Observability Dashboard."""
    stmt = select(Deployment).filter(Deployment.id == deployment_id)
    result = await db.execute(stmt)
    deployment = result.scalars().first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # Fetch drift records
    drift_stmt = select(DriftRecord).filter(
        DriftRecord.deployment_id == deployment_id
    ).order_by(DriftRecord.created_at.desc()).limit(20)
    drift_res = await db.execute(drift_stmt)
    drift_records = drift_res.scalars().all()

    # Latency & throughput stats
    pred_stmt = select(Prediction).filter(
        Prediction.deployment_id == deployment_id
    ).order_by(Prediction.created_at.desc()).limit(100)
    pred_res = await db.execute(pred_stmt)
    predictions = pred_res.scalars().all()

    latencies = [p.latency_ms for p in predictions if p.latency_ms is not None]
    avg_lat = round(float(np.mean(latencies)), 2) if latencies else 0.0
    p95_lat = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    p99_lat = round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0

    return {
        "deployment_id": deployment_id,
        "endpoint_name": deployment.endpoint_name,
        "status": deployment.status,
        "request_count": deployment.request_count or len(predictions),
        "latency": {
            "avg_ms": avg_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
        },
        "last_prediction_at": deployment.last_prediction_at.isoformat() if deployment.last_prediction_at else None,
        "drift_records": [
            {
                "id": d.id,
                "drift_type": d.drift_type,
                "feature_name": d.feature_name,
                "drift_score": d.drift_score,
                "is_drifted": d.is_drifted,
                "details": d.details,
                "created_at": d.created_at.isoformat(),
            }
            for d in drift_records
        ],
    }


@router.post("/deployments/{deployment_id}/check-drift")
async def trigger_drift_check(
    deployment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Trigger statistical drift analysis comparing recent inference logs against baseline training data."""
    stmt = select(Deployment).filter(Deployment.id == deployment_id)
    result = await db.execute(stmt)
    deployment = result.scalars().first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # Fetch reference dataset
    ds_stmt = select(Dataset).filter(Dataset.project_id == deployment.project_id).order_by(Dataset.created_at.desc())
    ds_res = await db.execute(ds_stmt)
    dataset = ds_res.scalars().first()
    if not dataset or not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=400, detail="Reference training data not available.")

    # Load baseline
    ref_df = pd.read_csv(dataset.file_path) if dataset.file_type == "csv" else pd.read_excel(dataset.file_path)

    # Fetch recent predictions
    pred_stmt = select(Prediction).filter(
        Prediction.deployment_id == deployment_id
    ).order_by(Prediction.created_at.desc()).limit(500)
    pred_res = await db.execute(pred_stmt)
    predictions = pred_res.scalars().all()

    if len(predictions) < 5:
        return {
            "status": "skipped",
            "message": "Insufficient inference logs (minimum 5 predictions required for drift test).",
            "predictions_count": len(predictions),
        }

    # Extract current features
    cur_records = []
    for p in predictions:
        if isinstance(p.input_data, dict):
            cur_records.append(p.input_data)
        elif isinstance(p.input_data, list) and p.input_data:
            cur_records.extend(p.input_data)

    cur_df = pd.DataFrame(cur_records)
    common_features = [col for col in cur_df.columns if col in ref_df.columns]

    if not common_features:
        return {"status": "skipped", "message": "No overlapping features between baseline and current data."}

    drift_report = drift_monitor.detect_data_drift(ref_df, cur_df, common_features)

    # Persist drift records
    for feat, f_info in drift_report.get("features", {}).items():
        rec = DriftRecord(
            deployment_id=deployment_id,
            drift_type="data_drift",
            feature_name=feat,
            drift_score=f_info.get("p_value", 0.0),
            is_drifted=f_info.get("drift_detected", False),
            details=f_info,
            created_at=datetime.utcnow(),
        )
        db.add(rec)

    await db.commit()

    return {
        "status": "completed",
        "deployment_id": deployment_id,
        "sample_size": len(cur_df),
        "overall_drift": drift_report.get("overall_drift", False),
        "drift_score": drift_report.get("drift_score", 0.0),
        "feature_details": drift_report.get("features", {}),
    }
