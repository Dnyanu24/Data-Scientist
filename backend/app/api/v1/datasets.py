"""Dataset Ingestion, Intelligent Understanding, Fingerprinting & Similarity API."""
import os
import shutil
import uuid
from typing import Any, List, Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.core.config import settings
from app.models.models import (
    Project, Dataset, DatasetFingerprint, BackgroundTask,
    ExperienceRecord, User, PipelinePhase
)
from app.schemas.schemas import DatasetResponse, DatasetFingerprintResponse
from app.workers.dispatcher import dispatch_task
from app.workers.tasks import profile_dataset_task, generate_fingerprint_task
from app.ml.fingerprinting import fingerprinter

router = APIRouter()


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 1: Ingest dataset (CSV or Excel) and perform initial inspection."""
    # Verify project
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx, .xls) files are supported.")

    unique_filename = f"{project_id}_{uuid.uuid4().hex[:8]}_{filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    # Read preview and determine schema
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, nrows=5000)
        else:
            df = pd.read_excel(file_path, nrows=5000)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse data file: {str(e)}")

    num_rows = len(df)
    num_columns = len(df.columns)

    # Column summary info
    columns_info = []
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        nunique = int(series.nunique(dropna=True))
        missing = int(series.isna().sum())
        sample_vals = series.dropna().head(3).tolist()

        columns_info.append({
            "name": str(col),
            "dtype": dtype_str,
            "nunique": nunique,
            "missing": missing,
            "missing_pct": round(missing / max(len(series), 1) * 100, 2),
            "sample": [str(v) for v in sample_vals],
        })

    dataset = Dataset(
        project_id=project_id,
        filename=unique_filename,
        original_filename=filename,
        file_path=file_path,
        file_size=file_size,
        file_type="csv" if ext == ".csv" else "xlsx",
        num_rows=num_rows,
        num_columns=num_columns,
        columns_info=columns_info,
    )
    db.add(dataset)

    # Update project phase
    project.phase = PipelinePhase.UPLOADED
    project.phase_progress = 0.1
    project.phase_message = f"Dataset '{filename}' uploaded ({num_rows} rows, {num_columns} columns)"

    # Automatically set default target column if not set
    if not project.target_column and columns_info:
        for c in columns_info:
            c_name = c["name"].lower()
            if any(term in c_name for term in ["target", "label", "churn", "survived", "price", "outcome"]):
                project.target_column = c["name"]
                break
        if not project.target_column:
            project.target_column = columns_info[-1]["name"]

    await db.commit()
    await db.refresh(dataset)
    return DatasetResponse.model_validate(dataset)


@router.get("/project/{project_id}", response_model=List[DatasetResponse])
async def list_project_datasets(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """List all datasets for a project."""
    stmt = select(Dataset).filter(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    result = await db.execute(stmt)
    datasets = result.scalars().all()
    return [DatasetResponse.model_validate(d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get dataset metadata."""
    stmt = select(Dataset).filter(Dataset.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get paginated row preview of dataset."""
    stmt = select(Dataset).filter(Dataset.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
    if not dataset or not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset or file not found")

    skip = (page - 1) * page_size
    if dataset.file_type == "csv":
        df = pd.read_csv(dataset.file_path, skiprows=range(1, skip + 1), nrows=page_size)
    else:
        df = pd.read_excel(dataset.file_path, skiprows=range(1, skip + 1), nrows=page_size)

    # Clean NaN for JSON serialization
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    return {
        "dataset_id": dataset_id,
        "page": page,
        "page_size": page_size,
        "total_rows": dataset.num_rows,
        "columns": list(df.columns),
        "data": records,
    }


@router.post("/{dataset_id}/profile")
async def trigger_profiling(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 2: Trigger Intelligent Data Profiling & Quality Analysis."""
    stmt = select(Dataset).filter(Dataset.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task = BackgroundTask(
        project_id=dataset.project_id,
        task_type="profiling",
        status="pending",
        progress=0.0,
        message="Profiling dataset queued...",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Dispatch task
    dispatch_task(profile_dataset_task, dataset.project_id, dataset.id, task.id)

    return {"status": "started", "task_id": task.id, "message": "Profiling task started in background."}


@router.get("/{dataset_id}/profile")
async def get_profiling_results(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get full profiling results, schema detection, and data quality metrics."""
    stmt = select(Dataset).filter(Dataset.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "dataset_id": dataset_id,
        "is_profiled": bool(dataset.profile_report),
        "profile_report": dataset.profile_report,
        "data_quality": dataset.data_quality,
        "target_analysis": dataset.target_analysis,
        "schema_info": dataset.schema_info,
    }


@router.post("/{dataset_id}/fingerprint")
async def trigger_fingerprinting(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 3: Trigger Dataset Fingerprinting (128-dim statistical embedding)."""
    stmt = select(Dataset).filter(Dataset.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task = BackgroundTask(
        project_id=dataset.project_id,
        task_type="fingerprinting",
        status="pending",
        progress=0.0,
        message="Fingerprinting queued...",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    dispatch_task(generate_fingerprint_task, dataset.project_id, dataset.id, task.id)

    return {"status": "started", "task_id": task.id, "message": "Fingerprinting task started in background."}


@router.get("/{dataset_id}/fingerprint", response_model=DatasetFingerprintResponse)
async def get_fingerprint(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get dataset fingerprint and meta-features."""
    stmt = select(DatasetFingerprint).filter(DatasetFingerprint.dataset_id == dataset_id)
    result = await db.execute(stmt)
    fp = result.scalars().first()
    if not fp:
        raise HTTPException(status_code=404, detail="Dataset has not been fingerprinted yet.")
    return DatasetFingerprintResponse.model_validate(fp)


@router.get("/{dataset_id}/similar")
async def get_similar_historical_datasets(
    dataset_id: int,
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 4: Experience Learning & Meta-Learning.
    
    Searches the knowledge base of historical experiments using fingerprint embedding
    cosine similarity to find the most similar datasets, which algorithms worked best,
    and recommended hyperparameter priors.
    """
    stmt = select(DatasetFingerprint).filter(DatasetFingerprint.dataset_id == dataset_id)
    result = await db.execute(stmt)
    fp = result.scalars().first()
    if not fp or not fp.embedding:
        raise HTTPException(status_code=400, detail="Fingerprint embedding not available. Run fingerprinting first.")

    target_embedding = list(fp.embedding)

    # Query all historical experience records
    exp_stmt = select(ExperienceRecord)
    exp_res = await db.execute(exp_stmt)
    experiences = exp_res.scalars().all()

    similarities = []
    for exp in experiences:
        if exp.fingerprint_embedding:
            sim = fingerprinter.compute_similarity(target_embedding, list(exp.fingerprint_embedding))
            similarities.append({
                "id": exp.id,
                "problem_type": exp.problem_type,
                "dataset_size": exp.dataset_size,
                "num_features": exp.num_features,
                "similarity_score": round(float(sim), 4),
                "best_algorithm": exp.best_algorithm,
                "best_metrics": exp.best_metrics,
                "best_hyperparameters": exp.best_hyperparameters,
                "preprocessing_steps": exp.preprocessing_steps,
                "all_results": exp.all_results,
            })

    # Sort descending by similarity
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_matches = similarities[:top_k]

    return {
        "dataset_id": dataset_id,
        "total_historical_records_searched": len(experiences),
        "similar_datasets": top_matches,
    }
