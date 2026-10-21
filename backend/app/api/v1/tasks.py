"""Background Task Progress & Status API."""
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from datetime import datetime
from app.models.models import BackgroundTask, User, Project, PipelinePhase
from app.schemas.schemas import BackgroundTaskResponse

router = APIRouter()

# Map task types to pipeline phases upon completion
TASK_PHASE_MAP = {
    "profiling": PipelinePhase.PROFILED.value,
    "fingerprinting": PipelinePhase.FINGERPRINTED.value,
    "recommendation": PipelinePhase.RECOMMENDED.value,
    "recommend": PipelinePhase.RECOMMENDED.value,
    "meta_learning": PipelinePhase.RECOMMENDED.value,
    "preprocessing": PipelinePhase.PREPROCESSED.value,
    "training": PipelinePhase.DEPLOYED.value,
}


@router.get("/{task_id}", response_model=BackgroundTaskResponse)
async def get_task_status(
    task_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get background task execution status and progress (0.0 to 1.0)."""
    stmt = select(BackgroundTask).filter(BackgroundTask.id == task_id)
    result = await db.execute(stmt)
    task = result.scalars().first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    project_phase = None
    if task.project_id:
        proj_stmt = select(Project).filter(Project.id == task.project_id)
        proj_result = await db.execute(proj_stmt)
        project = proj_result.scalars().first()
        if project:
            if task.status == "completed":
                target_phase = TASK_PHASE_MAP.get(task.task_type)
                if target_phase and project.phase != target_phase:
                    # Do not downgrade project phase if already deployed
                    if not (str(project.phase) == PipelinePhase.DEPLOYED.value):
                        project.phase = target_phase
                        project.phase_progress = 1.0
                        if task.message:
                            project.phase_message = task.message
                        project.updated_at = datetime.utcnow()
                        await db.commit()
                        await db.refresh(project)
            project_phase = project.phase.value if hasattr(project.phase, "value") else str(project.phase)

    data = BackgroundTaskResponse.model_validate(task).model_dump()
    data["project_phase"] = project_phase
    return BackgroundTaskResponse(**data)


@router.get("/project/{project_id}", response_model=List[BackgroundTaskResponse])
async def list_project_tasks(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """List recent background tasks for a project."""
    stmt = select(BackgroundTask).filter(
        BackgroundTask.project_id == project_id
    ).order_by(BackgroundTask.created_at.desc()).limit(20)
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    return [BackgroundTaskResponse.model_validate(t) for t in tasks]
