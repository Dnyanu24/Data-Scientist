"""Background Task Progress & Status API."""
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.models import BackgroundTask, User
from app.schemas.schemas import BackgroundTaskResponse

router = APIRouter()


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
    return BackgroundTaskResponse.model_validate(task)


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
