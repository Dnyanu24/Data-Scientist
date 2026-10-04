"""Project Management & Business Goal Understanding API endpoints."""
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.api.deps import get_db, get_current_user
from app.models.models import Project, Dataset, User, PipelinePhase, ProblemType
from app.schemas.schemas import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListResponse,
    GoalUnderstandingRequest, GoalUnderstandingResponse
)
from app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """List all projects owned by current user."""
    stmt = select(Project).filter(Project.owner_id == current_user.id).order_by(Project.created_at.desc())
    result = await db.execute(stmt)
    projects = result.scalars().all()
    return ProjectListResponse(
        projects=[ProjectResponse.model_validate(p) for p in projects],
        total=len(projects)
    )


@router.post("/", response_model=ProjectResponse)
async def create_project(
    project_in: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Create a new Data Science project."""
    project = Project(
        name=project_in.name,
        description=project_in.description,
        business_goal=project_in.business_goal,
        target_column=project_in.target_column,
        phase=PipelinePhase.CREATED,
        phase_progress=0.0,
        phase_message="Project created. Waiting for dataset upload.",
        owner_id=current_user.id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return ProjectResponse.model_validate(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get project by ID."""
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse.model_validate(project)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_in: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Update project configuration, business goal, target column, etc."""
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = project_in.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        if field == "problem_type" and val:
            try:
                setattr(project, field, ProblemType(val))
            except ValueError:
                setattr(project, field, val)
        else:
            setattr(project, field, val)

    await db.commit()
    await db.refresh(project)
    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Delete project and associated resources."""
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await db.delete(project)
    await db.commit()
    return {"status": "success", "message": f"Project {project_id} deleted"}


@router.post("/{project_id}/understand-goal", response_model=GoalUnderstandingResponse)
async def understand_business_goal(
    project_id: int,
    request: GoalUnderstandingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 1: Intelligent Business Goal Understanding using Hybrid LLM + NLP Heuristics.
    
    Translates free-form business objectives into machine learning tasks,
    recommends problem type, optimal target variable, primary evaluation metric,
    and business success criteria.
    """
    stmt = select(Project).filter(Project.id == project_id, Project.owner_id == current_user.id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    goal_text = request.business_goal or project.business_goal or ""
    if not goal_text.strip():
        raise HTTPException(status_code=400, detail="Business goal text is required.")

    # Fetch available columns from the project's dataset if uploaded
    ds_stmt = select(Dataset).filter(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    ds_res = await db.execute(ds_stmt)
    dataset = ds_res.scalars().first()
    available_columns = []
    if dataset and dataset.columns_info:
        available_columns = [col.get("name") for col in dataset.columns_info if "name" in col]

    # Heuristic fallback keywords
    goal_lower = goal_text.lower()
    inferred_problem_type = "binary_classification"
    inferred_metric = "f1_score"
    candidate_target = ""

    if any(k in goal_lower for k in ["how much", "price", "cost", "revenue", "sales", "forecast", "amount", "continuous"]):
        inferred_problem_type = "regression"
        inferred_metric = "r2_score"
    elif any(k in goal_lower for k in ["cluster", "segment", "group", "unsupervised"]):
        inferred_problem_type = "clustering"
        inferred_metric = "silhouette_score"
    elif any(k in goal_lower for k in ["multi", "category", "rating", "sentiment", "grade"]):
        inferred_problem_type = "multiclass_classification"
        inferred_metric = "weighted_f1"

    # Match target column candidate from available columns
    for col in available_columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ["churn", "target", "label", "outcome", "survived", "default", "status", "price", "class"]):
            candidate_target = col
            break
    if not candidate_target and available_columns:
        candidate_target = available_columns[-1]

    # Use LLM for in-depth reasoning
    prompt = f"""You are an elite Lead AI Data Scientist.
Analyze the following natural-language business goal and optional dataset columns:

Business Goal: "{goal_text}"
Available Dataset Columns: {available_columns}

Determine:
1. problem_type: Choose strictly from ["binary_classification", "multiclass_classification", "regression", "clustering", "time_series"]
2. recommended_target_column: The best matching column name from the dataset (or empty string if none)
3. recommended_metric: Primary evaluation metric (e.g. "f1_score", "roc_auc", "rmse", "r2_score")
4. business_objective_summary: Concise 1-2 sentence distillation of the business value
5. business_success_criteria: Measurable criteria for success
6. key_risks_and_considerations: List of 2-4 critical considerations (e.g. class imbalance, data leakage, false positive penalty)

Respond strictly in JSON format."""

    llm_result = await llm_service.generate_json(prompt)

    problem_type = llm_result.get("problem_type") or inferred_problem_type
    target_column = llm_result.get("recommended_target_column") or candidate_target
    metric = llm_result.get("recommended_metric") or inferred_metric
    summary = llm_result.get("business_objective_summary") or f"Automatically optimize predictive models for {goal_text[:60]}"
    criteria = llm_result.get("business_success_criteria") or f"Maximize {metric} with robust generalization"
    considerations = llm_result.get("key_risks_and_considerations") or [
        "Monitor for potential target leakage in historical features",
        "Calibrate prediction thresholds based on business cost-benefit trade-offs",
        "Establish baseline performance using simple linear/tree models"
    ]

    # Update project with inferred attributes
    project.business_goal = goal_text
    try:
        project.problem_type = ProblemType(problem_type)
    except Exception:
        pass
    if target_column:
        project.target_column = target_column

    await db.commit()
    await db.refresh(project)

    return GoalUnderstandingResponse(
        problem_type=problem_type,
        recommended_target_column=target_column,
        recommended_metric=metric,
        business_objective_summary=summary,
        business_success_criteria=criteria,
        key_risks_and_considerations=considerations,
    )
