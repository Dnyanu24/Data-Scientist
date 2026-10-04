"""Intelligent User Interaction & AI Data Scientist Chat Assistant API."""
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_user
from app.models.models import Project, Dataset, TrainedModel, User
from app.schemas.schemas import ChatRequest, ChatResponse
from app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()


@router.post("/", response_model=ChatResponse)
async def chat_with_data_scientist(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Phase 10: Conversational AI Data Scientist Assistant.
    
    Provides interactive explanations, statistical insights, model recommendations,
    and Python code snippets grounded in the active project's dataset and models.
    """
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Gather project context if project_id is provided
    context_str = ""
    if request.project_id:
        proj_stmt = select(Project).filter(Project.id == request.project_id)
        proj_res = await db.execute(proj_stmt)
        project = proj_res.scalars().first()
        if project:
            context_str += f"Project Name: {project.name}\n"
            context_str += f"Business Goal: {project.business_goal}\n"
            context_str += f"Problem Type: {project.problem_type}\n"
            context_str += f"Target Column: {project.target_column}\n"
            context_str += f"Current Phase: {project.phase}\n"

            # Check dataset
            ds_stmt = select(Dataset).filter(Dataset.project_id == project.id).order_by(Dataset.created_at.desc())
            ds_res = await db.execute(ds_stmt)
            dataset = ds_res.scalars().first()
            if dataset:
                context_str += f"Dataset: {dataset.original_filename} ({dataset.num_rows} rows, {dataset.num_columns} cols)\n"
                if dataset.data_quality:
                    context_str += f"Data Quality Score: {dataset.data_quality.get('scores', {}).get('overall')}\n"

            # Check best model
            best_stmt = select(TrainedModel).filter(TrainedModel.project_id == project.id, TrainedModel.is_best == True)
            best_res = await db.execute(best_stmt)
            best_model = best_res.scalars().first()
            if best_model:
                context_str += f"Best Model: {best_model.algorithm_name} (Metrics: {best_model.metrics})\n"

    system_prompt = (
        "You are 'Antigravity AI Data Scientist', a world-class principal data scientist and ML engineer. "
        "You help users analyze datasets, understand model decisions, troubleshoot ML pipelines, and write production-grade code. "
        "Be insightful, authoritative, concise, and mathematically sound."
    )

    prompt = f"""User Inquiry: "{user_msg}"

Active Project Context:
{context_str if context_str else "No active project loaded."}

Provide:
1. Clear, insightful explanation and guidance
2. 2-3 follow-up suggested questions or actions
3. If relevant, a clean, self-contained Python code block

Format response strictly as JSON with keys:
- "response": text explanation (in markdown)
- "suggestions": list of 2-3 short suggested questions or next steps
- "code_snippet": python code string or null if not applicable"""

    result = await llm_service.generate_json(prompt, system_prompt=system_prompt)

    response_text = result.get("response")
    if not response_text:
        # Fallback response
        response_text = (
            f"As your AI Data Scientist, I've analyzed your question regarding '{user_msg[:60]}'. "
            f"Based on your current pipeline state, ensure proper feature scaling and handle any potential class imbalance "
            f"before finalizing your deployment. Let me know if you would like me to inspect feature correlations or train ensembles."
        )

    suggestions = result.get("suggestions") or [
        "What features have the highest predictive power?",
        "How can we improve precision without sacrificing recall?",
        "Should we create a stacking ensemble for this dataset?"
    ]
    code_snippet = result.get("code_snippet")

    return ChatResponse(
        response=response_text,
        suggestions=suggestions,
        code_snippet=code_snippet,
    )
