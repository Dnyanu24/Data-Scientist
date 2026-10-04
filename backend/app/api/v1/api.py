"""V1 API Router consolidating all lifecycle phase endpoints."""
from fastapi import APIRouter

from app.api.v1 import (
    auth, projects, datasets, recommendations,
    experiments, deployments, monitoring, chat,
    knowledge, tasks
)

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(projects.router, prefix="/projects", tags=["Phase 1: Projects & Goal Understanding"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Phase 2-4: Data Understanding & Fingerprinting"])
api_router.include_router(recommendations.router, tags=["Phase 5-6: Recommendations & Preprocessing"])
api_router.include_router(experiments.router, tags=["Phase 7-8: Training & Evaluation"])
api_router.include_router(deployments.router, prefix="/deployments", tags=["Phase 9: Deployment & Inference"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["Phase 9 & 11: Drift Monitoring"])
api_router.include_router(chat.router, prefix="/chat", tags=["Phase 10: AI Data Scientist Chat Assistant"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["Phase 11: Continuous Learning & Knowledge Base"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["Background Tasks"])
