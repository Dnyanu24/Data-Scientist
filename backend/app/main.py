"""FastAPI Application Main Entry Point."""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.database.session import init_db, SyncSessionLocal
from app.models.models import BackgroundTask
from app.services.seed_data import seed_all
from app.api.v1.api import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_datascientist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    logger.info("Initializing AI Data Scientist backend...")
    # Initialize DB tables
    try:
        init_db()
        logger.info("Database schema initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")

    # Seed Knowledge Base & Meta-learning experiences
    try:
        seed_all()
        logger.info("Knowledge Base & Meta-Learning experiences ready.")
    except Exception as e:
        logger.warning(f"Seeding warning: {e}")

    yield

    logger.info("AI Data Scientist backend shutting down...")


app = FastAPI(
    title="AI Data Scientist API",
    description="Autonomous Full-Stack AI Data Scientist Lifecycle Engine (11 Phases)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development & local pairing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "app": "AI Data Scientist",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "phases": 11,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "database": "connected",
        "celery_enabled": getattr(settings, "USE_CELERY", False),
    }


@app.websocket("/ws/tasks/{task_id}")
async def task_websocket_endpoint(websocket: WebSocket, task_id: int):
    """Real-time task progress streaming over WebSocket."""
    await websocket.accept()
    try:
        while True:
            db = SyncSessionLocal()
            try:
                task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
                if task:
                    payload = {
                        "task_id": task.id,
                        "status": task.status,
                        "progress": task.progress or 0.0,
                        "message": task.message or "",
                        "result": task.result,
                        "error": task.error,
                    }
                    await websocket.send_json(payload)
                    if task.status in ["completed", "failed"]:
                        break
                else:
                    await websocket.send_json({"error": "Task not found", "status": "failed"})
                    break
            finally:
                db.close()

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for task {task_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for task {task_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
