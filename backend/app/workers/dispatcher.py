"""Task Dispatcher with graceful Celery and ThreadPool fallback."""
import concurrent.futures
import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global background thread pool for standalone / local execution
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml_worker")


def _execute_task_eager(celery_task: Any, args: tuple, kwargs: dict):
    try:
        if hasattr(celery_task, "apply"):
            return celery_task.apply(args=args, kwargs=kwargs)
        return celery_task(*args, **kwargs)
    except Exception as exc:
        logger.exception(f"Background task {celery_task} failed: {exc}")
        raise


def dispatch_task(celery_task: Any, *args: Any, **kwargs: Any):
    """Dispatch a background task.
    
    Tries Celery first if USE_CELERY is True, otherwise transparently
    executes in a managed background thread pool with identical DB progress tracking.
    """
    if getattr(settings, "USE_CELERY", False):
        try:
            return celery_task.delay(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"Celery broker unavailable ({exc}). Falling back to local background thread.")

    return _executor.submit(_execute_task_eager, celery_task, args, kwargs)
