from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

def _is_sqlite(url: str) -> bool:
    return "sqlite" in url.lower()

from sqlalchemy import create_engine, event

# Async engine for FastAPI
async_kwargs = {"echo": False}
if _is_sqlite(settings.DATABASE_URL):
    async_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60}
else:
    async_kwargs["pool_pre_ping"] = True
    async_kwargs["pool_size"] = 20
    async_kwargs["max_overflow"] = 10

async_engine = create_async_engine(
    settings.DATABASE_URL,
    **async_kwargs
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for background workers and initialization
sync_kwargs = {"echo": False}
if _is_sqlite(settings.DATABASE_URL_SYNC):
    sync_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60}
else:
    sync_kwargs["pool_pre_ping"] = True
    sync_kwargs["pool_size"] = 20
    sync_kwargs["max_overflow"] = 10

sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    **sync_kwargs
)

SyncSessionLocal = sessionmaker(bind=sync_engine)


@event.listens_for(sync_engine, "connect")
def _set_sqlite_sync_pragma(dbapi_connection, connection_record):
    if _is_sqlite(settings.DATABASE_URL_SYNC):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()


@event.listens_for(async_engine.sync_engine, "connect")
def _set_sqlite_async_pragma(dbapi_connection, connection_record):
    if _is_sqlite(settings.DATABASE_URL):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()



class Base(DeclarativeBase):
    pass


def init_db():
    """Create all database tables synchronously."""
    Base.metadata.create_all(bind=sync_engine)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db():
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()
