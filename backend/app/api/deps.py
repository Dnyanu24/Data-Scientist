"""API dependencies including database sessions and JWT authentication."""
from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.database.session import AsyncSessionLocal, SyncSessionLocal
from app.models.models import User

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"/api/v1/auth/login",
    auto_error=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db():
    """Dependency that yields a sync database session."""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Validate JWT token and return the authenticated user.
    
    If no token provided, creates or retrieves a default developer user
    so that local testing and frontend exploration work seamlessly without friction.
    """
    if not token:
        # Development / Guest fallback user
        result = await db.execute(select(User).filter(User.username == "admin"))
        user = result.scalars().first()
        if not user:
            from app.core.security import get_password_hash
            user = User(
                email="admin@aidatascientist.io",
                username="admin",
                full_name="AI Data Scientist Admin",
                hashed_password=get_password_hash("admin123"),
                is_active=True,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except (JWTError, ValidationError):
        raise credentials_exception

    result = await db.execute(select(User).filter(User.id == int(user_id)))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user
