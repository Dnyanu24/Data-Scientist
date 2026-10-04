from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash safely."""
    try:
        if isinstance(plain_password, str):
            p_bytes = plain_password.encode("utf-8")[:72]
        else:
            p_bytes = bytes(plain_password)[:72]

        if isinstance(hashed_password, str):
            h_bytes = hashed_password.encode("utf-8")
        else:
            h_bytes = bytes(hashed_password)

        return bcrypt.checkpw(p_bytes, h_bytes)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt safely."""
    if isinstance(password, str):
        p_bytes = password.encode("utf-8")[:72]
    else:
        p_bytes = bytes(password)[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(p_bytes, salt).decode("utf-8")


def create_access_token(data: Union[Dict[str, Any], int, str], expires_delta: Optional[timedelta] = None) -> str:
    """Generate JWT access token."""
    if isinstance(data, (int, str)):
        to_encode = {"sub": str(data)}
    else:
        to_encode = data.copy()

    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: Union[Dict[str, Any], int, str]) -> str:
    """Generate JWT refresh token."""
    if isinstance(data, (int, str)):
        to_encode = {"sub": str(data)}
    else:
        to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
