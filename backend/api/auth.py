"""
Authentication and authorization for TTD-DR Framework API
Implements JWT-based authentication with rate limiting
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ttdr-framework-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Simple password hashing for development (not for production)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash (simple comparison for development)"""
    return plain_password == hashed_password

def get_password_hash(password: str) -> str:
    """Hash a password (simple storage for development)"""
    return password

# JWT Bearer token scheme
security = HTTPBearer()

class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: str
    permissions: list[str] = []
    exp: datetime

class User(BaseModel):
    """User model for authentication"""
    id: str
    username: str
    email: str
    is_active: bool = True
    permissions: list[str] = []

# Mock user database (in production, use a real database)
MOCK_USERS = {
    "dev_user": {
        "id": "user_dev",
        "username": "dev_user",
        "email": "dev@ttdr.com",
        "hashed_password": "dev_password",
        "is_active": True,
        "permissions": ["*"]
    },
    "demo_user": {
        "id": "user_001",
        "username": "demo_user",
        "email": "demo@ttdr.com",
        "hashed_password": "demo_password",
        "is_active": True,
        "permissions": ["research:create", "research:read", "research:monitor"]
    },
    "admin_user": {
        "id": "user_002", 
        "username": "admin_user",
        "email": "admin@ttdr.com",
        "hashed_password": "admin_password",
        "is_active": True,
        "permissions": ["*"]  # All permissions
    }
}

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    user = MOCK_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        username: str = payload.get("username")
        permissions: list = payload.get("permissions", [])
        exp: datetime = datetime.fromtimestamp(payload.get("exp"))
        
        if user_id is None or username is None:
            return None
            
        return TokenData(
            user_id=user_id,
            username=username,
            permissions=permissions,
            exp=exp
        )
    except jwt.PyJWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    # Check if token is expired
    if datetime.utcnow() > token_data.exp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from mock database
    user_data = MOCK_USERS.get(token_data.username)
    if user_data is None:
        raise credentials_exception
    
    return User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        is_active=user_data["is_active"],
        permissions=user_data["permissions"]
    )

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        # Skip permission check in development mode
        return User(
            id="dev_user_001",
            username="dev_user",
            email="dev@ttdr.com",
            is_active=True,
            permissions=["research:create", "research:read", "research:monitor", "*"]
        )
    
    return permission_checker

# Optional authentication (for public endpoints with enhanced features for authenticated users)
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    if credentials is None:
        return None
    
    try:
        token_data = verify_token(credentials.credentials)
        if token_data is None:
            return None
        
        # Check if token is expired
        if datetime.utcnow() > token_data.exp:
            return None
        
        # Get user from mock database
        user_data = MOCK_USERS.get(token_data.username)
        if user_data is None:
            return None
        
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            is_active=user_data["is_active"],
            permissions=user_data["permissions"]
        )
    except Exception:
        return None