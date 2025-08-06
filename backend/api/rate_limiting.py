"""
Rate limiting middleware for TTD-DR Framework API
Implements token bucket algorithm for API rate limiting
"""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        async with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            time_passed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + time_passed * self.refill_rate
            )
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class AdvancedRateLimiter:
    """Advanced rate limiter with different limits for different endpoints"""
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.endpoint_limits = {
            # Research workflow endpoints (more restrictive)
            "/api/v1/research/initiate": {"capacity": 5, "refill_rate": 1/60},  # 1 per minute
            "/api/v1/research/status": {"capacity": 100, "refill_rate": 10},    # 10 per second
            "/api/v1/research/cancel": {"capacity": 10, "refill_rate": 1/10},   # 1 per 10 seconds
            
            # Authentication endpoints
            "/api/v1/auth/login": {"capacity": 10, "refill_rate": 1/60},        # 1 per minute
            "/api/v1/auth/refresh": {"capacity": 20, "refill_rate": 1/30},      # 1 per 30 seconds
            
            # General API endpoints
            "default": {"capacity": 60, "refill_rate": 1}                       # 1 per second
        }
    
    def _get_bucket_key(self, client_id: str, endpoint: str) -> str:
        """Generate bucket key for client and endpoint"""
        return f"{client_id}:{endpoint}"
    
    def _get_endpoint_limits(self, endpoint: str) -> Dict[str, float]:
        """Get rate limits for specific endpoint"""
        return self.endpoint_limits.get(endpoint, self.endpoint_limits["default"])
    
    async def is_allowed(self, client_id: str, endpoint: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed under rate limits
        
        Args:
            client_id: Client identifier (IP address or user ID)
            endpoint: API endpoint path
            tokens: Number of tokens to consume
            
        Returns:
            True if request is allowed, False otherwise
        """
        bucket_key = self._get_bucket_key(client_id, endpoint)
        
        if bucket_key not in self.buckets:
            limits = self._get_endpoint_limits(endpoint)
            self.buckets[bucket_key] = TokenBucket(
                capacity=limits["capacity"],
                refill_rate=limits["refill_rate"]
            )
        
        return await self.buckets[bucket_key].consume(tokens)
    
    def get_bucket_status(self, client_id: str, endpoint: str) -> Dict[str, float]:
        """Get current bucket status for debugging"""
        bucket_key = self._get_bucket_key(client_id, endpoint)
        
        if bucket_key not in self.buckets:
            limits = self._get_endpoint_limits(endpoint)
            return {
                "tokens": limits["capacity"],
                "capacity": limits["capacity"],
                "refill_rate": limits["refill_rate"]
            }
        
        bucket = self.buckets[bucket_key]
        return {
            "tokens": bucket.tokens,
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate
        }

# Global rate limiter instance
advanced_limiter = AdvancedRateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint function
        
    Returns:
        Response or raises HTTPException if rate limited
    """
    # Get client identifier
    client_ip = get_remote_address(request)
    
    # Get user ID if authenticated
    client_id = client_ip
    if hasattr(request.state, 'user') and request.state.user:
        client_id = f"user:{request.state.user.id}"
    
    # Get endpoint path
    endpoint = request.url.path
    
    # Check rate limit
    is_allowed = await advanced_limiter.is_allowed(client_id, endpoint)
    
    if not is_allowed:
        # Get bucket status for error details
        bucket_status = advanced_limiter.get_bucket_status(client_id, endpoint)
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests to {endpoint}",
                "retry_after": int(1 / bucket_status["refill_rate"]),
                "limit_info": {
                    "capacity": bucket_status["capacity"],
                    "refill_rate": bucket_status["refill_rate"],
                    "current_tokens": bucket_status["tokens"]
                }
            },
            headers={
                "Retry-After": str(int(1 / bucket_status["refill_rate"])),
                "X-RateLimit-Limit": str(bucket_status["capacity"]),
                "X-RateLimit-Remaining": str(int(bucket_status["tokens"])),
                "X-RateLimit-Reset": str(int(time.time() + 1 / bucket_status["refill_rate"]))
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    bucket_status = advanced_limiter.get_bucket_status(client_id, endpoint)
    response.headers["X-RateLimit-Limit"] = str(bucket_status["capacity"])
    response.headers["X-RateLimit-Remaining"] = str(int(bucket_status["tokens"]))
    response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))  # Reset in 1 minute
    
    return response

# Decorator for endpoint-specific rate limiting
from functools import wraps

def rate_limit(endpoint_path: str):
    """
    Decorator for endpoint-specific rate limiting
    
    Args:
        endpoint_path: Endpoint path for rate limiting configuration
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs or args
            request = None
            for arg in args + tuple(kwargs.values()):
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                return await func(*args, **kwargs)
                
            # Get client identifier
            client_ip = get_remote_address(request)
            client_id = client_ip
            
            # Check if user is authenticated
            if hasattr(request.state, 'user') and request.state.user:
                client_id = f"user:{request.state.user.id}"
            
            # Check rate limit
            is_allowed = await advanced_limiter.is_allowed(client_id, endpoint_path)
            
            if not is_allowed:
                bucket_status = advanced_limiter.get_bucket_status(client_id, endpoint_path)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests to {endpoint_path}",
                        "retry_after": int(1 / bucket_status["refill_rate"])
                    }
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Custom rate limit exceeded handler
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded errors"""
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "error": "Rate limit exceeded",
            "message": "Too many requests",
            "retry_after": 60
        }
    )