# TTD-DR Framework Backend Entry Point
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.endpoints import router as api_router
from api.rate_limiting import rate_limit_middleware, custom_rate_limit_exceeded_handler
from api.websocket_manager import ping_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter for slowapi
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting TTD-DR Framework API")
    
    # Start background tasks
    ping_task_handle = asyncio.create_task(ping_task())
    
    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down TTD-DR Framework API")
        ping_task_handle.cancel()
        try:
            await ping_task_handle
        except asyncio.CancelledError:
            pass

# Create FastAPI application
app = FastAPI(
    title="TTD-DR Framework API",
    description="Test-Time Diffusion Deep Researcher Framework REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiting state
app.state.limiter = limiter

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Add custom exception handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    return await custom_rate_limit_exceeded_handler(request, exc)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "timestamp": str(asyncio.get_event_loop().time())
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "path": str(request.url.path),
            "timestamp": str(asyncio.get_event_loop().time())
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": exc.errors(),
            "body": exc.body,
            "status_code": 422,
            "path": str(request.url.path),
            "timestamp": str(asyncio.get_event_loop().time())
        }
    )

# Include API router
app.include_router(api_router)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TTD-DR Framework API",
        "version": "1.0.0",
        "description": "Test-Time Diffusion Deep Researcher Framework",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_prefix": "/api/v1"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "TTD-DR Framework API",
        "version": "1.0.0",
        "timestamp": str(asyncio.get_event_loop().time())
    }

@app.get("/api/v1/health")
async def api_health_check():
    """API-specific health check"""
    return {
        "status": "healthy",
        "api_version": "v1",
        "endpoints": {
            "auth": "/api/v1/auth/login",
            "research": "/api/v1/research/initiate",
            "websocket": "/api/v1/research/ws/{execution_id}",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )