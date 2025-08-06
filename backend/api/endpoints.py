"""
REST API endpoints for TTD-DR Framework
Implements workflow initiation, monitoring, and management endpoints
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, StreamingResponse
import io
import json

from .models import (
    ResearchTopicRequest, WorkflowConfigRequest, ResearchInitiationResponse,
    WorkflowProgressResponse, WorkflowResultResponse, ErrorResponse,
    WorkflowStatus, QualityMetricsResponse, convert_draft_to_summary,
    convert_gap_to_response, convert_quality_metrics_to_response
)
from .auth import (
    get_current_user, require_permission, authenticate_user,
    create_access_token, get_current_user_optional, User
)
from .rate_limiting import rate_limit, advanced_limiter
from .websocket_manager import (
    connection_manager, authenticate_websocket, handle_websocket_messages
)
from workflow.workflow_orchestrator import (
    WorkflowExecutionEngine, WorkflowConfig, create_workflow_state
)
from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from services.monitoring_alerting import global_monitoring_system

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["TTD-DR Framework"])

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Global workflow engine
workflow_engine = WorkflowExecutionEngine()

# In-memory storage for workflow results (in production, use a database)
workflow_results: Dict[str, Dict[str, Any]] = {}

# Authentication endpoints
@router.post("/auth/login", response_model=Dict[str, Any])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token
    
    Args:
        form_data: Username and password from form
        
    Returns:
        Access token and user information
    """
    try:
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={
                "user_id": user["id"],
                "username": user["username"],
                "permissions": user["permissions"]
            },
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 1800,  # 30 minutes
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "permissions": user["permissions"]
            }
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.post("/auth/refresh", response_model=Dict[str, Any])
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    Refresh JWT token for authenticated user
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        New access token
    """
    try:
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={
                "user_id": current_user.id,
                "username": current_user.username,
                "permissions": current_user.permissions
            },
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 1800,
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "permissions": current_user.permissions
            }
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service error"
        )

# Research workflow endpoints
@router.post("/research/initiate", response_model=ResearchInitiationResponse)
async def initiate_research(
    request: ResearchTopicRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Initiate a new research workflow
    
    Args:
        request: Research topic and configuration
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Research initiation response with execution ID
    """
    try:
        logger.info("Initiating research workflow for development user")
        logger.info(f"Request data: {request.dict()}")
        
        # Generate unique execution ID
        execution_id = f"ttdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create research requirements
        requirements = ResearchRequirements(
            domain=request.domain,
            complexity_level=request.complexity_level,
            max_iterations=request.max_iterations,
            quality_threshold=request.quality_threshold,
            max_sources=request.max_sources,
            preferred_source_types=request.preferred_source_types
        )
        
        # Create initial workflow state
        initial_state = create_workflow_state(request.topic, requirements)
        
        # Store initial workflow data
        workflow_results[execution_id] = {
            "execution_id": execution_id,
            "status": WorkflowStatus.PENDING,
            "topic": request.topic,
            "user_id": "dev_user_001",
            "created_at": datetime.now(),
            "initial_state": initial_state,
            "progress": 0.0,
            "current_node": None,
            "error_message": None
        }
        
        # Start workflow execution in background
        background_tasks.add_task(
            execute_workflow_background,
            execution_id,
            initial_state,
            "dev_user_001"
        )
        
        # Calculate estimated duration based on complexity
        estimated_duration = _calculate_estimated_duration(request)
        
        logger.info(f"Initiated research workflow: {execution_id} for user: {current_user.username}")
        
        return ResearchInitiationResponse(
            execution_id=execution_id,
            status=WorkflowStatus.PENDING,
            message=f"Research workflow initiated for topic: {request.topic}",
            estimated_duration=estimated_duration,
            websocket_url=f"/api/v1/research/ws/{execution_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate research workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate research workflow: {str(e)}"
        )

@router.get("/research/status/{execution_id}", response_model=WorkflowProgressResponse)
async def get_workflow_status(
    execution_id: str,
    current_user: User = Depends(require_permission("research:read"))
):
    """
    Get current status of a research workflow
    
    Args:
        execution_id: Workflow execution ID
        current_user: Authenticated user
        
    Returns:
        Current workflow progress and status
    """
    try:
        if execution_id not in workflow_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution not found: {execution_id}"
            )
        
        workflow_data = workflow_results[execution_id]
        
        # Check user permissions (users can only see their own workflows unless admin)
        if (workflow_data["user_id"] != current_user.id and 
            "*" not in current_user.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this workflow execution"
            )
        
        # Get execution status from workflow engine
        engine_status = workflow_engine.get_execution_status(execution_id)
        
        # Calculate estimated completion time
        estimated_completion = None
        if workflow_data["status"] == WorkflowStatus.RUNNING:
            start_time = workflow_data["created_at"]
            elapsed = (datetime.now() - start_time).total_seconds()
            estimated_total = _calculate_estimated_duration_from_progress(
                workflow_data["progress"]
            )
            if estimated_total > elapsed:
                estimated_completion = start_time + timedelta(seconds=estimated_total)
        
        return WorkflowProgressResponse(
            execution_id=execution_id,
            status=workflow_data["status"],
            current_node=workflow_data.get("current_node"),
            progress_percentage=workflow_data.get("progress", 0.0),
            start_time=workflow_data["created_at"],
            estimated_completion=estimated_completion,
            nodes_completed=workflow_data.get("nodes_completed", []),
            iterations_completed=workflow_data.get("iterations_completed", 0),
            current_quality_score=workflow_data.get("current_quality_score"),
            error_message=workflow_data.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )

@router.get("/research/result/{execution_id}", response_model=WorkflowResultResponse)
async def get_workflow_result(
    execution_id: str,
    current_user: User = Depends(require_permission("research:read"))
):
    """
    Get final result of a completed research workflow
    
    Args:
        execution_id: Workflow execution ID
        current_user: Authenticated user
        
    Returns:
        Complete workflow result with final report
    """
    try:
        if execution_id not in workflow_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution not found: {execution_id}"
            )
        
        workflow_data = workflow_results[execution_id]
        
        # Check user permissions
        if (workflow_data["user_id"] != current_user.id and 
            "*" not in current_user.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this workflow execution"
            )
        
        # Check if workflow is completed
        if workflow_data["status"] != WorkflowStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Workflow is not completed. Current status: {workflow_data['status']}"
            )
        
        final_state = workflow_data.get("final_state")
        if not final_state:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Final workflow state not available"
            )
        
        # Convert models to response format
        quality_metrics = None
        if final_state.get("quality_metrics"):
            quality_metrics = convert_quality_metrics_to_response(
                final_state["quality_metrics"]
            )
        
        draft_summary = None
        if final_state.get("current_draft"):
            draft_summary = convert_draft_to_summary(final_state["current_draft"])
        
        information_gaps = []
        if final_state.get("information_gaps"):
            information_gaps = [
                convert_gap_to_response(gap) 
                for gap in final_state["information_gaps"]
            ]
        
        # Generate download URLs
        download_urls = {}
        if final_state.get("final_report"):
            download_urls = {
                "markdown": f"/api/v1/research/download/{execution_id}/markdown",
                "pdf": f"/api/v1/research/download/{execution_id}/pdf",
                "docx": f"/api/v1/research/download/{execution_id}/docx"
            }
        
        return WorkflowResultResponse(
            execution_id=execution_id,
            status=workflow_data["status"],
            final_report=final_state.get("final_report"),
            quality_metrics=quality_metrics,
            draft_summary=draft_summary,
            information_gaps=information_gaps,
            total_duration=workflow_data.get("total_duration", 0.0),
            iterations_completed=final_state.get("iteration_count", 0),
            sources_used=len(final_state.get("retrieved_info", [])),
            completion_time=workflow_data.get("completed_at", datetime.now()),
            download_urls=download_urls
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow result: {str(e)}"
        )

@router.post("/research/cancel/{execution_id}")
async def cancel_workflow(
    execution_id: str,
    current_user: User = Depends(require_permission("research:create"))
):
    """
    Cancel a running research workflow
    
    Args:
        execution_id: Workflow execution ID
        current_user: Authenticated user
        
    Returns:
        Cancellation confirmation
    """
    try:
        if execution_id not in workflow_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution not found: {execution_id}"
            )
        
        workflow_data = workflow_results[execution_id]
        
        # Check user permissions
        if (workflow_data["user_id"] != current_user.id and 
            "*" not in current_user.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this workflow execution"
            )
        
        # Check if workflow can be cancelled
        if workflow_data["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel workflow with status: {workflow_data['status']}"
            )
        
        # Cancel execution
        success = workflow_engine.cancel_execution(execution_id)
        
        if success:
            workflow_data["status"] = WorkflowStatus.CANCELLED
            workflow_data["completed_at"] = datetime.now()
            workflow_data["error_message"] = "Workflow cancelled by user"
            
            # Notify WebSocket connections
            await connection_manager.send_status_update(
                execution_id,
                WorkflowStatus.CANCELLED,
                progress=workflow_data.get("progress", 0.0)
            )
            
            logger.info(f"Cancelled workflow execution: {execution_id}")
            
            return {
                "message": f"Workflow execution cancelled: {execution_id}",
                "execution_id": execution_id,
                "status": WorkflowStatus.CANCELLED
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel workflow execution"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {str(e)}"
        )

@router.get("/research/list")
async def list_workflows(
    status: Optional[WorkflowStatus] = None,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(require_permission("research:read"))
):
    """
    List research workflows for the current user
    
    Args:
        status: Optional status filter
        limit: Maximum number of results
        offset: Pagination offset
        current_user: Authenticated user
        
    Returns:
        List of workflow executions
    """
    try:
        # Filter workflows by user (unless admin)
        user_workflows = []
        for execution_id, workflow_data in workflow_results.items():
            if (workflow_data["user_id"] == current_user.id or 
                "*" in current_user.permissions):
                
                if status is None or workflow_data["status"] == status:
                    user_workflows.append({
                        "execution_id": execution_id,
                        "topic": workflow_data["topic"],
                        "status": workflow_data["status"],
                        "created_at": workflow_data["created_at"],
                        "progress": workflow_data.get("progress", 0.0),
                        "user_id": workflow_data["user_id"]
                    })
        
        # Sort by creation time (newest first)
        user_workflows.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        total_count = len(user_workflows)
        paginated_workflows = user_workflows[offset:offset + limit]
        
        return {
            "workflows": paginated_workflows,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )

@router.get("/research/download/{execution_id}/{format}")
async def download_report(
    execution_id: str,
    format: str,
    current_user: User = Depends(require_permission("research:read"))
):
    """
    Download research report in specified format
    
    Args:
        execution_id: Workflow execution ID
        format: Output format (markdown, pdf, docx)
        current_user: Authenticated user
        
    Returns:
        File download response
    """
    try:
        if execution_id not in workflow_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution not found: {execution_id}"
            )
        
        workflow_data = workflow_results[execution_id]
        
        # Check user permissions
        if (workflow_data["user_id"] != current_user.id and 
            "*" not in current_user.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this workflow execution"
            )
        
        # Check if workflow is completed
        if workflow_data["status"] != WorkflowStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workflow must be completed to download report"
            )
        
        final_state = workflow_data.get("final_state")
        if not final_state or not final_state.get("final_report"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Final report not available"
            )
        
        final_report = final_state["final_report"]
        topic = workflow_data["topic"]
        
        # Generate filename
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        
        if format.lower() == "markdown":
            filename = f"{safe_topic}_report.md"
            content = final_report
            media_type = "text/markdown"
            
        elif format.lower() == "pdf":
            # For PDF generation, you'd typically use a library like reportlab or weasyprint
            # For now, return a simple text file
            filename = f"{safe_topic}_report.txt"
            content = f"Research Report: {topic}\n\n{final_report}"
            media_type = "text/plain"
            
        elif format.lower() == "docx":
            # For DOCX generation, you'd use python-docx library
            # For now, return a simple text file
            filename = f"{safe_topic}_report.txt"
            content = f"Research Report: {topic}\n\n{final_report}"
            media_type = "text/plain"
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Supported formats: markdown, pdf, docx"
            )
        
        # Create file-like object
        file_content = io.BytesIO(content.encode('utf-8'))
        
        return StreamingResponse(
            io.BytesIO(content.encode('utf-8')),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download report: {str(e)}"
        )

# WebSocket endpoint for real-time updates
@router.websocket("/research/ws/{execution_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    execution_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time workflow progress updates
    
    Args:
        websocket: WebSocket connection
        execution_id: Workflow execution ID to monitor
        token: Optional JWT token for authentication
    """
    try:
        # Authenticate WebSocket connection
        token_data = await authenticate_websocket(websocket, token)
        user_id = token_data.user_id if token_data else None
        
        # Connect to connection manager
        connected = await connection_manager.connect(websocket, execution_id, user_id)
        
        if not connected:
            logger.error(f"Failed to establish WebSocket connection for execution: {execution_id}")
            return
        
        # Handle incoming messages
        await handle_websocket_messages(websocket, execution_id)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution: {execution_id}")
    except Exception as e:
        logger.error(f"WebSocket error for execution {execution_id}: {e}")
    finally:
        await connection_manager.disconnect(websocket)

# Monitoring and alerting endpoints
@router.get("/monitoring/health")
async def get_system_health(
    current_user: User = Depends(require_permission("monitoring:read"))
):
    """
    Get overall system health status
    
    Args:
        current_user: Authenticated user with monitoring permissions
        
    Returns:
        System health status and component details
    """
    try:
        health_status = global_monitoring_system.get_system_health()
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )

@router.get("/monitoring/alerts")
async def get_active_alerts(
    current_user: User = Depends(require_permission("monitoring:read"))
):
    """
    Get all active alerts
    
    Args:
        current_user: Authenticated user with monitoring permissions
        
    Returns:
        List of active alerts
    """
    try:
        active_alerts = global_monitoring_system.get_active_alerts()
        return {
            "alerts": active_alerts,
            "count": len(active_alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active alerts: {str(e)}"
        )

@router.get("/monitoring/metrics")
async def get_performance_metrics(
    time_window_minutes: int = Query(default=60, ge=1, le=1440),
    current_user: User = Depends(require_permission("monitoring:read"))
):
    """
    Get performance metrics for a specified time window
    
    Args:
        time_window_minutes: Time window in minutes (1-1440)
        current_user: Authenticated user with monitoring permissions
        
    Returns:
        Performance metrics and statistics
    """
    try:
        metrics = global_monitoring_system.get_performance_metrics(time_window_minutes)
        return {
            "metrics": metrics,
            "time_window_minutes": time_window_minutes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

@router.post("/monitoring/workflow/{execution_id}/metrics")
async def record_workflow_metrics(
    execution_id: str,
    node_name: str,
    execution_time_ms: float,
    success: bool,
    memory_usage_mb: float = 0.0,
    error_count: int = 0,
    current_user: User = Depends(require_permission("research:create"))
):
    """
    Record workflow execution metrics
    
    Args:
        execution_id: Workflow execution ID
        node_name: Name of the workflow node
        execution_time_ms: Execution time in milliseconds
        success: Whether the execution was successful
        memory_usage_mb: Memory usage in MB
        error_count: Number of errors encountered
        current_user: Authenticated user
        
    Returns:
        Confirmation of metrics recording
    """
    try:
        global_monitoring_system.record_workflow_execution(
            workflow_id=execution_id,
            node_name=node_name,
            execution_time_ms=execution_time_ms,
            success=success,
            memory_usage_mb=memory_usage_mb,
            error_count=error_count
        )
        
        return {
            "message": "Workflow metrics recorded successfully",
            "execution_id": execution_id,
            "node_name": node_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to record workflow metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record workflow metrics: {str(e)}"
        )

@router.post("/monitoring/start")
async def start_monitoring(
    current_user: User = Depends(require_permission("*"))
):
    """
    Start the monitoring system (admin only)
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        Monitoring system start confirmation
    """
    try:
        if global_monitoring_system.monitoring_active:
            return {
                "message": "Monitoring system is already active",
                "status": "active",
                "timestamp": datetime.now().isoformat()
            }
        
        await global_monitoring_system.start_monitoring()
        
        return {
            "message": "Monitoring system started successfully",
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start monitoring system: {str(e)}"
        )

@router.post("/monitoring/stop")
async def stop_monitoring(
    current_user: User = Depends(require_permission("*"))
):
    """
    Stop the monitoring system (admin only)
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        Monitoring system stop confirmation
    """
    try:
        if not global_monitoring_system.monitoring_active:
            return {
                "message": "Monitoring system is already inactive",
                "status": "inactive",
                "timestamp": datetime.now().isoformat()
            }
        
        await global_monitoring_system.stop_monitoring()
        
        return {
            "message": "Monitoring system stopped successfully",
            "status": "inactive",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop monitoring system: {str(e)}"
        )

# Admin endpoints
@router.post("/debug/test-request")
async def debug_request(request: ResearchTopicRequest):
    """Debug endpoint to test request validation"""
    return {
        "received_data": request.dict(),
        "validation": "passed",
        "domain": request.domain,
        "complexity_level": request.complexity_level,
        "preferred_source_types": request.preferred_source_types
    }

@router.post("/debug/simple-initiate")
async def simple_initiate(request: dict):
    """Simple debug endpoint without authentication"""
    return {
        "received": request,
        "status": "ok"
    }

@router.get("/admin/stats")
async def get_system_stats(
    current_user: User = Depends(require_permission("*"))
):
    """
    Get system statistics (admin only)
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        System statistics
    """
    try:
        active_executions = workflow_engine.list_active_executions()
        connection_stats = connection_manager.get_connection_stats()
        
        # Calculate workflow statistics
        total_workflows = len(workflow_results)
        status_counts = {}
        for workflow_data in workflow_results.values():
            status = workflow_data["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get monitoring system status
        monitoring_status = {
            "active": global_monitoring_system.monitoring_active,
            "health_checks": len(global_monitoring_system.health_checks),
            "alert_channels": len(global_monitoring_system.alert_channels),
            "active_alerts": len([a for a in global_monitoring_system.active_alerts.values() if not a.resolved]),
            "monitoring_tasks": len(global_monitoring_system.monitoring_tasks)
        }
        
        return {
            "system": {
                "active_executions": len(active_executions),
                "total_workflows": total_workflows,
                "websocket_connections": connection_stats["total_connections"]
            },
            "workflows": {
                "status_distribution": status_counts,
                "active_executions": active_executions
            },
            "websockets": connection_stats,
            "monitoring": monitoring_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )

# Background task functions
async def execute_workflow_background(
    execution_id: str,
    initial_state: Dict[str, Any],
    user_id: str
):
    """
    Execute workflow in background task
    
    Args:
        execution_id: Workflow execution ID
        initial_state: Initial workflow state
        user_id: User ID who initiated the workflow
    """
    try:
        logger.info(f"Starting background workflow execution: {execution_id}")
        
        # Update status to running
        workflow_results[execution_id]["status"] = WorkflowStatus.RUNNING
        workflow_results[execution_id]["current_node"] = "draft_generator"
        
        # Notify WebSocket connections
        await connection_manager.send_status_update(
            execution_id,
            WorkflowStatus.RUNNING,
            current_node="draft_generator",
            progress=5.0
        )
        
        # Execute workflow
        final_state = workflow_engine.execute_workflow(initial_state, execution_id)
        
        # Update workflow results
        workflow_results[execution_id].update({
            "status": WorkflowStatus.COMPLETED,
            "final_state": final_state,
            "completed_at": datetime.now(),
            "progress": 100.0,
            "current_node": None
        })
        
        # Notify completion
        await connection_manager.send_workflow_completed(
            execution_id,
            final_report=final_state.get("final_report"),
            quality_score=final_state.get("quality_metrics", {}).get("overall_score")
        )
        
        logger.info(f"Completed background workflow execution: {execution_id}")
        
    except Exception as e:
        logger.error(f"Background workflow execution failed: {execution_id} - {e}")
        
        # Update status to failed
        workflow_results[execution_id].update({
            "status": WorkflowStatus.FAILED,
            "error_message": str(e),
            "completed_at": datetime.now()
        })
        
        # Notify error
        await connection_manager.send_error(execution_id, str(e))

# Utility functions
def _calculate_estimated_duration(request: ResearchTopicRequest) -> int:
    """Calculate estimated workflow duration based on request parameters"""
    base_duration = 300  # 5 minutes base
    
    # Adjust for complexity
    complexity_multiplier = {
        ComplexityLevel.BASIC: 0.7,
        ComplexityLevel.INTERMEDIATE: 1.0,
        ComplexityLevel.ADVANCED: 1.5,
        ComplexityLevel.EXPERT: 2.0
    }
    
    duration = base_duration * complexity_multiplier.get(request.complexity_level, 1.0)
    
    # Adjust for iterations and sources
    duration *= (1 + request.max_iterations * 0.2)
    duration *= (1 + request.max_sources * 0.01)
    
    return int(duration)

def _calculate_estimated_duration_from_progress(progress: float) -> float:
    """Calculate estimated total duration from current progress"""
    if progress <= 0:
        return 1800  # Default 30 minutes
    
    # Simple linear estimation
    elapsed_estimate = 300  # Assume 5 minutes elapsed for some progress
    return elapsed_estimate / (progress / 100.0)

# Export router
__all__ = ["router"]