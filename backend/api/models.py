"""
API models for TTD-DR Framework REST API
Implements request/response validation and formatting for React frontend
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from models.core import (
    ResearchDomain, ComplexityLevel, Priority, GapType,
    ResearchRequirements, QualityMetrics, Draft, InformationGap
)

# API Request Models

class ResearchTopicRequest(BaseModel):
    """Request model for initiating research workflow"""
    topic: str = Field(..., min_length=5, max_length=500, description="Research topic")
    domain: ResearchDomain = Field(default=ResearchDomain.GENERAL, description="Research domain")
    complexity_level: ComplexityLevel = Field(default=ComplexityLevel.INTERMEDIATE, description="Complexity level")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum iterations")
    quality_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Quality threshold")
    max_sources: int = Field(default=20, ge=5, le=100, description="Maximum sources to retrieve")
    preferred_source_types: List[str] = Field(default=["academic", "news", "official"], description="Preferred source types")
    
    @validator('topic')
    def validate_topic(cls, v):
        """Validate research topic"""
        if not v.strip():
            raise ValueError("Topic cannot be empty")
        # Remove excessive whitespace
        return ' '.join(v.split())
    
    @validator('preferred_source_types')
    def validate_source_types(cls, v):
        """Validate source types"""
        valid_types = ["academic", "news", "official", "blog", "wiki", "forum"]
        invalid_types = [t for t in v if t not in valid_types]
        if invalid_types:
            raise ValueError(f"Invalid source types: {invalid_types}")
        return v

class WorkflowConfigRequest(BaseModel):
    """Request model for workflow configuration"""
    enable_persistence: bool = Field(default=True, description="Enable state persistence")
    enable_recovery: bool = Field(default=True, description="Enable error recovery")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    max_execution_time: int = Field(default=1800, ge=300, le=3600, description="Max execution time in seconds")

# API Response Models

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NodeStatus(BaseModel):
    """Status of individual workflow node"""
    name: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error: Optional[str] = None

class WorkflowProgressResponse(BaseModel):
    """Response model for workflow progress"""
    execution_id: str
    status: WorkflowStatus
    current_node: Optional[str] = None
    progress_percentage: float = Field(ge=0.0, le=100.0)
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    nodes_completed: List[NodeStatus] = []
    iterations_completed: int = 0
    current_quality_score: Optional[float] = None
    error_message: Optional[str] = None

class ResearchInitiationResponse(BaseModel):
    """Response model for research initiation"""
    execution_id: str
    status: WorkflowStatus
    message: str
    estimated_duration: int  # in seconds
    websocket_url: str

class QualityMetricsResponse(BaseModel):
    """Response model for quality metrics"""
    completeness: float
    coherence: float
    accuracy: float
    citation_quality: float
    overall_score: float
    assessment_timestamp: datetime

class DraftSummaryResponse(BaseModel):
    """Response model for draft summary"""
    id: str
    topic: str
    word_count: int
    sections_count: int
    quality_score: float
    iteration: int
    last_updated: datetime

class InformationGapResponse(BaseModel):
    """Response model for information gaps"""
    id: str
    section_id: str
    gap_type: GapType
    description: str
    priority: Priority
    search_queries_count: int
    impact_score: float

class WorkflowResultResponse(BaseModel):
    """Response model for completed workflow"""
    execution_id: str
    status: WorkflowStatus
    final_report: Optional[str] = None
    quality_metrics: Optional[QualityMetricsResponse] = None
    draft_summary: Optional[DraftSummaryResponse] = None
    information_gaps: List[InformationGapResponse] = []
    total_duration: float
    iterations_completed: int
    sources_used: int
    completion_time: datetime
    download_urls: Dict[str, str] = {}  # format -> download_url

class ErrorResponse(BaseModel):
    """Response model for API errors"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

# WebSocket Message Models

class WebSocketMessageType(str, Enum):
    """WebSocket message types"""
    STATUS_UPDATE = "status_update"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    ITERATION_COMPLETED = "iteration_completed"
    QUALITY_UPDATE = "quality_update"
    ERROR = "error"
    WORKFLOW_COMPLETED = "workflow_completed"

class WebSocketMessage(BaseModel):
    """Base WebSocket message model"""
    type: WebSocketMessageType
    execution_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = {}

class StatusUpdateMessage(WebSocketMessage):
    """Status update WebSocket message"""
    type: WebSocketMessageType = WebSocketMessageType.STATUS_UPDATE
    
    def __init__(self, execution_id: str, status: WorkflowStatus, 
                 current_node: Optional[str] = None, progress: float = 0.0, **kwargs):
        super().__init__(
            execution_id=execution_id,
            data={
                "status": status,
                "current_node": current_node,
                "progress": progress,
                **kwargs
            }
        )

class NodeCompletedMessage(WebSocketMessage):
    """Node completion WebSocket message"""
    type: WebSocketMessageType = WebSocketMessageType.NODE_COMPLETED
    
    def __init__(self, execution_id: str, node_name: str, duration: float, **kwargs):
        super().__init__(
            execution_id=execution_id,
            data={
                "node_name": node_name,
                "duration": duration,
                **kwargs
            }
        )

class QualityUpdateMessage(WebSocketMessage):
    """Quality update WebSocket message"""
    type: WebSocketMessageType = WebSocketMessageType.QUALITY_UPDATE
    
    def __init__(self, execution_id: str, quality_metrics: QualityMetrics, **kwargs):
        super().__init__(
            execution_id=execution_id,
            data={
                "quality_metrics": {
                    "completeness": quality_metrics.completeness,
                    "coherence": quality_metrics.coherence,
                    "accuracy": quality_metrics.accuracy,
                    "citation_quality": quality_metrics.citation_quality,
                    "overall_score": quality_metrics.overall_score
                },
                **kwargs
            }
        )

# Utility functions for model conversion

def convert_draft_to_summary(draft: Draft) -> DraftSummaryResponse:
    """Convert Draft model to DraftSummaryResponse"""
    return DraftSummaryResponse(
        id=draft.id,
        topic=draft.topic,
        word_count=draft.metadata.word_count,
        sections_count=len(draft.structure.sections),
        quality_score=draft.quality_score,
        iteration=draft.iteration,
        last_updated=draft.metadata.updated_at
    )

def convert_gap_to_response(gap: InformationGap) -> InformationGapResponse:
    """Convert InformationGap model to InformationGapResponse"""
    return InformationGapResponse(
        id=gap.id,
        section_id=gap.section_id,
        gap_type=gap.gap_type,
        description=gap.description,
        priority=gap.priority,
        search_queries_count=len(gap.search_queries),
        impact_score=gap.impact_score
    )

def convert_quality_metrics_to_response(metrics: QualityMetrics) -> QualityMetricsResponse:
    """Convert QualityMetrics model to QualityMetricsResponse"""
    return QualityMetricsResponse(
        completeness=metrics.completeness,
        coherence=metrics.coherence,
        accuracy=metrics.accuracy,
        citation_quality=metrics.citation_quality,
        overall_score=metrics.overall_score,
        assessment_timestamp=datetime.now()
    )