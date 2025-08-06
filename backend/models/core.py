from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from typing_extensions import TypedDict

# Enums for type safety
class GapType(str, Enum):
    CONTENT = "content"
    EVIDENCE = "evidence"
    CITATION = "citation"
    ANALYSIS = "analysis"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplexityLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ResearchDomain(str, Enum):
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    BUSINESS = "business"
    ACADEMIC = "academic"
    GENERAL = "general"

# Core data models
class Source(BaseModel):
    """Represents a source of information"""
    url: str
    title: str
    domain: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    last_accessed: datetime = Field(default_factory=datetime.now)

class Section(BaseModel):
    """Represents a section in the research structure"""
    id: str
    title: str
    content: str = ""
    subsections: List['Section'] = []
    estimated_length: int = 0
    
    class Config:
        # Enable self-referencing
        arbitrary_types_allowed = True

class SectionRelationship(BaseModel):
    """Defines relationships between sections"""
    source_section_id: str
    target_section_id: str
    relationship_type: str  # "depends_on", "supports", "contradicts", etc.

class ResearchStructure(BaseModel):
    """Defines the overall structure of the research"""
    sections: List[Section]
    relationships: List[SectionRelationship] = []
    estimated_length: int
    complexity_level: ComplexityLevel
    domain: ResearchDomain = ResearchDomain.GENERAL

class DraftMetadata(BaseModel):
    """Metadata for research drafts"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    author: str = "TTD-DR Framework"
    version: str = "1.0"
    word_count: int = 0
    
class Draft(BaseModel):
    """Represents a research draft"""
    id: str
    topic: str
    structure: ResearchStructure
    content: Dict[str, str] = {}  # section_id -> content
    metadata: DraftMetadata = Field(default_factory=DraftMetadata)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    iteration: int = 0
    
    @validator('content')
    def validate_content_keys(cls, v, values):
        """Ensure content keys match section IDs"""
        if 'structure' in values:
            section_ids = {section.id for section in values['structure'].sections}
            invalid_keys = set(v.keys()) - section_ids
            if invalid_keys:
                raise ValueError(f"Invalid content keys: {invalid_keys}")
        return v

class SearchQuery(BaseModel):
    """Represents a search query"""
    query: str
    domain: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    expected_results: int = 10
    search_strategy: Optional[str] = None
    effectiveness_score: float = 0.7
    
    class Config:
        extra = "allow"  # Allow additional fields to be set dynamically

class InformationGap(BaseModel):
    """Represents an identified information gap"""
    id: str
    section_id: str
    gap_type: GapType
    description: str
    priority: Priority
    search_queries: List[SearchQuery] = []
    
    # Additional fields for gap analysis
    specific_needs: List[str] = []
    suggested_sources: List[str] = []
    affected_sections: List[str] = []
    section_connections: List[str] = []
    impact_score: float = 0.5
    
    class Config:
        extra = "allow"  # Allow additional fields to be set dynamically
    
class RetrievedInfo(BaseModel):
    """Represents retrieved information"""
    source: Source
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    gap_id: Optional[str] = None  # Links to the gap this info addresses

class QualityMetrics(BaseModel):
    """Quality assessment metrics"""
    completeness: float = Field(ge=0.0, le=1.0)
    coherence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    citation_quality: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def __init__(self, **data):
        # Calculate overall score if not provided
        if 'overall_score' not in data:
            metrics = [
                data.get('completeness', 0),
                data.get('coherence', 0),
                data.get('accuracy', 0),
                data.get('citation_quality', 0)
            ]
            data['overall_score'] = sum(metrics) / len(metrics) if metrics else 0.0
        super().__init__(**data)

class EvolutionRecord(BaseModel):
    """Records self-evolution improvements"""
    timestamp: datetime = Field(default_factory=datetime.now)
    component: str
    improvement_type: str
    description: str
    performance_before: float
    performance_after: float
    parameters_changed: Dict[str, Any] = {}

class ResearchRequirements(BaseModel):
    """Research requirements and constraints"""
    domain: ResearchDomain = ResearchDomain.GENERAL
    complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    max_iterations: int = 5
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_sources: int = 20
    preferred_source_types: List[str] = ["academic", "news", "official"]
    
# LangGraph State Definition
class TTDRState(TypedDict):
    """State object for LangGraph workflow"""
    topic: str
    requirements: ResearchRequirements
    current_draft: Optional[Draft]
    information_gaps: List[InformationGap]
    retrieved_info: List[RetrievedInfo]
    iteration_count: int
    quality_metrics: Optional[QualityMetrics]
    evolution_history: List[EvolutionRecord]
    final_report: Optional[str]
    error_log: List[str]  # For tracking errors during execution

# Update Section model to handle self-reference
Section.model_rebuild()