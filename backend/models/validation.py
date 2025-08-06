from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
import json
import logging

from .core import (
    TTDRState, Draft, InformationGap, RetrievedInfo, 
    QualityMetrics, EvolutionRecord, ResearchRequirements,
    ResearchStructure, Section, Source
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class ValidationResult:
    """Result of validation operation"""
    def __init__(self, is_valid: bool, data: Optional[Any] = None, errors: List[str] = None):
        self.is_valid = is_valid
        self.data = data
        self.errors = errors or []

class DataValidator:
    """Utility class for validating and serializing data models"""
    
    @staticmethod
    def validate_model(model_class: Type[T], data: Dict[str, Any]) -> ValidationResult:
        """Validate data against a Pydantic model"""
        try:
            validated_data = model_class(**data)
            return ValidationResult(True, validated_data)
        except ValidationError as e:
            errors = [f"{error['loc']}: {error['msg']}" for error in e.errors()]
            logger.error(f"Validation failed for {model_class.__name__}: {errors}")
            return ValidationResult(False, None, errors)
        except Exception as e:
            logger.error(f"Unexpected validation error for {model_class.__name__}: {e}")
            return ValidationResult(False, None, [str(e)])
    
    @staticmethod
    def serialize_model(model: BaseModel) -> Dict[str, Any]:
        """Serialize a Pydantic model to dictionary"""
        try:
            return model.model_dump()
        except Exception as e:
            logger.error(f"Serialization failed for {type(model).__name__}: {e}")
            return {}
    
    @staticmethod
    def serialize_to_json(model: BaseModel) -> str:
        """Serialize a Pydantic model to JSON string"""
        try:
            return model.model_dump_json()
        except Exception as e:
            logger.error(f"JSON serialization failed for {type(model).__name__}: {e}")
            return "{}"
    
    @staticmethod
    def deserialize_from_json(model_class: Type[T], json_str: str) -> ValidationResult:
        """Deserialize JSON string to Pydantic model"""
        try:
            data = json.loads(json_str)
            return DataValidator.validate_model(model_class, data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return ValidationResult(False, None, [f"Invalid JSON: {e}"])
        except Exception as e:
            logger.error(f"Unexpected deserialization error: {e}")
            return ValidationResult(False, None, [str(e)])

class TTDRStateValidator:
    """Specialized validator for TTDRState"""
    
    @staticmethod
    def validate_state(state: Dict[str, Any]) -> ValidationResult:
        """Validate TTDRState with comprehensive checks"""
        errors = []
        
        # Check required fields
        required_fields = ['topic', 'requirements', 'iteration_count']
        for field in required_fields:
            if field not in state:
                errors.append(f"Missing required field: {field}")
        
        # Validate requirements if present
        if 'requirements' in state:
            req_result = DataValidator.validate_model(ResearchRequirements, state['requirements'])
            if not req_result.is_valid:
                errors.extend([f"requirements.{error}" for error in req_result.errors])
        
        # Validate draft if present
        if 'current_draft' in state and state['current_draft'] is not None:
            draft_result = DataValidator.validate_model(Draft, state['current_draft'])
            if not draft_result.is_valid:
                errors.extend([f"current_draft.{error}" for error in draft_result.errors])
        
        # Validate information gaps
        if 'information_gaps' in state:
            for i, gap in enumerate(state['information_gaps']):
                gap_result = DataValidator.validate_model(InformationGap, gap)
                if not gap_result.is_valid:
                    errors.extend([f"information_gaps[{i}].{error}" for error in gap_result.errors])
        
        # Validate retrieved info
        if 'retrieved_info' in state:
            for i, info in enumerate(state['retrieved_info']):
                info_result = DataValidator.validate_model(RetrievedInfo, info)
                if not info_result.is_valid:
                    errors.extend([f"retrieved_info[{i}].{error}" for error in info_result.errors])
        
        # Validate quality metrics if present
        if 'quality_metrics' in state and state['quality_metrics'] is not None:
            metrics_result = DataValidator.validate_model(QualityMetrics, state['quality_metrics'])
            if not metrics_result.is_valid:
                errors.extend([f"quality_metrics.{error}" for error in metrics_result.errors])
        
        # Validate evolution history
        if 'evolution_history' in state:
            for i, record in enumerate(state['evolution_history']):
                record_result = DataValidator.validate_model(EvolutionRecord, record)
                if not record_result.is_valid:
                    errors.extend([f"evolution_history[{i}].{error}" for error in record_result.errors])
        
        # Validate iteration count
        if 'iteration_count' in state:
            if not isinstance(state['iteration_count'], int) or state['iteration_count'] < 0:
                errors.append("iteration_count must be a non-negative integer")
        
        return ValidationResult(len(errors) == 0, state if len(errors) == 0 else None, errors)
    
    @staticmethod
    def create_initial_state(topic: str, requirements: ResearchRequirements) -> TTDRState:
        """Create a valid initial TTDRState"""
        return {
            'topic': topic,
            'requirements': requirements,
            'current_draft': None,
            'information_gaps': [],
            'retrieved_info': [],
            'iteration_count': 0,
            'quality_metrics': None,
            'evolution_history': [],
            'final_report': None,
            'error_log': []
        }

class SchemaGenerator:
    """Generate JSON schemas for Kimi K2 structured responses"""
    
    @staticmethod
    def get_draft_schema() -> Dict[str, Any]:
        """Get JSON schema for Draft model"""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "topic": {"type": "string"},
                "structure": {
                    "type": "object",
                    "properties": {
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "subsections": {"type": "array"},
                                    "estimated_length": {"type": "integer"}
                                },
                                "required": ["id", "title"]
                            }
                        },
                        "estimated_length": {"type": "integer"},
                        "complexity_level": {"type": "string", "enum": ["basic", "intermediate", "advanced", "expert"]},
                        "domain": {"type": "string", "enum": ["technology", "science", "business", "academic", "general"]}
                    },
                    "required": ["sections", "estimated_length", "complexity_level"]
                },
                "content": {"type": "object"},
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "iteration": {"type": "integer", "minimum": 0}
            },
            "required": ["id", "topic", "structure"]
        }
    
    @staticmethod
    def get_information_gap_schema() -> Dict[str, Any]:
        """Get JSON schema for InformationGap model"""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "section_id": {"type": "string"},
                "gap_type": {"type": "string", "enum": ["content", "evidence", "citation", "analysis"]},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "search_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "domain": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                            "expected_results": {"type": "integer", "minimum": 1}
                        },
                        "required": ["query"]
                    }
                }
            },
            "required": ["id", "section_id", "gap_type", "description", "priority"]
        }
    
    @staticmethod
    def get_quality_metrics_schema() -> Dict[str, Any]:
        """Get JSON schema for QualityMetrics model"""
        return {
            "type": "object",
            "properties": {
                "completeness": {"type": "number", "minimum": 0, "maximum": 1},
                "coherence": {"type": "number", "minimum": 0, "maximum": 1},
                "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                "citation_quality": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["completeness", "coherence", "accuracy", "citation_quality"]
        }

# Global validator instances
data_validator = DataValidator()
state_validator = TTDRStateValidator()
schema_generator = SchemaGenerator()