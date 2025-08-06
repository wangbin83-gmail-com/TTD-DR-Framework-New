import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from models.core import (
    Draft, InformationGap, RetrievedInfo, QualityMetrics,
    ResearchRequirements, ResearchStructure, Section, Source,
    GapType, Priority, ComplexityLevel, ResearchDomain,
    TTDRState
)
from models.validation import DataValidator, TTDRStateValidator

class TestCoreModels:
    """Test core data models"""
    
    def test_source_model(self):
        """Test Source model validation"""
        source_data = {
            "url": "https://example.com",
            "title": "Test Article",
            "domain": "example.com",
            "credibility_score": 0.8
        }
        
        result = DataValidator.validate_model(Source, source_data)
        assert result.is_valid
        assert result.data.url == "https://example.com"
        assert result.data.credibility_score == 0.8
    
    def test_section_model(self):
        """Test Section model validation"""
        section_data = {
            "id": "intro",
            "title": "Introduction",
            "content": "This is the introduction section.",
            "estimated_length": 500
        }
        
        result = DataValidator.validate_model(Section, section_data)
        assert result.is_valid
        assert result.data.id == "intro"
        assert result.data.title == "Introduction"
    
    def test_research_structure_model(self):
        """Test ResearchStructure model validation"""
        structure_data = {
            "sections": [
                {
                    "id": "intro",
                    "title": "Introduction",
                    "estimated_length": 500
                }
            ],
            "estimated_length": 2000,
            "complexity_level": "intermediate",
            "domain": "technology"
        }
        
        result = DataValidator.validate_model(ResearchStructure, structure_data)
        assert result.is_valid
        assert result.data.complexity_level == ComplexityLevel.INTERMEDIATE
        assert result.data.domain == ResearchDomain.TECHNOLOGY
    
    def test_draft_model(self):
        """Test Draft model validation"""
        draft_data = {
            "id": "draft_1",
            "topic": "AI in Healthcare",
            "structure": {
                "sections": [
                    {
                        "id": "intro",
                        "title": "Introduction",
                        "estimated_length": 500
                    }
                ],
                "estimated_length": 2000,
                "complexity_level": "intermediate"
            },
            "content": {
                "intro": "Introduction content here"
            },
            "quality_score": 0.7,
            "iteration": 1
        }
        
        result = DataValidator.validate_model(Draft, draft_data)
        assert result.is_valid
        assert result.data.topic == "AI in Healthcare"
        assert result.data.quality_score == 0.7
    
    def test_information_gap_model(self):
        """Test InformationGap model validation"""
        gap_data = {
            "id": "gap_1",
            "section_id": "intro",
            "gap_type": "content",
            "description": "Missing background information",
            "priority": "high",
            "search_queries": [
                {
                    "query": "AI healthcare background",
                    "priority": "high",
                    "expected_results": 10
                }
            ]
        }
        
        result = DataValidator.validate_model(InformationGap, gap_data)
        assert result.is_valid
        assert result.data.gap_type == GapType.CONTENT
        assert result.data.priority == Priority.HIGH
    
    def test_quality_metrics_model(self):
        """Test QualityMetrics model validation and overall score calculation"""
        metrics_data = {
            "completeness": 0.8,
            "coherence": 0.7,
            "accuracy": 0.9,
            "citation_quality": 0.6
        }
        
        result = DataValidator.validate_model(QualityMetrics, metrics_data)
        assert result.is_valid
        # Overall score should be calculated as average
        expected_overall = (0.8 + 0.7 + 0.9 + 0.6) / 4
        assert abs(result.data.overall_score - expected_overall) < 0.001

class TestTTDRStateValidator:
    """Test TTDRState validation"""
    
    def test_create_initial_state(self):
        """Test creating initial TTDRState"""
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE
        )
        
        state = TTDRStateValidator.create_initial_state("AI in Healthcare", requirements)
        
        assert state['topic'] == "AI in Healthcare"
        assert state['requirements'] == requirements
        assert state['current_draft'] is None
        assert state['iteration_count'] == 0
        assert len(state['information_gaps']) == 0
        assert len(state['retrieved_info']) == 0
    
    def test_validate_complete_state(self):
        """Test validating a complete TTDRState"""
        requirements = ResearchRequirements()
        state_data = {
            'topic': 'Test Topic',
            'requirements': requirements.model_dump(),
            'current_draft': None,
            'information_gaps': [],
            'retrieved_info': [],
            'iteration_count': 0,
            'quality_metrics': None,
            'evolution_history': [],
            'final_report': None,
            'error_log': []
        }
        
        result = TTDRStateValidator.validate_state(state_data)
        assert result.is_valid
        assert len(result.errors) == 0

class TestDataSerialization:
    """Test data serialization and deserialization"""
    
    def test_model_serialization(self):
        """Test serializing models to dict and JSON"""
        source = Source(
            url="https://example.com",
            title="Test Article",
            domain="example.com",
            credibility_score=0.8
        )
        
        # Test dict serialization
        source_dict = DataValidator.serialize_model(source)
        assert source_dict['url'] == "https://example.com"
        assert source_dict['credibility_score'] == 0.8
        
        # Test JSON serialization
        source_json = DataValidator.serialize_to_json(source)
        assert '"url": "https://example.com"' in source_json
    
    def test_json_deserialization(self):
        """Test deserializing from JSON"""
        source_json = '''
        {
            "url": "https://example.com",
            "title": "Test Article",
            "domain": "example.com",
            "credibility_score": 0.8
        }
        '''
        
        result = DataValidator.deserialize_from_json(Source, source_json)
        assert result.is_valid
        assert result.data.url == "https://example.com"
        assert result.data.credibility_score == 0.8
    
    def test_invalid_json_deserialization(self):
        """Test handling invalid JSON"""
        invalid_json = '{"url": "https://example.com", "invalid": }'
        
        result = DataValidator.deserialize_from_json(Source, invalid_json)
        assert not result.is_valid
        assert len(result.errors) > 0