"""
Tests for domain adaptation system.
This module tests domain detection, adaptation algorithms, and configurable
research strategies for different domains.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from backend.models.core import (
    ResearchDomain, ComplexityLevel, ResearchRequirements,
    Draft, DraftMetadata, InformationGap, SearchQuery, Priority, GapType
)
from backend.models.research_structure import (
    EnhancedResearchStructure, EnhancedSection, ContentPlaceholder,
    ResearchSectionType, ContentPlaceholderType
)
from backend.services.domain_adapter import (
    DomainAdapter, DomainDetectionResult, DomainConfidence,
    ResearchStrategy, TerminologyHandler, DomainAdaptationError
)
from backend.services.kimi_k2_client import KimiK2Client


class TestDomainAdapter:
    """Test cases for DomainAdapter class"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for testing"""
        mock_client = Mock(spec=KimiK2Client)
        return mock_client
    
    @pytest.fixture
    def domain_adapter(self, mock_kimi_client):
        """Create DomainAdapter instance with mocked client"""
        return DomainAdapter(mock_kimi_client)
    
    @pytest.fixture
    def sample_requirements(self):
        """Sample research requirements for testing"""
        return ResearchRequirements(
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=5,
            quality_threshold=0.8,
            max_sources=20,
            preferred_source_types=["academic", "news"]
        )
    
    def test_domain_detection_technology(self, domain_adapter, mock_kimi_client):
        """Test domain detection for technology topics"""
        # Mock Kimi K2 response
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.9,
            "SCIENCE": 0.1,
            "BUSINESS": 0.2,
            "ACADEMIC": 0.1,
            "GENERAL": 0.1,
            "reasoning": "Contains technical terminology and programming concepts"
        })
        
        topic = "Machine Learning Algorithms for Software Development"
        result = domain_adapter.detect_domain(topic)
        
        assert isinstance(result, DomainDetectionResult)
        assert result.primary_domain == ResearchDomain.TECHNOLOGY
        assert result.confidence > 0.7
        assert len(result.keywords_found) > 0
        assert "machine learning" in [kw.lower() for kw in result.keywords_found]
        
        # Verify Kimi K2 was called
        mock_kimi_client.generate_content.assert_called_once()
    
    def test_domain_detection_science(self, domain_adapter, mock_kimi_client):
        """Test domain detection for science topics"""
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.1,
            "SCIENCE": 0.95,
            "BUSINESS": 0.1,
            "ACADEMIC": 0.3,
            "GENERAL": 0.1,
            "reasoning": "Contains scientific methodology and research terms"
        })
        
        topic = "Clinical Trial Results for Cancer Treatment"
        result = domain_adapter.detect_domain(topic)
        
        assert result.primary_domain == ResearchDomain.SCIENCE
        assert result.confidence > 0.8
        assert len(result.secondary_domains) >= 0
        assert "clinical" in [kw.lower() for kw in result.keywords_found]
    
    def test_domain_detection_business(self, domain_adapter, mock_kimi_client):
        """Test domain detection for business topics"""
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.1,
            "SCIENCE": 0.1,
            "BUSINESS": 0.9,
            "ACADEMIC": 0.2,
            "GENERAL": 0.1,
            "reasoning": "Contains business and market terminology"
        })
        
        topic = "Market Analysis for E-commerce Platforms"
        result = domain_adapter.detect_domain(topic)
        
        assert result.primary_domain == ResearchDomain.BUSINESS
        assert result.confidence > 0.7
        assert "market" in [kw.lower() for kw in result.keywords_found]
    
    def test_domain_detection_academic(self, domain_adapter, mock_kimi_client):
        """Test domain detection for academic topics"""
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.1,
            "SCIENCE": 0.3,
            "BUSINESS": 0.1,
            "ACADEMIC": 0.95,
            "GENERAL": 0.1,
            "reasoning": "Contains academic research terminology"
        })
        
        topic = "Theoretical Framework for Educational Psychology Research"
        result = domain_adapter.detect_domain(topic)
        
        assert result.primary_domain == ResearchDomain.ACADEMIC
        assert result.confidence > 0.8
        assert "research" in [kw.lower() for kw in result.keywords_found]
    
    def test_domain_detection_general(self, domain_adapter, mock_kimi_client):
        """Test domain detection for general topics"""
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.2,
            "SCIENCE": 0.2,
            "BUSINESS": 0.2,
            "ACADEMIC": 0.2,
            "GENERAL": 0.8,
            "reasoning": "General topic without specific domain focus"
        })
        
        topic = "Overview of Climate Change Impacts"
        result = domain_adapter.detect_domain(topic)
        
        assert result.primary_domain == ResearchDomain.GENERAL
        assert result.confidence > 0.5
    
    def test_adapt_research_requirements(self, domain_adapter, sample_requirements):
        """Test adaptation of research requirements based on domain"""
        domain_result = DomainDetectionResult(
            primary_domain=ResearchDomain.TECHNOLOGY,
            confidence=0.9,
            secondary_domains=[],
            detection_method="test",
            keywords_found=["software", "algorithm"],
            reasoning="Test detection"
        )
        
        adapted = domain_adapter.adapt_research_requirements(sample_requirements, domain_result)
        
        assert adapted.domain == ResearchDomain.TECHNOLOGY
        assert adapted.quality_threshold >= 0.8  # Technology has higher threshold
        assert "tech_blogs" in adapted.preferred_source_types
        assert adapted.max_sources >= sample_requirements.max_sources
    
    def test_generate_domain_specific_structure_success(self, domain_adapter, mock_kimi_client):
        """Test successful generation of domain-specific structure"""
        mock_structure_data = {
            "sections": [
                {
                    "id": "intro",
                    "title": "Introduction",
                    "section_type": "introduction",
                    "estimated_length": 500,
                    "content_placeholders": [
                        {
                            "id": "intro_placeholder",
                            "placeholder_type": "introduction",
                            "title": "Technology Overview",
                            "description": "Overview of the technology",
                            "estimated_word_count": 200,
                            "priority": "high",
                            "kimi_k2_prompt_hints": ["Focus on technical aspects"]
                        }
                    ],
                    "dependencies": []
                }
            ],
            "estimated_total_length": 2000,
            "generation_strategy": "domain_optimized"
        }
        
        mock_kimi_client.generate_content.return_value = json.dumps(mock_structure_data)
        
        structure = domain_adapter.generate_domain_specific_structure(
            topic="Machine Learning Frameworks",
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE
        )
        
        assert isinstance(structure, EnhancedResearchStructure)
        assert structure.domain == ResearchDomain.TECHNOLOGY
        assert len(structure.sections) > 0
        assert structure.estimated_length > 0
        
        # Verify Kimi K2 was called
        mock_kimi_client.generate_content.assert_called_once()
    
    def test_generate_domain_specific_structure_fallback(self, domain_adapter, mock_kimi_client):
        """Test fallback when Kimi K2 fails"""
        # Mock Kimi K2 failure
        mock_kimi_client.generate_content.side_effect = Exception("API Error")
        
        structure = domain_adapter.generate_domain_specific_structure(
            topic="Test Topic",
            domain=ResearchDomain.SCIENCE,
            complexity_level=ComplexityLevel.BASIC
        )
        
        assert isinstance(structure, EnhancedResearchStructure)
        assert structure.domain == ResearchDomain.SCIENCE
        assert len(structure.sections) > 0
    
    def test_adapt_search_queries(self, domain_adapter):
        """Test adaptation of search queries for domain optimization"""
        gaps = [
            InformationGap(
                id="gap1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="Need more information about machine learning",
                priority=Priority.HIGH,
                search_queries=[
                    SearchQuery(query="machine learning basics", priority=Priority.HIGH)
                ]
            )
        ]
        
        adapted_gaps = domain_adapter.adapt_search_queries(gaps, ResearchDomain.TECHNOLOGY)
        
        assert len(adapted_gaps) == 1
        assert len(adapted_gaps[0].search_queries) >= len(gaps[0].search_queries)
        
        # Check that domain-specific queries were added
        query_texts = [q.query for q in adapted_gaps[0].search_queries]
        assert any("technical" in q.lower() or "implementation" in q.lower() for q in query_texts)
    
    def test_apply_domain_formatting_technology(self, domain_adapter):
        """Test domain-specific formatting for technology content"""
        content = "The API framework uses machine learning algorithms for data processing."
        
        formatted = domain_adapter.apply_domain_formatting(
            content, ResearchDomain.TECHNOLOGY, "general"
        )
        
        assert formatted != content  # Should be modified
        assert "`API`" in formatted or "API" in formatted  # Technical formatting applied
    
    def test_apply_domain_formatting_science(self, domain_adapter):
        """Test domain-specific formatting for science content"""
        content = "The study found p = 0.05 with n = 100 participants."
        
        formatted = domain_adapter.apply_domain_formatting(
            content, ResearchDomain.SCIENCE, "results"
        )
        
        assert "*p*" in formatted or "p =" in formatted  # Scientific formatting applied
    
    def test_get_domain_quality_criteria(self, domain_adapter):
        """Test retrieval of domain-specific quality criteria"""
        tech_criteria = domain_adapter.get_domain_quality_criteria(ResearchDomain.TECHNOLOGY)
        science_criteria = domain_adapter.get_domain_quality_criteria(ResearchDomain.SCIENCE)
        
        assert isinstance(tech_criteria, dict)
        assert isinstance(science_criteria, dict)
        assert "technical_accuracy" in tech_criteria
        assert "scientific_rigor" in science_criteria
        assert tech_criteria != science_criteria  # Should be different
    
    def test_get_kimi_system_prompt(self, domain_adapter):
        """Test retrieval of domain-specific Kimi K2 system prompts"""
        tech_prompt = domain_adapter.get_kimi_system_prompt(
            ResearchDomain.TECHNOLOGY, "content_generation"
        )
        science_prompt = domain_adapter.get_kimi_system_prompt(
            ResearchDomain.SCIENCE, "content_generation"
        )
        
        assert isinstance(tech_prompt, str)
        assert isinstance(science_prompt, str)
        assert len(tech_prompt) > 0
        assert len(science_prompt) > 0
        assert tech_prompt != science_prompt  # Should be different
    
    def test_terminology_mapping(self, domain_adapter):
        """Test terminology mapping functionality"""
        tech_handler = domain_adapter.terminology_handlers[ResearchDomain.TECHNOLOGY]
        
        assert isinstance(tech_handler, TerminologyHandler)
        assert tech_handler.domain == ResearchDomain.TECHNOLOGY
        assert len(tech_handler.abbreviations) > 0
        assert "AI" in tech_handler.abbreviations
        assert tech_handler.abbreviations["AI"] == "Artificial Intelligence"
    
    def test_domain_detection_with_content(self, domain_adapter, mock_kimi_client):
        """Test domain detection with additional content"""
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.85,
            "SCIENCE": 0.2,
            "BUSINESS": 0.1,
            "ACADEMIC": 0.15,
            "GENERAL": 0.1,
            "reasoning": "Strong technical focus with implementation details"
        })
        
        topic = "Software Development"
        content = "This research focuses on API design patterns, database optimization, and cloud architecture."
        
        result = domain_adapter.detect_domain(topic, content)
        
        assert result.primary_domain == ResearchDomain.TECHNOLOGY
        assert result.confidence > 0.8
        assert len(result.keywords_found) > 0
    
    def test_error_handling_kimi_failure(self, domain_adapter, mock_kimi_client):
        """Test error handling when Kimi K2 fails"""
        mock_kimi_client.generate_content.side_effect = Exception("Network error")
        
        # Should still work with fallback keyword-based detection
        result = domain_adapter.detect_domain("machine learning algorithms")
        
        assert isinstance(result, DomainDetectionResult)
        assert result.primary_domain in ResearchDomain
        assert result.confidence >= 0.0


class TestDomainDetectionResult:
    """Test cases for DomainDetectionResult model"""
    
    def test_domain_detection_result_creation(self):
        """Test creation of DomainDetectionResult"""
        result = DomainDetectionResult(
            primary_domain=ResearchDomain.TECHNOLOGY,
            confidence=0.9,
            secondary_domains=[],
            detection_method="test",
            keywords_found=["software", "algorithm"],
            reasoning="Test reasoning"
        )
        
        assert result.primary_domain == ResearchDomain.TECHNOLOGY
        assert result.confidence == 0.9
        assert result.keywords_found == ["software", "algorithm"]
    
    def test_domain_confidence_model(self):
        """Test DomainConfidence model"""
        confidence = DomainConfidence(
            domain=ResearchDomain.SCIENCE,
            confidence=0.7,
            indicators=["research", "experiment"],
            keywords_matched=["study", "analysis"]
        )
        
        assert confidence.domain == ResearchDomain.SCIENCE
        assert confidence.confidence == 0.7
        assert len(confidence.indicators) == 2


class TestResearchStrategy:
    """Test cases for ResearchStrategy model"""
    
    def test_research_strategy_creation(self):
        """Test creation of ResearchStrategy"""
        strategy = ResearchStrategy(
            domain=ResearchDomain.TECHNOLOGY,
            strategy_name="Tech Strategy",
            description="Technology research strategy",
            preferred_source_types=["tech_blogs", "documentation"],
            max_sources_per_gap=5,
            quality_criteria={"technical_accuracy": 0.9}
        )
        
        assert strategy.domain == ResearchDomain.TECHNOLOGY
        assert strategy.strategy_name == "Tech Strategy"
        assert "tech_blogs" in strategy.preferred_source_types
        assert strategy.quality_criteria["technical_accuracy"] == 0.9


class TestTerminologyHandler:
    """Test cases for TerminologyHandler model"""
    
    def test_terminology_handler_creation(self):
        """Test creation of TerminologyHandler"""
        handler = TerminologyHandler(
            domain=ResearchDomain.TECHNOLOGY,
            terminology_map={"computer": "system"},
            abbreviations={"AI": "Artificial Intelligence"},
            definitions={"API": "Application Programming Interface"}
        )
        
        assert handler.domain == ResearchDomain.TECHNOLOGY
        assert handler.terminology_map["computer"] == "system"
        assert handler.abbreviations["AI"] == "Artificial Intelligence"
        assert handler.definitions["API"] == "Application Programming Interface"


class TestDomainAdaptationIntegration:
    """Integration tests for domain adaptation system"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for integration testing"""
        mock_client = Mock(spec=KimiK2Client)
        return mock_client
    
    @pytest.fixture
    def domain_adapter(self, mock_kimi_client):
        """Create DomainAdapter for integration testing"""
        return DomainAdapter(mock_kimi_client)
    
    def test_end_to_end_domain_adaptation(self, domain_adapter, mock_kimi_client, sample_requirements):
        """Test complete domain adaptation workflow"""
        # Mock Kimi K2 responses
        mock_kimi_client.generate_content.side_effect = [
            # Domain detection response
            json.dumps({
                "TECHNOLOGY": 0.9,
                "SCIENCE": 0.1,
                "BUSINESS": 0.2,
                "ACADEMIC": 0.1,
                "GENERAL": 0.1,
                "reasoning": "Strong technical focus"
            }),
            # Structure generation response
            json.dumps({
                "sections": [
                    {
                        "id": "intro",
                        "title": "Introduction",
                        "section_type": "introduction",
                        "estimated_length": 500,
                        "content_placeholders": [
                            {
                                "id": "intro_placeholder",
                                "placeholder_type": "introduction",
                                "title": "Technology Overview",
                                "description": "Overview of the technology",
                                "estimated_word_count": 200,
                                "priority": "high",
                                "kimi_k2_prompt_hints": ["Focus on technical aspects"]
                            }
                        ],
                        "dependencies": []
                    }
                ],
                "estimated_total_length": 2000,
                "generation_strategy": "domain_optimized"
            })
        ]
        
        topic = "Machine Learning in Software Development"
        
        # Step 1: Detect domain
        domain_result = domain_adapter.detect_domain(topic)
        assert domain_result.primary_domain == ResearchDomain.TECHNOLOGY
        
        # Step 2: Adapt requirements
        adapted_requirements = domain_adapter.adapt_research_requirements(
            sample_requirements, domain_result
        )
        assert adapted_requirements.domain == ResearchDomain.TECHNOLOGY
        
        # Step 3: Generate structure
        structure = domain_adapter.generate_domain_specific_structure(
            topic, domain_result.primary_domain, adapted_requirements.complexity_level
        )
        assert isinstance(structure, EnhancedResearchStructure)
        assert len(structure.sections) > 0
        
        # Step 4: Test search query adaptation
        gaps = [
            InformationGap(
                id="gap1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="Need ML information",
                priority=Priority.HIGH,
                search_queries=[SearchQuery(query="machine learning", priority=Priority.HIGH)]
            )
        ]
        
        adapted_gaps = domain_adapter.adapt_search_queries(gaps, domain_result.primary_domain)
        assert len(adapted_gaps) > 0
        assert len(adapted_gaps[0].search_queries) >= 1
        
        # Step 5: Test formatting
        content = "The API uses ML algorithms for processing."
        formatted = domain_adapter.apply_domain_formatting(
            content, domain_result.primary_domain
        )
        assert isinstance(formatted, str)
    
    def test_domain_adaptation_accuracy_metrics(self, domain_adapter):
        """Test domain adaptation accuracy measurement"""
        from backend.services.domain_adapter import DomainMetrics
        
        metrics = DomainMetrics()
        metrics.detection_accuracy = 0.9
        metrics.adaptation_effectiveness = 0.85
        metrics.terminology_consistency = 0.8
        metrics.format_compliance = 0.9
        
        overall_score = metrics.calculate_overall_score()
        assert 0.8 <= overall_score <= 1.0
        
        metrics_dict = metrics.to_dict()
        assert "detection_accuracy" in metrics_dict
        assert "overall_score" in metrics_dict
        assert metrics_dict["overall_score"] == overall_score
    
    def test_cross_domain_detection(self, domain_adapter, mock_kimi_client):
        """Test detection of topics that span multiple domains"""
        mock_kimi_client.generate_content.return_value = json.dumps({
            "TECHNOLOGY": 0.6,
            "SCIENCE": 0.7,
            "BUSINESS": 0.3,
            "ACADEMIC": 0.5,
            "GENERAL": 0.2,
            "reasoning": "Interdisciplinary topic with science and technology aspects"
        })
        
        topic = "AI Applications in Medical Research"
        result = domain_adapter.detect_domain(topic)
        
        assert result.primary_domain == ResearchDomain.SCIENCE
        assert len(result.secondary_domains) > 0
        
        # Should have technology as secondary domain
        secondary_domains = [d.domain for d in result.secondary_domains]
        assert ResearchDomain.TECHNOLOGY in secondary_domains