"""
Tests for research structure and content models with Kimi K2 optimization.
"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from models.core import (
    ResearchDomain, ComplexityLevel, Draft, QualityMetrics,
    ResearchRequirements, DraftMetadata
)
from models.research_structure import (
    EnhancedResearchStructure, EnhancedSection, ContentPlaceholder,
    DomainSpecificTemplate, KimiK2PromptTemplate, KimiK2QualityAssessment,
    QualityAssessmentCriteria, ResearchSectionType, ContentPlaceholderType,
    Priority, get_domain_template, get_prompt_template, DOMAIN_TEMPLATES,
    KIMI_K2_PROMPT_TEMPLATES
)
from services.kimi_k2_research_generator import (
    KimiK2ResearchStructureGenerator, KimiK2ContentGenerator, KimiK2QualityAssessor
)
from services.kimi_k2_client import KimiK2Client, KimiK2Response

class TestResearchStructureModels:
    """Test research structure data models"""
    
    def test_content_placeholder_creation(self):
        """Test ContentPlaceholder model creation and validation"""
        placeholder = ContentPlaceholder(
            id="intro_overview",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Technology Overview",
            description="Overview of the technology being researched",
            estimated_word_count=200,
            priority=Priority.HIGH,
            kimi_k2_prompt_hints=[
                "Focus on technical specifications",
                "Include current market adoption"
            ],
            required_elements=["definition", "key_features", "applications"]
        )
        
        assert placeholder.id == "intro_overview"
        assert placeholder.placeholder_type == ContentPlaceholderType.INTRODUCTION
        assert placeholder.priority == Priority.HIGH
        assert len(placeholder.kimi_k2_prompt_hints) == 2
        assert len(placeholder.required_elements) == 3
    
    def test_enhanced_section_creation(self):
        """Test EnhancedSection model creation and validation"""
        placeholder = ContentPlaceholder(
            id="placeholder_1",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Test Placeholder",
            description="Test description",
            estimated_word_count=100
        )
        
        section = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500,
            section_type=ResearchSectionType.INTRODUCTION,
            content_placeholders=[placeholder],
            kimi_k2_generation_context={"domain": "technology"},
            quality_requirements={"min_completeness": 0.8},
            dependencies=["background"]
        )
        
        assert section.section_type == ResearchSectionType.INTRODUCTION
        assert len(section.content_placeholders) == 1
        assert section.dependencies == ["background"]
        assert section.quality_requirements["min_completeness"] == 0.8
    
    def test_enhanced_section_placeholder_validation(self):
        """Test that duplicate placeholder IDs are rejected"""
        placeholder1 = ContentPlaceholder(
            id="duplicate_id",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Placeholder 1",
            description="Description 1",
            estimated_word_count=100
        )
        
        placeholder2 = ContentPlaceholder(
            id="duplicate_id",  # Same ID
            placeholder_type=ContentPlaceholderType.BACKGROUND,
            title="Placeholder 2",
            description="Description 2",
            estimated_word_count=150
        )
        
        with pytest.raises(ValueError, match="Content placeholder IDs must be unique"):
            EnhancedSection(
                id="test_section",
                title="Test Section",
                estimated_length=500,
                content_placeholders=[placeholder1, placeholder2]
            )
    
    def test_enhanced_research_structure_creation(self):
        """Test EnhancedResearchStructure creation and methods"""
        placeholder = ContentPlaceholder(
            id="placeholder_1",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Test Placeholder",
            description="Test description",
            estimated_word_count=100,
            priority=Priority.HIGH
        )
        
        section = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500,
            section_type=ResearchSectionType.INTRODUCTION,
            content_placeholders=[placeholder]
        )
        
        structure = EnhancedResearchStructure(
            sections=[section],
            estimated_length=2000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        # Test methods
        assert structure.get_section_by_type(ResearchSectionType.INTRODUCTION) == section
        assert structure.get_section_by_type(ResearchSectionType.METHODOLOGY) is None
        assert structure.calculate_total_placeholders() == 1
        
        high_priority_placeholders = structure.get_high_priority_placeholders()
        assert len(high_priority_placeholders) == 1
        assert high_priority_placeholders[0].priority == Priority.HIGH
    
    def test_kimi_k2_prompt_template_formatting(self):
        """Test KimiK2PromptTemplate formatting"""
        template = KimiK2PromptTemplate(
            template_id="test_template",
            domain=ResearchDomain.TECHNOLOGY,
            section_type=ResearchSectionType.INTRODUCTION,
            prompt_template="Generate content for {topic} in {domain} domain with {complexity} complexity",
            variables=["topic", "domain", "complexity"],
            expected_output_format="text"
        )
        
        formatted = template.format_prompt(
            topic="AI in Healthcare",
            domain="technology",
            complexity="intermediate"
        )
        
        expected = "Generate content for AI in Healthcare in technology domain with intermediate complexity"
        assert formatted == expected
    
    def test_kimi_k2_prompt_template_missing_variable(self):
        """Test KimiK2PromptTemplate with missing variables"""
        template = KimiK2PromptTemplate(
            template_id="test_template",
            domain=ResearchDomain.TECHNOLOGY,
            section_type=ResearchSectionType.INTRODUCTION,
            prompt_template="Generate content for {topic} in {domain} domain",
            variables=["topic", "domain"]
        )
        
        with pytest.raises(ValueError, match="Missing required variable"):
            template.format_prompt(topic="AI in Healthcare")  # Missing domain
    
    def test_quality_assessment_criteria_validation(self):
        """Test QualityAssessmentCriteria validation"""
        # Valid criteria
        criteria = QualityAssessmentCriteria(
            criteria_id="test_criteria",
            name="Test Criteria",
            description="Test description",
            weight=0.5,
            kimi_k2_evaluation_prompt="Test prompt",
            expected_score_range=(0.2, 0.8)
        )
        
        assert criteria.expected_score_range == (0.2, 0.8)
        
        # Invalid score range
        with pytest.raises(ValueError, match="Score range must be"):
            QualityAssessmentCriteria(
                criteria_id="invalid_criteria",
                name="Invalid Criteria",
                description="Invalid description",
                weight=0.5,
                kimi_k2_evaluation_prompt="Test prompt",
                expected_score_range=(0.8, 0.2)  # min > max
            )
    
    def test_kimi_k2_quality_assessment_score_validation(self):
        """Test KimiK2QualityAssessment score validation"""
        # Valid assessment
        assessment = KimiK2QualityAssessment(
            assessment_id="test_assessment",
            draft_id="test_draft",
            criteria_scores={
                "completeness": 0.8,
                "coherence": 0.7,
                "accuracy": 0.9
            },
            overall_score=0.8,  # Close to calculated average
            detailed_feedback="Good quality draft"
        )
        
        assert assessment.overall_score == 0.8
        
        # Invalid overall score (too far from calculated)
        with pytest.raises(ValueError, match="Overall score .* doesn't match calculated score"):
            KimiK2QualityAssessment(
                assessment_id="invalid_assessment",
                draft_id="test_draft",
                criteria_scores={
                    "completeness": 0.8,
                    "coherence": 0.7,
                    "accuracy": 0.9
                },
                overall_score=0.5,  # Too far from calculated average (0.8)
                detailed_feedback="Invalid assessment"
            )

class TestDomainTemplates:
    """Test domain-specific templates"""
    
    def test_get_domain_template_technology(self):
        """Test getting technology domain template"""
        template = get_domain_template(ResearchDomain.TECHNOLOGY)
        
        assert template.domain == ResearchDomain.TECHNOLOGY
        assert template.template_name == "Technology Research"
        assert len(template.default_sections) >= 2
        assert "technical_accuracy" in template.quality_criteria
    
    def test_get_domain_template_science(self):
        """Test getting science domain template"""
        template = get_domain_template(ResearchDomain.SCIENCE)
        
        assert template.domain == ResearchDomain.SCIENCE
        assert template.template_name == "Scientific Research"
        assert "scientific_rigor" in template.quality_criteria
        assert template.quality_criteria["scientific_rigor"] >= 0.9
    
    def test_get_domain_template_business(self):
        """Test getting business domain template"""
        template = get_domain_template(ResearchDomain.BUSINESS)
        
        assert template.domain == ResearchDomain.BUSINESS
        assert template.template_name == "Business Analysis"
        assert "business_relevance" in template.quality_criteria
    
    def test_get_domain_template_fallback(self):
        """Test fallback for unknown domain"""
        # Should fallback to GENERAL template
        template = get_domain_template(ResearchDomain.GENERAL)
        assert template is not None

class TestPromptTemplates:
    """Test Kimi K2 prompt templates"""
    
    def test_get_prompt_template_structure_generation(self):
        """Test getting structure generation template"""
        template = get_prompt_template("structure_generation")
        
        assert template is not None
        assert template.template_id == "structure_gen"
        assert "topic" in template.variables
        assert "domain" in template.variables
        assert template.expected_output_format == "json"
    
    def test_get_prompt_template_content_placeholder(self):
        """Test getting content placeholder template"""
        template = get_prompt_template("content_placeholder")
        
        assert template is not None
        assert template.template_id == "content_placeholder"
        assert "section_title" in template.variables
        assert template.expected_output_format == "markdown"
    
    def test_get_prompt_template_quality_assessment(self):
        """Test getting quality assessment template"""
        template = get_prompt_template("quality_assessment")
        
        assert template is not None
        assert template.template_id == "quality_assessment"
        assert "topic" in template.variables
        assert template.expected_output_format == "json"
    
    def test_get_prompt_template_nonexistent(self):
        """Test getting non-existent template"""
        template = get_prompt_template("nonexistent_template")
        assert template is None

class TestKimiK2ResearchStructureGenerator:
    """Test Kimi K2 research structure generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kimi_client = MagicMock(spec=KimiK2Client)
        self.mock_kimi_client.model = "test-model"  # Add model attribute
        self.generator = KimiK2ResearchStructureGenerator(self.mock_kimi_client)
    
    @pytest.mark.asyncio
    async def test_generate_research_structure_success(self):
        """Test successful research structure generation"""
        # Mock Kimi K2 response
        mock_structure_data = {
            "sections": [
                {
                    "id": "intro",
                    "title": "Introduction",
                    "section_type": "introduction",
                    "estimated_length": 500,
                    "content_placeholders": [
                        {
                            "id": "overview",
                            "placeholder_type": "introduction",
                            "title": "Overview",
                            "description": "Technology overview",
                            "estimated_word_count": 200,
                            "priority": "medium",
                            "kimi_k2_prompt_hints": ["Focus on innovation"]
                        }
                    ],
                    "dependencies": []
                }
            ],
            "estimated_total_length": 2000,
            "generation_strategy": {"method": "structured_approach", "confidence": 0.9}
        }
        
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            return_value=mock_structure_data
        )
        
        structure = await self.generator.generate_research_structure(
            topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            target_length=2000
        )
        
        assert isinstance(structure, EnhancedResearchStructure)
        assert len(structure.sections) == 1
        assert structure.sections[0].title == "Introduction"
        assert structure.domain == ResearchDomain.TECHNOLOGY
        assert structure.estimated_length == 2000
        
        # Verify Kimi K2 client was called
        self.mock_kimi_client.generate_structured_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_research_structure_fallback(self):
        """Test fallback when Kimi K2 fails"""
        # Mock Kimi K2 failure
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        structure = await self.generator.generate_research_structure(
            topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            target_length=2000
        )
        
        assert isinstance(structure, EnhancedResearchStructure)
        assert len(structure.sections) >= 1  # Should have fallback sections
        assert structure.generation_metadata["generation_method"] == "fallback_template"
    
    @pytest.mark.asyncio
    async def test_create_enhanced_section(self):
        """Test creating enhanced section from data"""
        section_data = {
            "id": "methodology",
            "title": "Methodology",
            "section_type": "methodology",
            "estimated_length": 800,
            "content_placeholders": [
                {
                    "id": "approach",
                    "placeholder_type": "methodology",
                    "title": "Research Approach",
                    "description": "Description of research approach",
                    "estimated_word_count": 300,
                    "priority": "high",
                    "kimi_k2_prompt_hints": ["Include quantitative methods"]
                }
            ],
            "dependencies": ["intro"]
        }
        
        section = await self.generator._create_enhanced_section(
            section_data, ResearchDomain.SCIENCE
        )
        
        assert isinstance(section, EnhancedSection)
        assert section.title == "Methodology"
        assert section.section_type == ResearchSectionType.METHODOLOGY
        assert len(section.content_placeholders) == 1
        assert section.dependencies == ["intro"]

class TestKimiK2ContentGenerator:
    """Test Kimi K2 content generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kimi_client = MagicMock(spec=KimiK2Client)
        self.generator = KimiK2ContentGenerator(self.mock_kimi_client)
    
    @pytest.mark.asyncio
    async def test_generate_placeholder_content_success(self):
        """Test successful placeholder content generation"""
        placeholder = ContentPlaceholder(
            id="overview",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Technology Overview",
            description="Overview of AI technology",
            estimated_word_count=200,
            kimi_k2_prompt_hints=["Focus on innovation", "Include market impact"]
        )
        
        section = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500,
            content_placeholders=[placeholder]
        )
        
        # Mock Kimi K2 response
        mock_response = KimiK2Response(
            content="# Technology Overview\n\nAI technology represents a significant innovation...",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            model="moonshot-v1-8k"
        )
        
        self.mock_kimi_client.generate_text = AsyncMock(return_value=mock_response)
        
        content = await self.generator.generate_placeholder_content(
            placeholder=placeholder,
            section=section,
            research_topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY
        )
        
        assert "Technology Overview" in content
        assert "AI technology" in content
        assert len(content) > 50  # Should have substantial content
        
        # Verify Kimi K2 client was called
        self.mock_kimi_client.generate_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_placeholder_content_fallback(self):
        """Test fallback content generation"""
        placeholder = ContentPlaceholder(
            id="overview",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Technology Overview",
            description="Overview of AI technology",
            estimated_word_count=200,
            required_elements=["definition", "applications"],
            kimi_k2_prompt_hints=["Focus on innovation"]
        )
        
        section = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500,
            content_placeholders=[placeholder]
        )
        
        # Mock Kimi K2 failure
        self.mock_kimi_client.generate_text = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        content = await self.generator.generate_placeholder_content(
            placeholder=placeholder,
            section=section,
            research_topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY
        )
        
        # Should contain fallback content
        assert "Technology Overview" in content
        assert "definition" in content
        assert "applications" in content
        assert "Focus on innovation" in content
        assert "200 words needed" in content
    
    @pytest.mark.asyncio
    async def test_generate_section_content(self):
        """Test generating complete section content"""
        placeholder1 = ContentPlaceholder(
            id="overview",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Overview",
            description="Technology overview",
            estimated_word_count=150
        )
        
        placeholder2 = ContentPlaceholder(
            id="objectives",
            placeholder_type=ContentPlaceholderType.INTRODUCTION,
            title="Objectives",
            description="Research objectives",
            estimated_word_count=100
        )
        
        section = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500,
            content_placeholders=[placeholder1, placeholder2],
            dependencies=["background"]
        )
        
        # Mock Kimi K2 responses
        mock_responses = [
            KimiK2Response(content="Overview content here..."),
            KimiK2Response(content="Objectives content here...")
        ]
        
        self.mock_kimi_client.generate_text = AsyncMock(side_effect=mock_responses)
        
        content = await self.generator.generate_section_content(
            section=section,
            research_topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            previous_sections_content={"background": "Background content..."}
        )
        
        assert "# Introduction" in content
        assert "Overview content here" in content
        assert "Objectives content here" in content
        
        # Should have called generate_text twice (once per placeholder)
        assert self.mock_kimi_client.generate_text.call_count == 2

class TestKimiK2QualityAssessor:
    """Test Kimi K2 quality assessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kimi_client = MagicMock(spec=KimiK2Client)
        self.assessor = KimiK2QualityAssessor(self.mock_kimi_client)
    
    def create_test_draft(self) -> Draft:
        """Create a test draft for assessment"""
        section = EnhancedSection(
            id="intro",
            title="Introduction",
            estimated_length=500
        )
        
        structure = EnhancedResearchStructure(
            sections=[section],
            estimated_length=2000,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        return Draft(
            id="test_draft",
            topic="AI in Healthcare",
            structure=structure,
            content={"intro": "Introduction content with detailed analysis..."},
            metadata=DraftMetadata(),
            quality_score=0.0,
            iteration=1
        )
    
    @pytest.mark.asyncio
    async def test_assess_draft_quality_success(self):
        """Test successful draft quality assessment"""
        draft = self.create_test_draft()
        
        # Mock Kimi K2 assessment response
        mock_assessment_data = {
            "criteria_scores": {
                "completeness": 0.8,
                "coherence": 0.7,
                "accuracy": 0.9,
                "citation_quality": 0.6
            },
            "overall_score": 0.75,
            "detailed_feedback": "Good draft with room for improvement in citations",
            "improvement_suggestions": [
                "Add more citations",
                "Improve section transitions"
            ],
            "confidence": 0.85
        }
        
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            return_value=mock_assessment_data
        )
        
        assessment = await self.assessor.assess_draft_quality(draft)
        
        assert isinstance(assessment, KimiK2QualityAssessment)
        assert assessment.draft_id == "test_draft"
        assert assessment.overall_score == 0.75
        assert assessment.criteria_scores["completeness"] == 0.8
        assert len(assessment.improvement_suggestions) == 2
        assert assessment.kimi_k2_confidence == 0.85
        
        # Verify Kimi K2 client was called
        self.mock_kimi_client.generate_structured_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assess_draft_quality_fallback(self):
        """Test fallback quality assessment"""
        draft = self.create_test_draft()
        
        # Mock Kimi K2 failure
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        assessment = await self.assessor.assess_draft_quality(draft)
        
        assert isinstance(assessment, KimiK2QualityAssessment)
        assert assessment.draft_id == "test_draft"
        assert 0.0 <= assessment.overall_score <= 1.0
        assert "Fallback assessment" in assessment.detailed_feedback
        assert len(assessment.improvement_suggestions) > 0
        assert assessment.kimi_k2_confidence == 0.5  # Lower confidence for fallback
    
    @pytest.mark.asyncio
    async def test_suggest_improvements(self):
        """Test improvement suggestions based on assessment"""
        draft = self.create_test_draft()
        
        assessment = KimiK2QualityAssessment(
            assessment_id="test_assessment",
            draft_id="test_draft",
            criteria_scores={
                "completeness": 0.6,  # Low
                "coherence": 0.8,     # Good
                "accuracy": 0.6,      # Low
                "citation_quality": 0.5  # Low
            },
            overall_score=0.625,
            detailed_feedback="Needs improvement",
            improvement_suggestions=["Original suggestion"]
        )
        
        suggestions = await self.assessor.suggest_improvements(assessment, draft)
        
        # Should include original suggestion plus specific ones based on low scores
        assert "Original suggestion" in suggestions
        assert any("detailed content" in s.lower() for s in suggestions)  # completeness
        assert any("factual claims" in s.lower() for s in suggestions)    # accuracy
        assert any("citations" in s.lower() for s in suggestions)         # citation_quality
        
        # Should not suggest coherence improvements (score was good)
        coherence_suggestions = [s for s in suggestions if "coherence" in s.lower() or "flow" in s.lower()]
        assert len(coherence_suggestions) <= 1  # At most the generic flow suggestion
    
    def test_prepare_draft_content(self):
        """Test preparing draft content for assessment"""
        draft = self.create_test_draft()
        
        content = self.assessor._prepare_draft_content(draft)
        
        assert "Topic: AI in Healthcare" in content
        assert "## Introduction" in content
        assert "Introduction content with detailed analysis" in content
    
    def test_assessment_criteria_initialization(self):
        """Test that assessment criteria are properly initialized"""
        criteria = self.assessor.assessment_criteria
        
        assert len(criteria) == 4
        criteria_ids = [c.criteria_id for c in criteria]
        assert "completeness" in criteria_ids
        assert "coherence" in criteria_ids
        assert "accuracy" in criteria_ids
        assert "citation_quality" in criteria_ids
        
        # Check weights sum to 1.0
        total_weight = sum(c.weight for c in criteria)
        assert abs(total_weight - 1.0) < 0.001

if __name__ == "__main__":
    pytest.main([__file__])