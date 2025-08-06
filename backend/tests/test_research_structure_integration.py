"""
Integration tests for research structure and content models with Kimi K2.
These tests demonstrate the complete workflow of generating research structures,
content, and quality assessments.
"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock, patch

from models.core import (
    ResearchDomain, ComplexityLevel, Draft, QualityMetrics,
    ResearchRequirements, DraftMetadata
)
from models.research_structure import (
    EnhancedResearchStructure, get_domain_template, get_prompt_template
)
from services.kimi_k2_research_generator import (
    KimiK2ResearchStructureGenerator, KimiK2ContentGenerator, KimiK2QualityAssessor
)
from services.kimi_k2_client import KimiK2Client, KimiK2Response

class TestResearchStructureIntegration:
    """Integration tests for the complete research structure workflow"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kimi_client = MagicMock(spec=KimiK2Client)
        self.mock_kimi_client.model = "test-model"
        
        self.structure_generator = KimiK2ResearchStructureGenerator(self.mock_kimi_client)
        self.content_generator = KimiK2ContentGenerator(self.mock_kimi_client)
        self.quality_assessor = KimiK2QualityAssessor(self.mock_kimi_client)
    
    @pytest.mark.asyncio
    async def test_complete_research_workflow(self):
        """Test the complete workflow from structure generation to quality assessment"""
        
        # Step 1: Generate research structure
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
                            "title": "AI Overview",
                            "description": "Overview of AI in healthcare",
                            "estimated_word_count": 250,
                            "priority": "high",
                            "kimi_k2_prompt_hints": ["Focus on current applications"]
                        }
                    ],
                    "dependencies": []
                },
                {
                    "id": "analysis",
                    "title": "Analysis",
                    "section_type": "analysis",
                    "estimated_length": 800,
                    "content_placeholders": [
                        {
                            "id": "benefits",
                            "placeholder_type": "analysis",
                            "title": "Benefits Analysis",
                            "description": "Analysis of AI benefits in healthcare",
                            "estimated_word_count": 400,
                            "priority": "high",
                            "kimi_k2_prompt_hints": ["Include specific examples"]
                        }
                    ],
                    "dependencies": ["intro"]
                }
            ],
            "estimated_total_length": 1300,
            "generation_strategy": {"method": "domain_optimized", "confidence": 0.9}
        }
        
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            return_value=mock_structure_data
        )
        
        structure = await self.structure_generator.generate_research_structure(
            topic="AI in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            target_length=1300
        )
        
        assert isinstance(structure, EnhancedResearchStructure)
        assert len(structure.sections) == 2
        assert structure.domain == ResearchDomain.TECHNOLOGY
        
        # Step 2: Generate content for each section
        mock_content_responses = [
            KimiK2Response(content="# AI Overview\n\nArtificial Intelligence in healthcare represents..."),
            KimiK2Response(content="# Benefits Analysis\n\nThe benefits of AI in healthcare include...")
        ]
        
        self.mock_kimi_client.generate_text = AsyncMock(side_effect=mock_content_responses)
        
        # Generate content for all sections
        section_contents = {}
        for section in structure.sections:
            content = await self.content_generator.generate_section_content(
                section=section,
                research_topic="AI in Healthcare",
                domain=ResearchDomain.TECHNOLOGY,
                previous_sections_content=section_contents
            )
            section_contents[section.id] = content
            assert len(content) > 0
            assert section.title in content
        
        # Step 3: Create a draft with the generated content
        draft = Draft(
            id="test_draft_integration",
            topic="AI in Healthcare",
            structure=structure,
            content=section_contents,
            metadata=DraftMetadata(),
            quality_score=0.0,
            iteration=1
        )
        
        # Step 4: Assess the quality of the draft
        mock_assessment_data = {
            "criteria_scores": {
                "completeness": 0.85,
                "coherence": 0.80,
                "accuracy": 0.90,
                "citation_quality": 0.70
            },
            "overall_score": 0.81,
            "detailed_feedback": "Good comprehensive draft with strong technical content. Could benefit from more citations.",
            "improvement_suggestions": [
                "Add more peer-reviewed citations",
                "Expand on implementation challenges",
                "Include more recent case studies"
            ],
            "confidence": 0.88
        }
        
        # Reset mock for structured response (used by quality assessor)
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            return_value=mock_assessment_data
        )
        
        assessment = await self.quality_assessor.assess_draft_quality(draft)
        
        assert assessment.overall_score == 0.81
        assert assessment.criteria_scores["completeness"] == 0.85
        assert len(assessment.improvement_suggestions) == 3
        assert assessment.kimi_k2_confidence == 0.88
        
        # Step 5: Generate improvement suggestions
        suggestions = await self.quality_assessor.suggest_improvements(assessment, draft)
        
        assert len(suggestions) >= 3  # Should include original + generated suggestions
        assert any("citations" in s.lower() for s in suggestions)
        
        # Verify the complete workflow
        assert draft.topic == "AI in Healthcare"
        assert len(draft.content) == 2  # Two sections
        assert all(len(content) > 0 for content in draft.content.values())
        assert assessment.draft_id == draft.id
    
    @pytest.mark.asyncio
    async def test_domain_specific_workflow(self):
        """Test workflow with different research domains"""
        
        domains_to_test = [
            ResearchDomain.SCIENCE,
            ResearchDomain.BUSINESS,
            ResearchDomain.TECHNOLOGY
        ]
        
        for domain in domains_to_test:
            # Get domain template
            template = get_domain_template(domain)
            assert template.domain == domain
            
            # Mock structure generation for this domain
            mock_structure_data = {
                "sections": [
                    {
                        "id": f"{domain.value}_intro",
                        "title": "Introduction",
                        "section_type": "introduction",
                        "estimated_length": 400,
                        "content_placeholders": [
                            {
                                "id": "overview",
                                "placeholder_type": "introduction",
                                "title": f"{domain.value.title()} Overview",
                                "description": f"Overview for {domain.value} domain",
                                "estimated_word_count": 200,
                                "priority": "medium"
                            }
                        ],
                        "dependencies": []
                    }
                ],
                "estimated_total_length": 400,
                "generation_strategy": {"method": f"{domain.value}_optimized"}
            }
            
            self.mock_kimi_client.generate_structured_response = AsyncMock(
                return_value=mock_structure_data
            )
            
            structure = await self.structure_generator.generate_research_structure(
                topic=f"Test Topic for {domain.value}",
                domain=domain,
                complexity_level=ComplexityLevel.BASIC,
                target_length=400
            )
            
            assert structure.domain == domain
            assert len(structure.sections) >= 1
            assert structure.domain_template.domain == domain
    
    @pytest.mark.asyncio
    async def test_fallback_behavior(self):
        """Test fallback behavior when Kimi K2 is unavailable"""
        
        # Mock Kimi K2 failures
        self.mock_kimi_client.generate_structured_response = AsyncMock(
            side_effect=Exception("API unavailable")
        )
        self.mock_kimi_client.generate_text = AsyncMock(
            side_effect=Exception("API unavailable")
        )
        
        # Structure generation should fall back to templates
        structure = await self.structure_generator.generate_research_structure(
            topic="Test Topic",
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.BASIC,
            target_length=1000
        )
        
        assert isinstance(structure, EnhancedResearchStructure)
        assert structure.generation_metadata["generation_method"] == "fallback_template"
        assert len(structure.sections) >= 1
        
        # Content generation should fall back to placeholders
        section = structure.sections[0]
        if section.content_placeholders:
            placeholder = section.content_placeholders[0]
            content = await self.content_generator.generate_placeholder_content(
                placeholder=placeholder,
                section=section,
                research_topic="Test Topic",
                domain=ResearchDomain.GENERAL
            )
            
            assert "Content placeholder" in content
            assert str(placeholder.estimated_word_count) in content
        
        # Create a simple draft for quality assessment fallback
        draft = Draft(
            id="fallback_test",
            topic="Test Topic",
            structure=structure,
            content={section.id: "Test content" for section in structure.sections},
            metadata=DraftMetadata(),
            quality_score=0.0,
            iteration=1
        )
        
        # Quality assessment should fall back to heuristics
        assessment = await self.quality_assessor.assess_draft_quality(draft)
        
        assert "Fallback assessment" in assessment.detailed_feedback
        assert assessment.kimi_k2_confidence == 0.5  # Lower confidence for fallback
        assert 0.0 <= assessment.overall_score <= 1.0
    
    def test_prompt_template_coverage(self):
        """Test that all required prompt templates are available"""
        
        required_templates = [
            "structure_generation",
            "content_placeholder", 
            "quality_assessment"
        ]
        
        for template_id in required_templates:
            template = get_prompt_template(template_id)
            assert template is not None, f"Template {template_id} not found"
            assert len(template.variables) > 0, f"Template {template_id} has no variables"
            assert template.prompt_template, f"Template {template_id} has empty prompt"
    
    def test_domain_template_coverage(self):
        """Test that all research domains have templates"""
        
        for domain in ResearchDomain:
            template = get_domain_template(domain)
            assert template is not None, f"No template for domain {domain}"
            assert template.domain == domain
            assert len(template.default_sections) > 0
            assert len(template.quality_criteria) > 0

if __name__ == "__main__":
    pytest.main([__file__])