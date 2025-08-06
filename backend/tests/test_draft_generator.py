"""
Unit tests for draft generator functionality with Kimi K2 integration
"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime

from workflow.draft_generator import (
    KimiK2DraftGenerator, DraftGenerationError, draft_generator_node
)
from models.core import (
    TTDRState, ResearchRequirements, ResearchDomain, ComplexityLevel,
    Draft, ResearchStructure, Section
)
from services.kimi_k2_client import KimiK2Response, KimiK2Error

class TestKimiK2DraftGenerator:
    """Test KimiK2DraftGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = KimiK2DraftGenerator()
        self.test_topic = "Artificial Intelligence in Healthcare"
        self.test_requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.8
        )
    
    @pytest.mark.asyncio
    async def test_generate_initial_draft_success(self):
        """Test successful initial draft generation"""
        # Mock Kimi K2 responses
        structure_response = {
            "sections": [
                {
                    "id": "introduction",
                    "title": "Introduction",
                    "description": "Overview of AI in healthcare",
                    "estimated_length": 500
                },
                {
                    "id": "applications",
                    "title": "Applications",
                    "description": "Current AI applications in healthcare",
                    "estimated_length": 800
                }
            ],
            "total_estimated_length": 1300,
            "key_themes": ["machine learning", "diagnostics", "patient care"]
        }
        
        content_responses = [
            KimiK2Response(content="# Introduction\n\nAI in healthcare represents..."),
            KimiK2Response(content="# Applications\n\nCurrent applications include...")
        ]
        
        with patch.object(self.generator.kimi_client, 'generate_structured_response') as mock_structured:
            with patch.object(self.generator.kimi_client, 'generate_text') as mock_text:
                mock_structured.return_value = structure_response
                mock_text.side_effect = content_responses
                
                draft = await self.generator.generate_initial_draft(
                    self.test_topic, self.test_requirements
                )
                
                # Verify draft properties
                assert draft is not None
                assert hasattr(draft, 'topic')
                assert hasattr(draft, 'structure')
                assert draft.topic == self.test_topic
                assert len(draft.structure.sections) == 2
                assert draft.structure.complexity_level == ComplexityLevel.INTERMEDIATE
                assert draft.structure.domain == ResearchDomain.TECHNOLOGY
                assert draft.iteration == 0
                assert draft.quality_score == 0.3  # Initial low score
                
                # Verify content was generated
                assert "introduction" in draft.content
                assert "applications" in draft.content
                assert "AI in healthcare represents" in draft.content["introduction"]
                
                # Verify API calls
                mock_structured.assert_called_once()
                assert mock_text.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_initial_draft_kimi_error_fallback(self):
        """Test fallback to template when Kimi K2 fails"""
        with patch.object(self.generator.kimi_client, 'generate_structured_response') as mock_structured:
            mock_structured.side_effect = KimiK2Error("API error", 500, "server")
            
            draft = await self.generator.generate_initial_draft(
                self.test_topic, self.test_requirements
            )
            
            # Should still create a draft using fallback
            assert draft is not None
            assert hasattr(draft, 'topic')
            assert draft.topic == self.test_topic
            assert len(draft.structure.sections) > 0
            assert draft.structure.domain == ResearchDomain.TECHNOLOGY
    
    @pytest.mark.asyncio
    async def test_create_research_structure_success(self):
        """Test successful research structure creation"""
        mock_response = {
            "sections": [
                {
                    "id": "intro",
                    "title": "Introduction",
                    "description": "Overview section",
                    "estimated_length": 400
                }
            ],
            "total_estimated_length": 400,
            "key_themes": ["AI", "healthcare"]
        }
        
        with patch.object(self.generator.kimi_client, 'generate_structured_response') as mock_api:
            mock_api.return_value = mock_response
            
            structure = await self.generator._create_research_structure(
                self.test_topic, self.test_requirements
            )
            
            assert structure is not None
            assert hasattr(structure, 'sections')
            assert len(structure.sections) == 1
            assert structure.sections[0].id == "intro"
            assert structure.sections[0].title == "Introduction"
            assert structure.estimated_length == 400
            assert structure.complexity_level == ComplexityLevel.INTERMEDIATE
    
    def test_build_structure_prompt(self):
        """Test structure prompt building"""
        prompt = self.generator._build_structure_prompt(
            self.test_topic, self.test_requirements
        )
        
        assert self.test_topic in prompt
        assert "technology" in prompt.lower()
        assert "intermediate" in prompt.lower()
        assert "research structure" in prompt.lower()
        assert len(prompt) > 100  # Should be substantial
    
    @pytest.mark.asyncio
    async def test_generate_section_content_success(self):
        """Test successful section content generation"""
        section = Section(
            id="test_section",
            title="Test Section",
            estimated_length=300
        )
        structure = ResearchStructure(
            sections=[section],
            estimated_length=300,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        mock_response = KimiK2Response(content="# Test Section\n\nThis is test content...")
        
        with patch.object(self.generator.kimi_client, 'generate_text') as mock_api:
            mock_api.return_value = mock_response
            
            content = await self.generator._generate_section_content(
                self.test_topic, structure, self.test_requirements
            )
            
            assert "test_section" in content
            assert "This is test content" in content["test_section"]
    
    @pytest.mark.asyncio
    async def test_generate_single_section_content(self):
        """Test single section content generation"""
        section = Section(
            id="methodology",
            title="Methodology",
            estimated_length=500
        )
        structure = ResearchStructure(
            sections=[section],
            estimated_length=500,
            complexity_level=ComplexityLevel.ADVANCED,
            domain=ResearchDomain.SCIENCE
        )
        
        mock_response = KimiK2Response(content="# Methodology\n\nThe research methodology...")
        
        with patch.object(self.generator.kimi_client, 'generate_text') as mock_api:
            mock_api.return_value = mock_response
            
            content = await self.generator._generate_single_section_content(
                self.test_topic, section, structure, self.test_requirements
            )
            
            assert "Methodology" in content
            assert "research methodology" in content
            
            # Verify API call parameters
            mock_api.assert_called_once()
            call_args = mock_api.call_args
            assert call_args[1]['temperature'] == 0.6
            assert call_args[1]['max_tokens'] <= 1500
    
    def test_build_section_content_prompt(self):
        """Test section content prompt building"""
        section = Section(
            id="results",
            title="Results and Analysis",
            estimated_length=600
        )
        structure = ResearchStructure(
            sections=[
                section,
                Section(id="intro", title="Introduction", estimated_length=300)
            ],
            estimated_length=900,
            complexity_level=ComplexityLevel.EXPERT,
            domain=ResearchDomain.ACADEMIC
        )
        
        prompt = self.generator._build_section_content_prompt(
            self.test_topic, section, structure, self.test_requirements
        )
        
        assert "Results and Analysis" in prompt
        assert self.test_topic in prompt
        assert "academic" in prompt.lower()
        assert "600 words" in prompt
        assert "Introduction" in prompt  # Other section context
    
    def test_create_placeholder_content(self):
        """Test placeholder content creation"""
        section = Section(
            id="introduction",
            title="Introduction",
            estimated_length=400
        )
        
        content = self.generator._create_placeholder_content(section, self.test_topic)
        
        assert "Introduction" in content
        assert self.test_topic in content
        assert len(content) > 100  # Should be substantial
        assert "requires additional research" in content
    
    def test_create_fallback_structure_technology_domain(self):
        """Test fallback structure creation for technology domain"""
        structure = self.generator._create_fallback_structure(
            self.test_topic, self.test_requirements
        )
        
        # Check type and basic properties
        assert structure is not None
        assert hasattr(structure, 'domain')
        assert hasattr(structure, 'sections')
        assert structure.domain == ResearchDomain.TECHNOLOGY
        assert structure.complexity_level == ComplexityLevel.INTERMEDIATE
        assert len(structure.sections) > 0
        
        # Check for technology-specific sections
        section_ids = [s.id for s in structure.sections]
        assert "technical_overview" in section_ids or "implementation" in section_ids
    
    def test_create_fallback_structure_different_domains(self):
        """Test fallback structure for different domains"""
        domains_to_test = [
            ResearchDomain.SCIENCE,
            ResearchDomain.BUSINESS,
            ResearchDomain.ACADEMIC,
            ResearchDomain.GENERAL
        ]
        
        for domain in domains_to_test:
            requirements = ResearchRequirements(
                domain=domain,
                complexity_level=ComplexityLevel.INTERMEDIATE
            )
            
            structure = self.generator._create_fallback_structure(
                self.test_topic, requirements
            )
            
            assert structure.domain == domain
            assert len(structure.sections) > 0
            
            # Verify domain-specific sections
            section_titles = [s.title.lower() for s in structure.sections]
            if domain == ResearchDomain.SCIENCE:
                assert any("methodology" in title for title in section_titles)
            elif domain == ResearchDomain.BUSINESS:
                assert any("market" in title for title in section_titles)
    
    def test_complexity_multipliers(self):
        """Test complexity level affects structure length"""
        base_requirements = ResearchRequirements(
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.BASIC
        )
        
        basic_structure = self.generator._create_fallback_structure(
            self.test_topic, base_requirements
        )
        
        expert_requirements = ResearchRequirements(
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.EXPERT
        )
        
        expert_structure = self.generator._create_fallback_structure(
            self.test_topic, expert_requirements
        )
        
        # Expert should be longer than basic
        assert expert_structure.estimated_length > basic_structure.estimated_length

class TestDraftGeneratorNode:
    """Test draft_generator_node function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_state = {
            "topic": "Machine Learning in Finance",
            "requirements": ResearchRequirements(
                domain=ResearchDomain.BUSINESS,
                complexity_level=ComplexityLevel.ADVANCED
            ),
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 0,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
    
    @pytest.mark.asyncio
    async def test_draft_generator_node_success(self):
        """Test successful draft generator node execution"""
        mock_draft = Draft(
            id="test_draft",
            topic=self.test_state["topic"],
            structure=ResearchStructure(
                sections=[Section(id="intro", title="Introduction", estimated_length=300)],
                estimated_length=300,
                complexity_level=ComplexityLevel.ADVANCED,
                domain=ResearchDomain.BUSINESS
            ),
            content={"intro": "Introduction content"},
            iteration=0
        )
        
        with patch('workflow.draft_generator.KimiK2DraftGenerator') as mock_generator_class:
            mock_generator = mock_generator_class.return_value
            mock_generator.generate_initial_draft = AsyncMock(return_value=mock_draft)
            
            result_state = await draft_generator_node(self.test_state)
            
            # Verify state updates
            assert result_state["current_draft"] == mock_draft
            assert result_state["iteration_count"] == 0
            assert result_state["topic"] == self.test_state["topic"]
            assert len(result_state["error_log"]) == 0
            
            # Verify generator was called correctly
            mock_generator.generate_initial_draft.assert_called_once_with(
                self.test_state["topic"],
                self.test_state["requirements"]
            )
    
    @pytest.mark.asyncio
    async def test_draft_generator_node_error_handling(self):
        """Test error handling in draft generator node"""
        with patch('workflow.draft_generator.KimiK2DraftGenerator') as mock_generator_class:
            mock_generator = mock_generator_class.return_value
            mock_generator.generate_initial_draft = AsyncMock(
                side_effect=DraftGenerationError("Test error")
            )
            
            result_state = await draft_generator_node(self.test_state)
            
            # Should handle error gracefully
            assert result_state["current_draft"] is None
            assert result_state["iteration_count"] == 0
            assert len(result_state["error_log"]) == 1
            assert "Draft generation failed" in result_state["error_log"][0]
    
    @pytest.mark.asyncio
    async def test_draft_generator_node_preserves_other_state(self):
        """Test that node preserves other state fields"""
        # Add some existing state
        self.test_state["evolution_history"] = [{"test": "data"}]
        self.test_state["retrieved_info"] = [{"test": "info"}]
        
        mock_draft = Draft(
            id="test_draft",
            topic=self.test_state["topic"],
            structure=ResearchStructure(
                sections=[],
                estimated_length=0,
                complexity_level=ComplexityLevel.BASIC,
                domain=ResearchDomain.GENERAL
            ),
            content={},
            iteration=0
        )
        
        with patch('workflow.draft_generator.KimiK2DraftGenerator') as mock_generator_class:
            mock_generator = mock_generator_class.return_value
            mock_generator.generate_initial_draft = AsyncMock(return_value=mock_draft)
            
            result_state = await draft_generator_node(self.test_state)
            
            # Should preserve existing state
            assert result_state["evolution_history"] == [{"test": "data"}]
            assert result_state["retrieved_info"] == [{"test": "info"}]
            assert result_state["requirements"] == self.test_state["requirements"]

class TestDraftGeneratorIntegration:
    """Integration tests for draft generator with real-like scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_draft_generation_mock(self):
        """Test end-to-end draft generation with mocked Kimi K2"""
        generator = KimiK2DraftGenerator()
        topic = "Blockchain Technology in Supply Chain Management"
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.EXPERT,
            max_iterations=5,
            quality_threshold=0.9
        )
        
        # Mock comprehensive Kimi K2 responses
        structure_response = {
            "sections": [
                {
                    "id": "introduction",
                    "title": "Introduction to Blockchain in Supply Chain",
                    "description": "Overview and motivation",
                    "estimated_length": 600
                },
                {
                    "id": "technical_architecture",
                    "title": "Technical Architecture and Implementation",
                    "description": "Detailed technical analysis",
                    "estimated_length": 1200
                },
                {
                    "id": "use_cases",
                    "title": "Industry Use Cases and Applications",
                    "description": "Real-world implementations",
                    "estimated_length": 1000
                },
                {
                    "id": "challenges",
                    "title": "Challenges and Limitations",
                    "description": "Current obstacles and solutions",
                    "estimated_length": 800
                },
                {
                    "id": "future_directions",
                    "title": "Future Directions and Opportunities",
                    "description": "Emerging trends and predictions",
                    "estimated_length": 600
                }
            ],
            "total_estimated_length": 4200,
            "key_themes": ["decentralization", "transparency", "traceability", "smart contracts"]
        }
        
        content_responses = [
            KimiK2Response(content=f"# Introduction to Blockchain in Supply Chain\n\nBlockchain technology has emerged as a revolutionary approach to supply chain management, offering unprecedented levels of transparency, traceability, and security. This comprehensive analysis explores the integration of blockchain technology within supply chain ecosystems, examining both the technical foundations and practical implementations that are transforming how goods and information flow through global networks.\n\n## The Supply Chain Revolution\n\nTraditional supply chains face numerous challenges including lack of transparency, difficulty in tracking products, counterfeit goods, and inefficient processes. Blockchain technology addresses these issues by providing a distributed ledger system that creates an immutable record of transactions and product movements throughout the supply chain.\n\n## Research Scope and Objectives\n\nThis research examines the current state of blockchain implementation in supply chain management, analyzing successful use cases, technical architectures, and the challenges that organizations face when adopting this technology. The analysis provides insights for decision-makers considering blockchain integration and identifies future opportunities for innovation in this rapidly evolving field."),
            
            KimiK2Response(content=f"# Technical Architecture and Implementation\n\nThe technical implementation of blockchain in supply chain management requires careful consideration of architecture design, consensus mechanisms, and integration with existing systems. This section provides a detailed analysis of the technical components and implementation strategies.\n\n## Blockchain Architecture for Supply Chains\n\nSupply chain blockchain implementations typically utilize either public, private, or consortium blockchain networks, each offering different advantages depending on the specific use case and organizational requirements.\n\n### Private Blockchain Networks\n\nPrivate blockchains are often preferred for supply chain applications due to their controlled access, higher transaction throughput, and compliance with regulatory requirements. These networks allow organizations to maintain control over data while still benefiting from blockchain's immutability and transparency features.\n\n### Smart Contract Integration\n\nSmart contracts play a crucial role in automating supply chain processes, enabling automatic execution of agreements when predefined conditions are met. This automation reduces manual intervention, minimizes errors, and accelerates transaction processing throughout the supply chain.\n\n## Implementation Challenges\n\nTechnical implementation faces several challenges including scalability limitations, integration complexity with legacy systems, and the need for standardized protocols across different organizations and platforms."),
            
            KimiK2Response(content=f"# Industry Use Cases and Applications\n\nBlockchain technology has been successfully implemented across various industries, demonstrating its versatility and effectiveness in addressing supply chain challenges. This section examines real-world applications and their outcomes.\n\n## Food and Agriculture\n\nThe food industry has been an early adopter of blockchain technology, using it to enhance food safety, reduce fraud, and improve traceability from farm to table. Major retailers and food producers have implemented blockchain solutions to track products throughout the supply chain.\n\n### Walmart's Food Traceability Initiative\n\nWalmart has implemented a blockchain-based system that can trace the origin of food products in seconds rather than days, significantly improving response times during food safety incidents and recalls.\n\n## Pharmaceutical Industry\n\nThe pharmaceutical sector utilizes blockchain to combat counterfeit drugs, ensure medication authenticity, and maintain compliance with regulatory requirements for drug traceability.\n\n## Luxury Goods and Authentication\n\nLuxury brands employ blockchain technology to verify product authenticity, combat counterfeiting, and provide customers with proof of genuine products through digital certificates stored on the blockchain.\n\n## Manufacturing and Automotive\n\nManufacturing companies use blockchain to track components, verify supplier credentials, and ensure quality control throughout complex multi-tier supply chains."),
            
            KimiK2Response(content=f"# Challenges and Limitations\n\nDespite its potential benefits, blockchain implementation in supply chain management faces several significant challenges that organizations must address for successful adoption.\n\n## Scalability and Performance Issues\n\nCurrent blockchain networks face limitations in transaction throughput and processing speed, which can be problematic for high-volume supply chain operations requiring real-time data processing.\n\n## Integration Complexity\n\nIntegrating blockchain technology with existing enterprise resource planning (ERP) systems, warehouse management systems, and other legacy infrastructure presents significant technical and organizational challenges.\n\n## Cost and Resource Requirements\n\nImplementing blockchain solutions requires substantial investment in technology infrastructure, staff training, and ongoing maintenance, which can be prohibitive for smaller organizations.\n\n## Regulatory and Compliance Concerns\n\nThe regulatory landscape for blockchain technology is still evolving, creating uncertainty for organizations regarding compliance requirements and legal implications of blockchain-based supply chain systems.\n\n## Interoperability Challenges\n\nDifferent blockchain platforms and protocols may not be compatible with each other, creating silos and limiting the potential for industry-wide adoption and collaboration."),
            
            KimiK2Response(content=f"# Future Directions and Opportunities\n\nThe future of blockchain in supply chain management holds significant promise, with emerging technologies and evolving industry standards creating new opportunities for innovation and adoption.\n\n## Emerging Technologies Integration\n\nThe convergence of blockchain with other emerging technologies such as Internet of Things (IoT), artificial intelligence, and machine learning is creating new possibilities for intelligent, automated supply chain management.\n\n### IoT and Sensor Integration\n\nIntegration with IoT devices and sensors enables real-time monitoring of products throughout the supply chain, automatically recording data such as temperature, humidity, and location on the blockchain.\n\n## Industry Standardization Efforts\n\nOngoing efforts to develop industry standards and protocols will facilitate greater interoperability and widespread adoption of blockchain technology across different organizations and supply chain networks.\n\n## Sustainability and Environmental Impact\n\nBlockchain technology is increasingly being used to track and verify sustainability claims, enabling consumers and organizations to make more informed decisions about environmental impact and ethical sourcing.\n\n## Regulatory Evolution\n\nAs regulatory frameworks mature, organizations will have greater clarity on compliance requirements, potentially accelerating adoption and reducing implementation risks.\n\n## Conclusion\n\nThe future of blockchain in supply chain management is promising, with continued technological advancement and growing industry acceptance driving innovation and creating new opportunities for organizations to enhance their supply chain operations.")
        ]
        
        with patch.object(generator.kimi_client, 'generate_structured_response') as mock_structured:
            with patch.object(generator.kimi_client, 'generate_text') as mock_text:
                mock_structured.return_value = structure_response
                mock_text.side_effect = content_responses
                
                draft = await generator.generate_initial_draft(topic, requirements)
                
                # Comprehensive verification
                assert draft is not None
                assert hasattr(draft, 'topic')
                assert draft.topic == topic
                assert len(draft.structure.sections) == 5
                assert draft.structure.estimated_length == 4200
                assert draft.structure.complexity_level == ComplexityLevel.EXPERT
                assert draft.structure.domain == ResearchDomain.TECHNOLOGY
                
                # Verify all sections have content
                assert len(draft.content) == 5
                for section in draft.structure.sections:
                    assert section.id in draft.content
                    assert len(draft.content[section.id]) > 100  # Substantial content
                
                # Verify content quality
                intro_content = draft.content["introduction"]
                assert "blockchain technology" in intro_content.lower()
                assert "supply chain" in intro_content.lower()
                
                tech_content = draft.content["technical_architecture"]
                assert "smart contract" in tech_content.lower()
                assert "implementation" in tech_content.lower()
                
                # Verify metadata
                assert draft.metadata.word_count > 0
                assert draft.iteration == 0
                assert 0 <= draft.quality_score <= 1
    
    def test_domain_template_coverage(self):
        """Test that all domains have appropriate templates"""
        generator = KimiK2DraftGenerator()
        
        for domain in ResearchDomain:
            assert domain in generator.domain_templates
            template = generator.domain_templates[domain]
            
            # Verify template structure
            assert len(template) > 0
            total_weight = sum(section["weight"] for section in template)
            assert abs(total_weight - 1.0) < 0.01  # Should sum to approximately 1.0
            
            # Verify required fields
            for section in template:
                assert "id" in section
                assert "title" in section
                assert "weight" in section
                assert isinstance(section["weight"], (int, float))
                assert 0 < section["weight"] <= 1
    
    def test_complexity_multiplier_coverage(self):
        """Test that all complexity levels have multipliers"""
        generator = KimiK2DraftGenerator()
        
        for complexity in ComplexityLevel:
            assert complexity in generator.complexity_multipliers
            multiplier = generator.complexity_multipliers[complexity]
            assert isinstance(multiplier, (int, float))
            assert multiplier > 0
        
        # Verify ordering (expert should be highest)
        assert (generator.complexity_multipliers[ComplexityLevel.EXPERT] > 
                generator.complexity_multipliers[ComplexityLevel.BASIC])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])