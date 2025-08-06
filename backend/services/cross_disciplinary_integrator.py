"""
Cross-disciplinary research integration system for TTD-DR framework.
This module provides multi-domain knowledge integration, conflict resolution,
and specialized output formatting for cross-disciplinary research.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from models.core import (
    ResearchDomain, Draft, QualityMetrics, InformationGap, 
    RetrievedInfo, ResearchRequirements, ComplexityLevel
)
from models.research_structure import EnhancedResearchStructure
from services.domain_adapter import DomainAdapter, DomainDetectionResult
from services.kimi_k2_client import KimiK2Client

logger = logging.getLogger(__name__)


class DisciplinaryRelationship(Enum):
    """Types of relationships between disciplines"""
    COMPLEMENTARY = "complementary"  # Disciplines support each other
    OVERLAPPING = "overlapping"      # Disciplines share common ground
    CONFLICTING = "conflicting"      # Disciplines have contradictory views
    HIERARCHICAL = "hierarchical"    # One discipline builds on another
    PARALLEL = "parallel"            # Disciplines address similar questions differently


class CrossDisciplinaryConflict(BaseModel):
    """Represents a conflict between different disciplinary perspectives"""
    conflict_id: str
    domains_involved: List[ResearchDomain]
    conflict_type: str  # "methodological", "theoretical", "empirical", "terminological"
    description: str
    conflicting_information: List[RetrievedInfo]
    severity: float = Field(ge=0.0, le=1.0)  # 0 = minor, 1 = major conflict
    resolution_strategy: Optional[str] = None
    resolved: bool = False


class DisciplinaryPerspective(BaseModel):
    """Represents a single disciplinary perspective on a topic"""
    domain: ResearchDomain
    confidence: float = Field(ge=0.0, le=1.0)
    key_concepts: List[str] = []
    methodological_approach: str = ""
    theoretical_framework: str = ""
    evidence_types: List[str] = []
    terminology: Dict[str, str] = {}
    sources: List[RetrievedInfo] = []


class CrossDisciplinaryIntegration(BaseModel):
    """Result of cross-disciplinary integration process"""
    primary_domains: List[ResearchDomain]
    disciplinary_perspectives: List[DisciplinaryPerspective]
    integration_strategy: str
    conflicts_identified: List[CrossDisciplinaryConflict]
    conflicts_resolved: List[CrossDisciplinaryConflict]
    synthesis_approach: str
    coherence_score: float = Field(ge=0.0, le=1.0)
    integration_metadata: Dict[str, Any] = {}


class CrossDisciplinaryIntegrator:
    """Main class for cross-disciplinary research integration"""
    
    def __init__(self, kimi_client: Optional[KimiK2Client] = None):
        self.kimi_client = kimi_client or KimiK2Client()
        self.domain_adapter = DomainAdapter(kimi_client)
        self.disciplinary_relationships = self._initialize_disciplinary_relationships()
        self.integration_strategies = self._initialize_integration_strategies()
        self.conflict_resolution_methods = self._initialize_conflict_resolution_methods()
    
    def detect_cross_disciplinary_nature(
        self, 
        topic: str, 
        retrieved_info: List[RetrievedInfo]
    ) -> Tuple[bool, List[ResearchDomain]]:
        """
        Detect if research topic requires cross-disciplinary approach
        
        Args:
            topic: Research topic
            retrieved_info: Information retrieved from various sources
            
        Returns:
            Tuple of (is_cross_disciplinary, involved_domains)
        """
        try:
            # Analyze topic for multiple domain indicators
            topic_domains = self._analyze_topic_domains(topic)
            
            # Analyze retrieved information for domain diversity
            source_domains = self._analyze_source_domains(retrieved_info)
            
            # Combine analysis results
            all_domains = set(topic_domains + source_domains)
            
            # Consider cross-disciplinary if multiple domains detected
            is_cross_disciplinary = len(all_domains) > 1
            
            logger.info(f"Cross-disciplinary analysis: {is_cross_disciplinary}, domains: {[d.value for d in all_domains]}")
            
            return is_cross_disciplinary, list(all_domains)
            
        except Exception as e:
            logger.error(f"Error detecting cross-disciplinary nature: {e}")
            return False, [ResearchDomain.GENERAL]
    
    def integrate_multi_domain_knowledge(
        self,
        topic: str,
        domains: List[ResearchDomain],
        retrieved_info: List[RetrievedInfo],
        current_draft: Optional[Draft] = None
    ) -> CrossDisciplinaryIntegration:
        """
        Integrate knowledge from multiple domains
        
        Args:
            topic: Research topic
            domains: List of involved research domains
            retrieved_info: Information from various sources
            current_draft: Current draft if available
            
        Returns:
            CrossDisciplinaryIntegration result
        """
        try:
            logger.info(f"Starting multi-domain knowledge integration for domains: {[d.value for d in domains]}")
            
            # Analyze each disciplinary perspective
            disciplinary_perspectives = []
            for domain in domains:
                perspective = self._analyze_disciplinary_perspective(
                    topic, domain, retrieved_info
                )
                disciplinary_perspectives.append(perspective)
            
            # Identify conflicts between perspectives
            conflicts = self._identify_cross_disciplinary_conflicts(
                disciplinary_perspectives, retrieved_info
            )
            
            # Resolve conflicts using appropriate strategies
            resolved_conflicts = []
            for conflict in conflicts:
                resolution = self._resolve_disciplinary_conflict(conflict, disciplinary_perspectives)
                if resolution:
                    resolved_conflicts.append(resolution)
            
            # Determine integration strategy
            integration_strategy = self._determine_integration_strategy(
                domains, disciplinary_perspectives, conflicts
            )
            
            # Calculate coherence score
            coherence_score = self._calculate_integration_coherence(
                disciplinary_perspectives, resolved_conflicts
            )
            
            # Create integration result
            integration = CrossDisciplinaryIntegration(
                primary_domains=domains,
                disciplinary_perspectives=disciplinary_perspectives,
                integration_strategy=integration_strategy,
                conflicts_identified=conflicts,
                conflicts_resolved=resolved_conflicts,
                synthesis_approach=self._determine_synthesis_approach(domains),
                coherence_score=coherence_score,
                integration_metadata={
                    "topic": topic,
                    "integration_timestamp": datetime.now().isoformat(),
                    "domains_count": len(domains),
                    "conflicts_count": len(conflicts),
                    "resolved_conflicts_count": len(resolved_conflicts)
                }
            )
            
            logger.info(f"Multi-domain integration completed. Coherence score: {coherence_score:.2f}")
            
            return integration
            
        except Exception as e:
            logger.error(f"Error in multi-domain knowledge integration: {e}")
            # Return fallback integration
            return self._create_fallback_integration(topic, domains)
    
    def resolve_cross_disciplinary_conflicts(
        self,
        conflicts: List[CrossDisciplinaryConflict],
        disciplinary_perspectives: List[DisciplinaryPerspective]
    ) -> List[CrossDisciplinaryConflict]:
        """
        Resolve conflicts between different disciplinary perspectives
        
        Args:
            conflicts: List of identified conflicts
            disciplinary_perspectives: Perspectives from different domains
            
        Returns:
            List of resolved conflicts
        """
        resolved_conflicts = []
        
        for conflict in conflicts:
            try:
                logger.info(f"Resolving conflict: {conflict.conflict_type} between {[d.value for d in conflict.domains_involved]}")
                
                # Select appropriate resolution method
                resolution_method = self._select_resolution_method(conflict)
                
                # Apply resolution strategy
                resolved_conflict = self._apply_resolution_strategy(
                    conflict, disciplinary_perspectives, resolution_method
                )
                
                if resolved_conflict:
                    resolved_conflicts.append(resolved_conflict)
                    logger.info(f"Conflict resolved using {resolution_method} strategy")
                
            except Exception as e:
                logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
                continue
        
        return resolved_conflicts
    
    def format_cross_disciplinary_output(
        self,
        draft: Draft,
        integration: CrossDisciplinaryIntegration,
        output_format: str = "comprehensive"
    ) -> str:
        """
        Format output for cross-disciplinary research
        
        Args:
            draft: Current research draft
            integration: Cross-disciplinary integration result
            output_format: Desired output format
            
        Returns:
            Formatted cross-disciplinary research output
        """
        try:
            logger.info(f"Formatting cross-disciplinary output in {output_format} format")
            
            if output_format == "comprehensive":
                return self._format_comprehensive_output(draft, integration)
            elif output_format == "comparative":
                return self._format_comparative_output(draft, integration)
            elif output_format == "synthesis":
                return self._format_synthesis_output(draft, integration)
            elif output_format == "domain_specific":
                return self._format_domain_specific_output(draft, integration)
            else:
                return self._format_comprehensive_output(draft, integration)
                
        except Exception as e:
            logger.error(f"Error formatting cross-disciplinary output: {e}")
            return self._format_fallback_output(draft, integration)
    
    # Private helper methods
    
    def _analyze_topic_domains(self, topic: str) -> List[ResearchDomain]:
        """Analyze topic to identify potential domains"""
        try:
            # Use Kimi K2 for sophisticated domain analysis
            prompt = f"""
            Analyze this research topic and identify all relevant research domains:
            
            Topic: {topic}
            
            Consider these domains: TECHNOLOGY, SCIENCE, BUSINESS, ACADEMIC, GENERAL
            
            Return a JSON list of domains that are relevant to this topic, ordered by relevance.
            Include reasoning for each domain selection.
            
            Format: {{"domains": ["{{"domain": "DOMAIN_NAME", "relevance": 0.8, "reasoning": "explanation"}}"], "is_cross_disciplinary": true/false}}
            """
            
            response = self._generate_content(prompt, temperature=0.3)
            analysis = json.loads(response)
            
            domains = []
            for domain_info in analysis.get("domains", []):
                if domain_info.get("relevance", 0) > 0.3:  # Threshold for relevance
                    try:
                        domain = ResearchDomain(domain_info["domain"])
                        domains.append(domain)
                    except ValueError:
                        continue
            
            return domains[:3]  # Limit to top 3 domains
            
        except Exception as e:
            logger.error(f"Error analyzing topic domains: {e}")
            # Fallback to basic keyword analysis
            return self._fallback_topic_analysis(topic)
    
    def _analyze_source_domains(self, retrieved_info: List[RetrievedInfo]) -> List[ResearchDomain]:
        """Analyze retrieved information to identify source domains"""
        domain_counts = {}
        
        for info in retrieved_info:
            # Use domain adapter to detect domain of each source
            try:
                domain_result = self.domain_adapter.detect_domain(
                    topic="", content=info.content[:500]  # Use first 500 chars
                )
                domain = domain_result.primary_domain
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            except Exception:
                continue
        
        # Return domains sorted by frequency
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, count in sorted_domains if count > 0]
    
    def _analyze_disciplinary_perspective(
        self,
        topic: str,
        domain: ResearchDomain,
        retrieved_info: List[RetrievedInfo]
    ) -> DisciplinaryPerspective:
        """Analyze a single disciplinary perspective"""
        try:
            # Filter information relevant to this domain
            domain_info = [
                info for info in retrieved_info
                if self._is_info_relevant_to_domain(info, domain)
            ]
            
            # Use Kimi K2 to analyze disciplinary perspective
            prompt = f"""
            Analyze the {domain.value} perspective on this research topic:
            
            Topic: {topic}
            Domain: {domain.value}
            
            Relevant Information:
            {json.dumps([{"content": info.content[:300], "source": str(info.source)} for info in domain_info[:5]], indent=2)}
            
            Provide analysis in JSON format:
            {{
                "key_concepts": ["concept1", "concept2"],
                "methodological_approach": "description of typical methods",
                "theoretical_framework": "main theoretical approaches",
                "evidence_types": ["type1", "type2"],
                "terminology": {{"term1": "definition1"}},
                "confidence": 0.8
            }}
            """
            
            response = self._generate_content(prompt, temperature=0.4)
            analysis = json.loads(response)
            
            return DisciplinaryPerspective(
                domain=domain,
                confidence=analysis.get("confidence", 0.5),
                key_concepts=analysis.get("key_concepts", []),
                methodological_approach=analysis.get("methodological_approach", ""),
                theoretical_framework=analysis.get("theoretical_framework", ""),
                evidence_types=analysis.get("evidence_types", []),
                terminology=analysis.get("terminology", {}),
                sources=domain_info
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {domain.value} perspective: {e}")
            return self._create_fallback_perspective(domain, retrieved_info)
    
    def _identify_cross_disciplinary_conflicts(
        self,
        perspectives: List[DisciplinaryPerspective],
        retrieved_info: List[RetrievedInfo]
    ) -> List[CrossDisciplinaryConflict]:
        """Identify conflicts between disciplinary perspectives"""
        conflicts = []
        
        # Compare perspectives pairwise
        for i, perspective1 in enumerate(perspectives):
            for j, perspective2 in enumerate(perspectives[i+1:], i+1):
                conflict = self._detect_perspective_conflict(
                    perspective1, perspective2, retrieved_info
                )
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_perspective_conflict(
        self,
        perspective1: DisciplinaryPerspective,
        perspective2: DisciplinaryPerspective,
        retrieved_info: List[RetrievedInfo]
    ) -> Optional[CrossDisciplinaryConflict]:
        """Detect conflict between two perspectives"""
        try:
            # Use Kimi K2 to detect conflicts
            prompt = f"""
            Analyze potential conflicts between these two disciplinary perspectives:
            
            {perspective1.domain.value} Perspective:
            - Key concepts: {perspective1.key_concepts}
            - Methodology: {perspective1.methodological_approach}
            - Framework: {perspective1.theoretical_framework}
            
            {perspective2.domain.value} Perspective:
            - Key concepts: {perspective2.key_concepts}
            - Methodology: {perspective2.methodological_approach}
            - Framework: {perspective2.theoretical_framework}
            
            Identify any conflicts in:
            1. Methodological approaches
            2. Theoretical frameworks
            3. Terminology usage
            4. Evidence interpretation
            
            Return JSON:
            {{
                "has_conflict": true/false,
                "conflict_type": "methodological/theoretical/empirical/terminological",
                "description": "detailed description",
                "severity": 0.7,
                "conflicting_aspects": ["aspect1", "aspect2"]
            }}
            """
            
            response = self._generate_content(prompt, temperature=0.3)
            analysis = json.loads(response)
            
            if analysis.get("has_conflict", False):
                return CrossDisciplinaryConflict(
                    conflict_id=f"conflict_{perspective1.domain.value}_{perspective2.domain.value}_{datetime.now().strftime('%H%M%S')}",
                    domains_involved=[perspective1.domain, perspective2.domain],
                    conflict_type=analysis.get("conflict_type", "general"),
                    description=analysis.get("description", ""),
                    conflicting_information=perspective1.sources + perspective2.sources,
                    severity=analysis.get("severity", 0.5)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting perspective conflict: {e}")
            return None
    
    def _resolve_disciplinary_conflict(
        self,
        conflict: CrossDisciplinaryConflict,
        perspectives: List[DisciplinaryPerspective]
    ) -> Optional[CrossDisciplinaryConflict]:
        """Resolve a specific disciplinary conflict"""
        try:
            resolution_method = self._select_resolution_method(conflict)
            
            # Use Kimi K2 to generate resolution
            prompt = f"""
            Resolve this cross-disciplinary conflict using {resolution_method} approach:
            
            Conflict: {conflict.description}
            Type: {conflict.conflict_type}
            Domains: {[d.value for d in conflict.domains_involved]}
            Severity: {conflict.severity}
            
            Resolution Method: {resolution_method}
            
            Provide a resolution strategy that:
            1. Acknowledges both perspectives
            2. Finds common ground where possible
            3. Explains differences constructively
            4. Suggests integrated approach
            
            Return JSON:
            {{
                "resolution_strategy": "detailed strategy",
                "integrated_approach": "how to combine perspectives",
                "remaining_tensions": ["any unresolved issues"],
                "confidence": 0.8
            }}
            """
            
            response = self._generate_content(prompt, temperature=0.4)
            resolution = json.loads(response)
            
            # Create resolved conflict
            resolved_conflict = conflict.copy(deep=True)
            resolved_conflict.resolution_strategy = resolution.get("resolution_strategy", "")
            resolved_conflict.resolved = True
            
            return resolved_conflict
            
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
            return None
    
    def _generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using Kimi K2 with fallback"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_generate_content(prompt, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._async_generate_content(prompt, **kwargs))
        except Exception as e:
            logger.warning(f"Kimi K2 generation failed, using fallback: {e}")
            return self._fallback_content_generation(prompt, **kwargs)
    
    async def _async_generate_content(self, prompt: str, **kwargs) -> str:
        """Async content generation using Kimi K2"""
        response = await self.kimi_client.generate_text(prompt, **kwargs)
        return response.content
    
    def _fallback_content_generation(self, prompt: str, **kwargs) -> str:
        """Fallback content generation when Kimi K2 is not available"""
        if "domains" in prompt.lower():
            return json.dumps({
                "domains": [
                    {"domain": "TECHNOLOGY", "relevance": 0.6, "reasoning": "Technical aspects present"},
                    {"domain": "SCIENCE", "relevance": 0.4, "reasoning": "Research methodology"}
                ],
                "is_cross_disciplinary": True
            })
        elif "conflict" in prompt.lower():
            return json.dumps({
                "has_conflict": False,
                "conflict_type": "none",
                "description": "No significant conflicts detected",
                "severity": 0.1
            })
        elif "resolution" in prompt.lower():
            return json.dumps({
                "resolution_strategy": "Integrate perspectives through balanced approach",
                "integrated_approach": "Combine methodologies where appropriate",
                "remaining_tensions": [],
                "confidence": 0.7
            })
        else:
            return json.dumps({
                "key_concepts": ["interdisciplinary", "integration"],
                "methodological_approach": "Mixed methods",
                "theoretical_framework": "Interdisciplinary framework",
                "evidence_types": ["multiple"],
                "terminology": {},
                "confidence": 0.5
            })
    
    # Additional helper methods for initialization and formatting
    
    def _initialize_disciplinary_relationships(self) -> Dict[Tuple[ResearchDomain, ResearchDomain], DisciplinaryRelationship]:
        """Initialize relationships between disciplines"""
        return {
            (ResearchDomain.TECHNOLOGY, ResearchDomain.SCIENCE): DisciplinaryRelationship.COMPLEMENTARY,
            (ResearchDomain.TECHNOLOGY, ResearchDomain.BUSINESS): DisciplinaryRelationship.OVERLAPPING,
            (ResearchDomain.SCIENCE, ResearchDomain.ACADEMIC): DisciplinaryRelationship.HIERARCHICAL,
            (ResearchDomain.BUSINESS, ResearchDomain.ACADEMIC): DisciplinaryRelationship.PARALLEL,
            # Add more relationships as needed
        }
    
    def _initialize_integration_strategies(self) -> Dict[str, str]:
        """Initialize integration strategies"""
        return {
            "synthesis": "Combine perspectives into unified framework",
            "comparative": "Present multiple perspectives side by side",
            "hierarchical": "Organize perspectives by level of analysis",
            "dialectical": "Explore tensions and contradictions",
            "pragmatic": "Focus on practical applications"
        }
    
    def _initialize_conflict_resolution_methods(self) -> Dict[str, str]:
        """Initialize conflict resolution methods"""
        return {
            "methodological": "triangulation",
            "theoretical": "synthesis",
            "empirical": "evidence_weighting",
            "terminological": "clarification",
            "general": "integration"
        }
    
    # Formatting methods
    
    def _format_comprehensive_output(self, draft: Draft, integration: CrossDisciplinaryIntegration) -> str:
        """Format comprehensive cross-disciplinary output"""
        sections = []
        
        # Executive summary
        sections.append("# Cross-Disciplinary Research Report\n")
        sections.append(f"## Executive Summary\n")
        sections.append(f"This report integrates perspectives from {len(integration.primary_domains)} disciplines: {', '.join([d.value for d in integration.primary_domains])}.\n")
        
        # Disciplinary perspectives
        sections.append("## Disciplinary Perspectives\n")
        for perspective in integration.disciplinary_perspectives:
            sections.append(f"### {perspective.domain.value} Perspective\n")
            sections.append(f"**Key Concepts:** {', '.join(perspective.key_concepts)}\n")
            sections.append(f"**Methodological Approach:** {perspective.methodological_approach}\n")
            sections.append(f"**Theoretical Framework:** {perspective.theoretical_framework}\n\n")
        
        # Conflicts and resolutions
        if integration.conflicts_resolved:
            sections.append("## Cross-Disciplinary Integration\n")
            sections.append("### Resolved Conflicts\n")
            for conflict in integration.conflicts_resolved:
                sections.append(f"**{conflict.conflict_type.title()} Conflict:** {conflict.description}\n")
                sections.append(f"**Resolution:** {conflict.resolution_strategy}\n\n")
        
        # Main content
        sections.append("## Integrated Analysis\n")
        for section_id, content in draft.content.items():
            if content:
                sections.append(f"### {section_id.replace('_', ' ').title()}\n")
                sections.append(f"{content}\n\n")
        
        return "\n".join(sections)
    
    def _format_comparative_output(self, draft: Draft, integration: CrossDisciplinaryIntegration) -> str:
        """Format comparative cross-disciplinary output"""
        # Implementation for comparative format
        return self._format_comprehensive_output(draft, integration)  # Simplified for now
    
    def _format_synthesis_output(self, draft: Draft, integration: CrossDisciplinaryIntegration) -> str:
        """Format synthesis cross-disciplinary output"""
        # Implementation for synthesis format
        return self._format_comprehensive_output(draft, integration)  # Simplified for now
    
    def _format_domain_specific_output(self, draft: Draft, integration: CrossDisciplinaryIntegration) -> str:
        """Format domain-specific cross-disciplinary output"""
        # Implementation for domain-specific format
        return self._format_comprehensive_output(draft, integration)  # Simplified for now
    
    def _format_fallback_output(self, draft: Draft, integration: CrossDisciplinaryIntegration) -> str:
        """Fallback formatting when other methods fail"""
        return f"# Cross-Disciplinary Research Report\n\nTopic: {draft.topic}\n\nDomains: {', '.join([d.value for d in integration.primary_domains])}\n\n{json.dumps(draft.content, indent=2)}"
    
    # Additional helper methods
    
    def _fallback_topic_analysis(self, topic: str) -> List[ResearchDomain]:
        """Fallback topic analysis using keyword matching"""
        keywords = topic.lower().split()
        domain_scores = {domain: 0 for domain in ResearchDomain}
        
        # Simple keyword matching
        tech_keywords = ["technology", "software", "ai", "digital", "computer"]
        science_keywords = ["research", "study", "analysis", "experiment", "data"]
        business_keywords = ["market", "business", "economic", "financial", "strategy"]
        
        for keyword in keywords:
            if any(tk in keyword for tk in tech_keywords):
                domain_scores[ResearchDomain.TECHNOLOGY] += 1
            if any(sk in keyword for sk in science_keywords):
                domain_scores[ResearchDomain.SCIENCE] += 1
            if any(bk in keyword for bk in business_keywords):
                domain_scores[ResearchDomain.BUSINESS] += 1
        
        # Return domains with scores > 0
        return [domain for domain, score in domain_scores.items() if score > 0] or [ResearchDomain.GENERAL]
    
    def _is_info_relevant_to_domain(self, info: RetrievedInfo, domain: ResearchDomain) -> bool:
        """Check if information is relevant to a specific domain"""
        try:
            domain_result = self.domain_adapter.detect_domain(topic="", content=info.content[:200])
            return domain_result.primary_domain == domain or domain in [d.domain for d in domain_result.secondary_domains]
        except Exception:
            return False
    
    def _create_fallback_perspective(self, domain: ResearchDomain, retrieved_info: List[RetrievedInfo]) -> DisciplinaryPerspective:
        """Create fallback perspective when analysis fails"""
        return DisciplinaryPerspective(
            domain=domain,
            confidence=0.5,
            key_concepts=[f"{domain.value.lower()}_concept"],
            methodological_approach=f"{domain.value} methodology",
            theoretical_framework=f"{domain.value} framework",
            evidence_types=["general"],
            terminology={},
            sources=retrieved_info[:3]  # Limit to first 3 sources
        )
    
    def _create_fallback_integration(self, topic: str, domains: List[ResearchDomain]) -> CrossDisciplinaryIntegration:
        """Create fallback integration when main process fails"""
        return CrossDisciplinaryIntegration(
            primary_domains=domains,
            disciplinary_perspectives=[self._create_fallback_perspective(d, []) for d in domains],
            integration_strategy="basic_synthesis",
            conflicts_identified=[],
            conflicts_resolved=[],
            synthesis_approach="simple_combination",
            coherence_score=0.5,
            integration_metadata={"fallback": True, "topic": topic}
        )
    
    def _select_resolution_method(self, conflict: CrossDisciplinaryConflict) -> str:
        """Select appropriate resolution method for conflict"""
        return self.conflict_resolution_methods.get(conflict.conflict_type, "integration")
    
    def _apply_resolution_strategy(
        self,
        conflict: CrossDisciplinaryConflict,
        perspectives: List[DisciplinaryPerspective],
        method: str
    ) -> Optional[CrossDisciplinaryConflict]:
        """Apply resolution strategy to conflict"""
        # Simplified implementation - would be more sophisticated in practice
        resolved_conflict = conflict.copy(deep=True)
        resolved_conflict.resolution_strategy = f"Applied {method} resolution method"
        resolved_conflict.resolved = True
        return resolved_conflict
    
    def _determine_integration_strategy(
        self,
        domains: List[ResearchDomain],
        perspectives: List[DisciplinaryPerspective],
        conflicts: List[CrossDisciplinaryConflict]
    ) -> str:
        """Determine best integration strategy"""
        if len(conflicts) > 2:
            return "dialectical"  # Handle many conflicts
        elif len(domains) > 3:
            return "hierarchical"  # Organize many domains
        else:
            return "synthesis"  # Default synthesis approach
    
    def _determine_synthesis_approach(self, domains: List[ResearchDomain]) -> str:
        """Determine synthesis approach based on domains"""
        if ResearchDomain.SCIENCE in domains and ResearchDomain.TECHNOLOGY in domains:
            return "evidence_based_synthesis"
        elif ResearchDomain.BUSINESS in domains:
            return "practical_synthesis"
        else:
            return "theoretical_synthesis"
    
    def _calculate_integration_coherence(
        self,
        perspectives: List[DisciplinaryPerspective],
        resolved_conflicts: List[CrossDisciplinaryConflict]
    ) -> float:
        """Calculate coherence score for integration"""
        base_score = sum(p.confidence for p in perspectives) / len(perspectives) if perspectives else 0.5
        conflict_penalty = len(resolved_conflicts) * 0.1
        return max(0.0, min(1.0, base_score - conflict_penalty))
    
    def _format_disciplinary_perspectives(self, perspectives: List[DisciplinaryPerspective]) -> str:
        """Format disciplinary perspectives for output"""
        sections = []
        sections.append("# Disciplinary Perspectives\n")
        
        for perspective in perspectives:
            sections.append(f"## {perspective.domain.value} Perspective\n")
            sections.append(f"**Confidence:** {perspective.confidence:.2f}\n")
            sections.append(f"**Key Concepts:** {', '.join(perspective.key_concepts)}\n")
            sections.append(f"**Methodological Approach:** {perspective.methodological_approach}\n")
            sections.append(f"**Theoretical Framework:** {perspective.theoretical_framework}\n")
            sections.append(f"**Evidence Types:** {', '.join(perspective.evidence_types)}\n")
            
            if perspective.terminology:
                sections.append("**Key Terminology:**\n")
                for term, definition in perspective.terminology.items():
                    sections.append(f"- **{term}:** {definition}\n")
            
            sections.append(f"**Sources:** {len(perspective.sources)} sources analyzed\n\n")
        
        return "\n".join(sections)
    
    def _format_conflict_resolutions(self, resolved_conflicts: List[CrossDisciplinaryConflict]) -> str:
        """Format conflict resolutions for output"""
        sections = []
        sections.append("# Cross-Disciplinary Conflict Resolutions\n")
        
        for conflict in resolved_conflicts:
            sections.append(f"## {conflict.conflict_type.title()} Conflict\n")
            sections.append(f"**Domains Involved:** {', '.join([d.value for d in conflict.domains_involved])}\n")
            sections.append(f"**Description:** {conflict.description}\n")
            sections.append(f"**Severity:** {conflict.severity:.2f}\n")
            sections.append(f"**Resolution Strategy:** {conflict.resolution_strategy}\n\n")
        
        return "\n".join(sections)
    
    def _assess_disciplinary_balance(self, integration: CrossDisciplinaryIntegration) -> float:
        """Assess balance between disciplinary perspectives"""
        if not integration.disciplinary_perspectives:
            return 0.5
        
        # Calculate balance based on confidence scores
        confidences = [p.confidence for p in integration.disciplinary_perspectives]
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Lower variance indicates better balance
        balance_score = max(0.0, 1.0 - variance)
        return balance_score
    
    def _assess_conflict_resolution(self, integration: CrossDisciplinaryIntegration) -> float:
        """Assess effectiveness of conflict resolution"""
        total_conflicts = len(integration.conflicts_identified)
        if total_conflicts == 0:
            return 1.0  # No conflicts to resolve
        
        resolved_conflicts = len(integration.conflicts_resolved)
        resolution_rate = resolved_conflicts / total_conflicts
        
        # Weight by conflict severity
        if integration.conflicts_resolved:
            avg_severity = sum(c.severity for c in integration.conflicts_resolved) / len(integration.conflicts_resolved)
            severity_bonus = 1.0 - avg_severity  # Lower severity = better resolution
            return min(1.0, resolution_rate + severity_bonus * 0.2)
        
        return resolution_rate
    
    def _assess_synthesis_quality(self, integration: CrossDisciplinaryIntegration) -> float:
        """Assess quality of cross-domain synthesis"""
        # Base score from coherence
        base_score = integration.coherence_score
        
        # Bonus for multiple perspectives
        perspective_bonus = min(0.2, len(integration.disciplinary_perspectives) * 0.05)
        
        # Penalty for unresolved conflicts
        unresolved_conflicts = len(integration.conflicts_identified) - len(integration.conflicts_resolved)
        conflict_penalty = unresolved_conflicts * 0.1
        
        return max(0.0, min(1.0, base_score + perspective_bonus - conflict_penalty))
    
    def _assess_methodological_integration(self, integration: CrossDisciplinaryIntegration) -> float:
        """Assess integration of different methodological approaches"""
        if not integration.disciplinary_perspectives:
            return 0.5
        
        # Count unique methodological approaches
        methodologies = set()
        for perspective in integration.disciplinary_perspectives:
            if perspective.methodological_approach:
                methodologies.add(perspective.methodological_approach.lower())
        
        # Score based on methodology diversity and integration strategy
        diversity_score = min(1.0, len(methodologies) / len(integration.disciplinary_perspectives))
        
        # Bonus for appropriate integration strategy
        strategy_bonus = 0.2 if integration.integration_strategy in ["synthesis", "triangulation"] else 0.0
        
        return min(1.0, diversity_score + strategy_bonus)


# Export main class
__all__ = [
    "CrossDisciplinaryIntegrator",
    "CrossDisciplinaryIntegration", 
    "CrossDisciplinaryConflict",
    "DisciplinaryPerspective",
    "DisciplinaryRelationship"
]