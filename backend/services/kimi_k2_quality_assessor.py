"""
Kimi K2-powered quality assessment service for TTD-DR framework.
Provides comprehensive evaluation metrics for research drafts.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from models.core import (
    Draft, QualityMetrics, TTDRState, ResearchRequirements,
    ComplexityLevel, ResearchDomain
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class KimiK2QualityAssessor:
    """Kimi K2-powered quality assessment for research drafts"""
    
    def __init__(self):
        """Initialize the quality assessor with Kimi K2 client"""
        self.kimi_client = KimiK2Client()
        
        # Quality assessment criteria weights
        self.criteria_weights = {
            "completeness": 0.3,
            "coherence": 0.25,
            "accuracy": 0.25,
            "citation_quality": 0.2
        }
        
        # Domain-specific quality thresholds
        self.domain_thresholds = {
            ResearchDomain.ACADEMIC: 0.85,
            ResearchDomain.SCIENCE: 0.8,
            ResearchDomain.TECHNOLOGY: 0.75,
            ResearchDomain.BUSINESS: 0.7,
            ResearchDomain.GENERAL: 0.65
        }
    
    async def evaluate_draft(self, draft: Draft, requirements: Optional[ResearchRequirements] = None) -> QualityMetrics:
        """
        Evaluate draft quality using Kimi K2 comprehensive analysis
        
        Args:
            draft: Research draft to evaluate
            requirements: Optional research requirements for context
            
        Returns:
            QualityMetrics with detailed assessment scores
        """
        logger.info(f"Starting quality assessment for draft: {draft.id}")
        
        try:
            # Prepare draft content for analysis
            full_content = self._prepare_draft_content(draft)
            
            # Run parallel assessments for different quality dimensions
            assessment_tasks = [
                self._assess_completeness(draft, full_content, requirements),
                self._assess_coherence(draft, full_content),
                self._assess_accuracy(draft, full_content),
                self._assess_citation_quality(draft, full_content)
            ]
            
            # Execute assessments concurrently
            completeness, coherence, accuracy, citation_quality = await asyncio.gather(
                *assessment_tasks, return_exceptions=True
            )
            
            # Handle any exceptions from parallel execution
            completeness = self._handle_assessment_result(completeness, "completeness", 0.5)
            coherence = self._handle_assessment_result(coherence, "coherence", 0.5)
            accuracy = self._handle_assessment_result(accuracy, "accuracy", 0.5)
            citation_quality = self._handle_assessment_result(citation_quality, "citation_quality", 0.5)
            
            # Calculate overall score using weighted average
            overall_score = (
                completeness * self.criteria_weights["completeness"] +
                coherence * self.criteria_weights["coherence"] +
                accuracy * self.criteria_weights["accuracy"] +
                citation_quality * self.criteria_weights["citation_quality"]
            )
            
            quality_metrics = QualityMetrics(
                completeness=completeness,
                coherence=coherence,
                accuracy=accuracy,
                citation_quality=citation_quality,
                overall_score=overall_score
            )
            
            logger.info(f"Quality assessment completed. Overall score: {overall_score:.3f}")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return fallback metrics
            return QualityMetrics(
                completeness=0.3,
                coherence=0.3,
                accuracy=0.3,
                citation_quality=0.3,
                overall_score=0.3
            )
    
    async def _assess_completeness(self, draft: Draft, content: str, requirements: Optional[ResearchRequirements]) -> float:
        """Assess content completeness using Kimi K2"""
        prompt = self._build_completeness_prompt(draft, content, requirements)
        
        try:
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "completeness_score": "float between 0.0 and 1.0",
                    "missing_elements": ["list of missing content areas"],
                    "coverage_analysis": "detailed analysis of topic coverage",
                    "recommendations": ["list of improvement suggestions"]
                }
            )
            
            return max(0.0, min(1.0, float(response.get("completeness_score", 0.5))))
            
        except Exception as e:
            logger.error(f"Completeness assessment failed: {e}")
            # Fallback to simple heuristic
            return self._fallback_completeness_assessment(draft, content)
    
    async def _assess_coherence(self, draft: Draft, content: str) -> float:
        """Assess logical coherence and flow using Kimi K2"""
        prompt = self._build_coherence_prompt(draft, content)
        
        try:
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "coherence_score": "float between 0.0 and 1.0",
                    "flow_analysis": "analysis of logical flow between sections",
                    "consistency_check": "assessment of argument consistency",
                    "transition_quality": "evaluation of section transitions"
                }
            )
            
            return max(0.0, min(1.0, float(response.get("coherence_score", 0.5))))
            
        except Exception as e:
            logger.error(f"Coherence assessment failed: {e}")
            return self._fallback_coherence_assessment(draft, content)
    
    async def _assess_accuracy(self, draft: Draft, content: str) -> float:
        """Assess factual accuracy and reliability using Kimi K2"""
        prompt = self._build_accuracy_prompt(draft, content)
        
        try:
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "accuracy_score": "float between 0.0 and 1.0",
                    "fact_verification": "analysis of factual claims",
                    "source_reliability": "assessment of information sources",
                    "potential_errors": ["list of potential factual issues"]
                }
            )
            
            return max(0.0, min(1.0, float(response.get("accuracy_score", 0.5))))
            
        except Exception as e:
            logger.error(f"Accuracy assessment failed: {e}")
            return self._fallback_accuracy_assessment(draft, content)
    
    async def _assess_citation_quality(self, draft: Draft, content: str) -> float:
        """Assess citation quality and source attribution using Kimi K2"""
        prompt = self._build_citation_prompt(draft, content)
        
        try:
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "citation_score": "float between 0.0 and 1.0",
                    "citation_coverage": "analysis of citation completeness",
                    "source_diversity": "assessment of source variety",
                    "attribution_quality": "evaluation of proper attribution"
                }
            )
            
            return max(0.0, min(1.0, float(response.get("citation_score", 0.5))))
            
        except Exception as e:
            logger.error(f"Citation assessment failed: {e}")
            return self._fallback_citation_assessment(draft, content)
    
    def _prepare_draft_content(self, draft: Draft) -> str:
        """Prepare draft content for analysis"""
        content_parts = [f"# {draft.topic}\n"]
        
        for section in draft.structure.sections:
            content_parts.append(f"\n## {section.title}\n")
            if section.id in draft.content:
                content_parts.append(draft.content[section.id])
            else:
                content_parts.append("[Content placeholder - not yet filled]")
        
        return "\n".join(content_parts)
    
    def _build_completeness_prompt(self, draft: Draft, content: str, requirements: Optional[ResearchRequirements]) -> str:
        """Build prompt for completeness assessment"""
        domain_context = f"Research domain: {draft.structure.domain.value}"
        complexity_context = f"Complexity level: {draft.structure.complexity_level.value}"
        
        requirements_context = ""
        if requirements:
            requirements_context = f"""
Research Requirements:
- Quality threshold: {requirements.quality_threshold}
- Max sources: {requirements.max_sources}
- Preferred source types: {', '.join(requirements.preferred_source_types)}
"""
        
        return f"""
As an expert research quality assessor, evaluate the completeness of this research draft.

{domain_context}
{complexity_context}
{requirements_context}

Research Draft:
{content[:4000]}  # Truncate for API limits

Assess the completeness by considering:
1. Topic coverage breadth and depth
2. Essential sections and subsections presence
3. Adequate detail level for the complexity level
4. Missing critical information areas
5. Balance between different aspects of the topic

Provide a completeness score from 0.0 (completely incomplete) to 1.0 (fully complete).
Consider the research domain and complexity level in your assessment.
"""
    
    def _build_coherence_prompt(self, draft: Draft, content: str) -> str:
        """Build prompt for coherence assessment"""
        return f"""
As an expert research quality assessor, evaluate the logical coherence and flow of this research draft.

Research Topic: {draft.topic}
Domain: {draft.structure.domain.value}

Research Draft:
{content[:4000]}  # Truncate for API limits

Assess the coherence by considering:
1. Logical flow between sections and ideas
2. Consistency of arguments throughout
3. Clear transitions between topics
4. Overall narrative structure
5. Absence of contradictions or logical gaps

Provide a coherence score from 0.0 (completely incoherent) to 1.0 (perfectly coherent).
Focus on how well the ideas connect and build upon each other.
"""
    
    def _build_accuracy_prompt(self, draft: Draft, content: str) -> str:
        """Build prompt for accuracy assessment"""
        return f"""
As an expert research quality assessor, evaluate the factual accuracy and reliability of this research draft.

Research Topic: {draft.topic}
Domain: {draft.structure.domain.value}

Research Draft:
{content[:4000]}  # Truncate for API limits

Assess the accuracy by considering:
1. Factual correctness of claims and statements
2. Appropriate use of technical terminology
3. Consistency with established knowledge in the field
4. Potential for misinformation or outdated information
5. Overall reliability of presented information

Provide an accuracy score from 0.0 (highly inaccurate) to 1.0 (highly accurate).
Consider the research domain's standards for accuracy.
"""
    
    def _build_citation_prompt(self, draft: Draft, content: str) -> str:
        """Build prompt for citation quality assessment"""
        return f"""
As an expert research quality assessor, evaluate the citation quality and source attribution of this research draft.

Research Topic: {draft.topic}
Domain: {draft.structure.domain.value}

Research Draft:
{content[:4000]}  # Truncate for API limits

Assess the citation quality by considering:
1. Proper attribution of sources and claims
2. Diversity and credibility of sources
3. Appropriate citation frequency and placement
4. Completeness of bibliographic information
5. Adherence to academic citation standards

Provide a citation score from 0.0 (poor citations) to 1.0 (excellent citations).
Consider the research domain's citation standards and expectations.
"""
    
    def _handle_assessment_result(self, result: Any, metric_name: str, fallback_value: float) -> float:
        """Handle assessment result, including exceptions"""
        if isinstance(result, Exception):
            logger.error(f"{metric_name} assessment failed: {result}")
            return fallback_value
        
        if isinstance(result, (int, float)):
            return max(0.0, min(1.0, float(result)))
        
        return fallback_value
    
    def _fallback_completeness_assessment(self, draft: Draft, content: str) -> float:
        """Fallback completeness assessment using simple heuristics"""
        # Count filled sections vs total sections
        total_sections = len(draft.structure.sections)
        filled_sections = sum(1 for section in draft.structure.sections 
                            if section.id in draft.content and draft.content[section.id].strip())
        
        if total_sections == 0:
            return 0.0
        
        section_completeness = filled_sections / total_sections
        
        # Consider content length
        total_content_length = sum(len(draft.content.get(section.id, "")) 
                                 for section in draft.structure.sections)
        
        # Estimate expected length based on complexity
        expected_length = {
            ComplexityLevel.BASIC: 2000,
            ComplexityLevel.INTERMEDIATE: 4000,
            ComplexityLevel.ADVANCED: 6000,
            ComplexityLevel.EXPERT: 8000
        }.get(draft.structure.complexity_level, 4000)
        
        length_completeness = min(total_content_length / expected_length, 1.0)
        
        return (section_completeness * 0.6 + length_completeness * 0.4)
    
    def _fallback_coherence_assessment(self, draft: Draft, content: str) -> float:
        """Fallback coherence assessment using simple heuristics"""
        # Basic coherence indicators
        coherence_score = 0.5  # Base score
        
        # Check for section transitions
        if "however" in content.lower() or "furthermore" in content.lower() or "therefore" in content.lower():
            coherence_score += 0.1
        
        # Check for consistent terminology
        topic_words = draft.topic.lower().split()
        topic_mentions = sum(content.lower().count(word) for word in topic_words if len(word) > 3)
        if topic_mentions > len(draft.structure.sections):
            coherence_score += 0.1
        
        # Penalize very short sections
        short_sections = sum(1 for section in draft.structure.sections 
                           if len(draft.content.get(section.id, "")) < 100)
        if short_sections > len(draft.structure.sections) / 2:
            coherence_score -= 0.2
        
        return max(0.0, min(1.0, coherence_score))
    
    def _fallback_accuracy_assessment(self, draft: Draft, content: str) -> float:
        """Fallback accuracy assessment using simple heuristics"""
        # Base accuracy score
        accuracy_score = 0.6
        
        # Check for hedging language (indicates uncertainty/accuracy awareness)
        hedging_words = ["may", "might", "could", "appears", "suggests", "indicates", "likely"]
        hedging_count = sum(content.lower().count(word) for word in hedging_words)
        if hedging_count > 0:
            accuracy_score += 0.1
        
        # Check for specific numbers/dates (indicates factual content)
        import re
        numbers = re.findall(r'\b\d{4}\b|\b\d+%\b|\b\d+\.\d+\b', content)
        if len(numbers) > 5:
            accuracy_score += 0.1
        
        # Domain-specific adjustments
        if draft.structure.domain == ResearchDomain.ACADEMIC:
            accuracy_score += 0.1  # Higher baseline for academic content
        
        return max(0.0, min(1.0, accuracy_score))
    
    def _fallback_citation_assessment(self, draft: Draft, content: str) -> float:
        """Fallback citation assessment using simple heuristics"""
        # Look for citation indicators
        citation_indicators = ["(", ")", "[", "]", "http", "www", "doi:", "et al"]
        citation_count = sum(content.count(indicator) for indicator in citation_indicators)
        
        # Estimate citation quality based on indicators
        content_length = len(content)
        if content_length == 0:
            return 0.0
        
        citation_density = citation_count / max(content_length / 1000, 1)  # Citations per 1000 chars
        citation_score = min(citation_density / 5, 1.0)  # Normalize to 0-1
        
        return citation_score

class KimiK2QualityChecker:
    """Kimi K2-powered decision logic for iteration control"""
    
    def __init__(self):
        """Initialize the quality checker with Kimi K2 client"""
        self.kimi_client = KimiK2Client()
    
    async def should_continue_iteration(self, quality_metrics: QualityMetrics, 
                                      iteration_count: int, requirements: ResearchRequirements) -> bool:
        """
        Determine if iteration should continue using Kimi K2 intelligence
        
        Args:
            quality_metrics: Current quality assessment
            iteration_count: Current iteration number
            requirements: Research requirements and constraints
            
        Returns:
            True if iteration should continue, False otherwise
        """
        logger.info(f"Evaluating iteration continuation (iteration {iteration_count})")
        
        # Hard constraints check first
        if iteration_count >= requirements.max_iterations:
            logger.info(f"Max iterations ({requirements.max_iterations}) reached")
            return False
        
        if quality_metrics.overall_score >= requirements.quality_threshold:
            logger.info(f"Quality threshold ({requirements.quality_threshold}) met")
            return False
        
        # Use Kimi K2 for intelligent decision making
        try:
            prompt = self._build_continuation_prompt(quality_metrics, iteration_count, requirements)
            
            response = await self.kimi_client.generate_structured_response(
                prompt,
                {
                    "should_continue": "boolean",
                    "reasoning": "detailed explanation of the decision",
                    "improvement_potential": "float between 0.0 and 1.0",
                    "recommended_focus": ["list of areas to focus on next iteration"]
                }
            )
            
            decision = response.get("should_continue", True)
            reasoning = response.get("reasoning", "No reasoning provided")
            
            logger.info(f"Kimi K2 continuation decision: {decision} - {reasoning}")
            return decision
            
        except Exception as e:
            logger.error(f"Kimi K2 continuation decision failed: {e}")
            # Fallback to simple logic
            return self._fallback_continuation_decision(quality_metrics, iteration_count, requirements)
    
    def _build_continuation_prompt(self, quality_metrics: QualityMetrics, 
                                 iteration_count: int, requirements: ResearchRequirements) -> str:
        """Build prompt for continuation decision"""
        return f"""
As an expert research workflow optimizer, decide whether to continue iterating on this research draft.

Current Quality Metrics:
- Overall Score: {quality_metrics.overall_score:.3f}
- Completeness: {quality_metrics.completeness:.3f}
- Coherence: {quality_metrics.coherence:.3f}
- Accuracy: {quality_metrics.accuracy:.3f}
- Citation Quality: {quality_metrics.citation_quality:.3f}

Iteration Status:
- Current Iteration: {iteration_count}
- Max Iterations: {requirements.max_iterations}
- Quality Threshold: {requirements.quality_threshold}

Consider:
1. Current quality vs. target threshold
2. Remaining iterations available
3. Potential for meaningful improvement
4. Diminishing returns from additional iterations
5. Cost-benefit of continued iteration

Decide whether to continue iteration or proceed to final synthesis.
Provide clear reasoning for your decision.
"""
    
    def _fallback_continuation_decision(self, quality_metrics: QualityMetrics, 
                                      iteration_count: int, requirements: ResearchRequirements) -> bool:
        """Fallback continuation decision using simple logic"""
        # Continue if we're significantly below threshold and have iterations left
        quality_gap = requirements.quality_threshold - quality_metrics.overall_score
        iterations_remaining = requirements.max_iterations - iteration_count
        
        # Continue if quality gap is significant and we have iterations
        if quality_gap > 0.1 and iterations_remaining > 0:
            return True
        
        # Continue if we're making progress (assume some progress each iteration)
        if quality_gap > 0.05 and iterations_remaining > 1:
            return True
        
        return False