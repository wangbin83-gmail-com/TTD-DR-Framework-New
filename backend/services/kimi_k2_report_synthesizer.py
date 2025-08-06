"""
Kimi K2-powered report synthesizer for TTD-DR framework.
Handles final report generation, formatting, and quality assurance.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from models.core import (
    Draft, QualityMetrics, EvolutionRecord, ResearchRequirements,
    Source, Section
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class KimiK2ReportSynthesizer:
    """
    Kimi K2-powered report synthesizer for generating final polished research reports
    """
    
    def __init__(self):
        self.kimi_client = KimiK2Client()
        
    async def synthesize_report(self, 
                              draft: Draft, 
                              quality_metrics: QualityMetrics,
                              evolution_history: List[EvolutionRecord],
                              requirements: Optional[ResearchRequirements] = None) -> str:
        """
        Generate final polished research report using Kimi K2
        
        Args:
            draft: Current research draft
            quality_metrics: Quality assessment metrics
            evolution_history: Self-evolution history
            requirements: Research requirements (optional)
            
        Returns:
            Final polished research report as string
        """
        logger.info(f"Starting Kimi K2 report synthesis for topic: {draft.topic}")
        
        try:
            # Build comprehensive synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(
                draft, quality_metrics, evolution_history, requirements
            )
            
            # Generate final report using Kimi K2
            response = await self.kimi_client.generate_text(
                synthesis_prompt,
                max_tokens=8000,  # Increased for comprehensive reports
                temperature=0.3   # Lower temperature for consistency
            )
            
            final_report = response.content
            
            # Apply final formatting and structure optimization
            formatted_report = await self._format_final_report(final_report, draft)
            
            logger.info(f"Report synthesis completed - Length: {len(formatted_report)} characters")
            return formatted_report
            
        except KimiK2Error as e:
            logger.error(f"Kimi K2 report synthesis failed: {e}")
            # Fallback to basic report generation
            return self._generate_fallback_report(draft, quality_metrics)
        except Exception as e:
            logger.error(f"Unexpected error in report synthesis: {e}")
            return self._generate_fallback_report(draft, quality_metrics)
    
    def _build_synthesis_prompt(self, 
                               draft: Draft, 
                               quality_metrics: QualityMetrics,
                               evolution_history: List[EvolutionRecord],
                               requirements: Optional[ResearchRequirements]) -> str:
        """
        Build comprehensive prompt for Kimi K2 report synthesis
        
        Args:
            draft: Current research draft
            quality_metrics: Quality metrics
            evolution_history: Evolution history
            requirements: Research requirements
            
        Returns:
            Formatted prompt for Kimi K2
        """
        # Extract key information
        topic = draft.topic
        structure_info = self._extract_structure_info(draft.structure)
        content_summary = self._extract_content_summary(draft.content)
        quality_summary = self._extract_quality_summary(quality_metrics)
        evolution_insights = self._extract_evolution_insights(evolution_history)
        
        # Build domain-specific requirements
        domain_requirements = ""
        if requirements:
            domain_requirements = f"""
Domain: {requirements.domain.value}
Complexity Level: {requirements.complexity_level.value}
Quality Threshold: {requirements.quality_threshold}
Preferred Source Types: {', '.join(requirements.preferred_source_types)}
"""
        
        prompt = f"""You are an expert research report synthesizer tasked with creating a final, polished research report. Your goal is to transform the provided research draft into a comprehensive, professional, and well-structured final report.

RESEARCH TOPIC: {topic}

CURRENT DRAFT STRUCTURE:
{structure_info}

DRAFT CONTENT SUMMARY:
{content_summary}

QUALITY ASSESSMENT:
{quality_summary}

EVOLUTION INSIGHTS:
{evolution_insights}

RESEARCH REQUIREMENTS:
{domain_requirements}

SYNTHESIS INSTRUCTIONS:

1. COMPREHENSIVE INTEGRATION:
   - Synthesize all draft content into a cohesive narrative
   - Ensure logical flow between sections
   - Integrate insights from the evolution process
   - Address any remaining quality gaps

2. PROFESSIONAL FORMATTING:
   - Use clear, professional academic writing style
   - Include proper headings and subheadings
   - Ensure consistent formatting throughout
   - Add executive summary and conclusion

3. QUALITY ENHANCEMENT:
   - Strengthen weak areas identified in quality metrics
   - Improve coherence and readability
   - Enhance evidence presentation
   - Optimize citation integration

4. STRUCTURE OPTIMIZATION:
   - Follow standard research report format
   - Ensure balanced section lengths
   - Create smooth transitions between topics
   - Add methodology section if appropriate

5. FINAL POLISH:
   - Eliminate redundancy and improve clarity
   - Ensure professional tone throughout
   - Verify factual consistency
   - Optimize for target audience

Please generate a complete, final research report that represents the culmination of the iterative research process. The report should be publication-ready and demonstrate the highest quality standards.

FINAL REPORT:"""

        return prompt
    
    def _extract_structure_info(self, structure) -> str:
        """Extract structure information for prompt"""
        sections_info = []
        for section in structure.sections:
            sections_info.append(f"- {section.title} (ID: {section.id})")
            for subsection in section.subsections:
                sections_info.append(f"  - {subsection.title} (ID: {subsection.id})")
        
        return f"""
Sections ({len(structure.sections)} main sections):
{chr(10).join(sections_info)}

Estimated Length: {structure.estimated_length} words
Complexity: {structure.complexity_level.value}
Domain: {structure.domain.value}
"""
    
    def _extract_content_summary(self, content: Dict[str, str]) -> str:
        """Extract content summary for prompt"""
        if not content:
            return "No content available in draft"
        
        summary_parts = []
        for section_id, section_content in content.items():
            content_length = len(section_content.split())
            content_preview = section_content[:200] + "..." if len(section_content) > 200 else section_content
            summary_parts.append(f"Section {section_id}: {content_length} words - {content_preview}")
        
        return "\n".join(summary_parts)
    
    def _extract_quality_summary(self, quality_metrics: QualityMetrics) -> str:
        """Extract quality summary for prompt"""
        return f"""
Overall Score: {quality_metrics.overall_score:.3f}
- Completeness: {quality_metrics.completeness:.3f}
- Coherence: {quality_metrics.coherence:.3f}
- Accuracy: {quality_metrics.accuracy:.3f}
- Citation Quality: {quality_metrics.citation_quality:.3f}

Quality Grade: {self._get_quality_grade(quality_metrics.overall_score)}
"""
    
    def _extract_evolution_insights(self, evolution_history: List[EvolutionRecord]) -> str:
        """Extract evolution insights for prompt"""
        if not evolution_history:
            return "No evolution history available"
        
        recent_records = evolution_history[-3:]  # Last 3 evolution records
        insights = []
        
        for record in recent_records:
            improvement = record.performance_after - record.performance_before
            insights.append(f"- {record.component}: {record.improvement_type} (Δ{improvement:+.3f}) - {record.description}")
        
        total_improvement = sum(r.performance_after - r.performance_before for r in evolution_history)
        
        return f"""
Recent Evolution Activities ({len(recent_records)} records):
{chr(10).join(insights)}

Total Performance Improvement: {total_improvement:+.3f}
Evolution Cycles: {len(evolution_history)}
"""
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return "A (Excellent)"
        elif score >= 0.8:
            return "B (Good)"
        elif score >= 0.7:
            return "C (Satisfactory)"
        elif score >= 0.6:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"
    
    async def _format_final_report(self, report_content: str, draft: Draft) -> str:
        """
        Apply final formatting and structure optimization using Kimi K2
        
        Args:
            report_content: Raw report content from synthesis
            draft: Original draft for context
            
        Returns:
            Formatted final report
        """
        try:
            formatting_prompt = f"""Please apply professional formatting to this research report. Ensure it follows standard academic/professional report structure with:

1. Title page information
2. Executive summary
3. Table of contents (if appropriate)
4. Main content with proper headings
5. Conclusion
6. References/Bibliography (if citations present)

Original Topic: {draft.topic}
Report Date: {datetime.now().strftime("%B %d, %Y")}

Raw Report Content:
{report_content}

Please return the professionally formatted version:"""

            response = await self.kimi_client.generate_text(
                formatting_prompt,
                max_tokens=8000,
                temperature=0.2
            )
            
            return response.content
            
        except Exception as e:
            logger.warning(f"Report formatting failed, using original content: {e}")
            return self._apply_basic_formatting(report_content, draft)
    
    def _apply_basic_formatting(self, content: str, draft: Draft) -> str:
        """Apply basic formatting as fallback"""
        formatted_report = f"""# {draft.topic}

**Research Report**  
Generated by TTD-DR Framework  
Date: {datetime.now().strftime("%B %d, %Y")}

---

{content}

---

*Report generated using Test-Time Diffusion Deep Researcher (TTD-DR) Framework*  
*Quality Score: {draft.quality_score:.3f}*  
*Iteration: {draft.iteration}*
"""
        return formatted_report
    
    def _generate_fallback_report(self, draft: Draft, quality_metrics: QualityMetrics) -> str:
        """Generate basic fallback report when Kimi K2 fails"""
        logger.info("Generating fallback report due to synthesis failure")
        
        # Combine all draft content
        combined_content = []
        for section_id, content in draft.content.items():
            if content.strip():
                combined_content.append(f"## {section_id}\n\n{content}")
        
        fallback_report = f"""# {draft.topic}

**Research Report - Fallback Generation**  
Generated by TTD-DR Framework  
Date: {datetime.now().strftime("%B %d, %Y")}

## Executive Summary

This research report on "{draft.topic}" was generated through an iterative research process. The report synthesizes information from multiple sources and has undergone {draft.iteration} iterations of refinement.

**Quality Metrics:**
- Overall Score: {quality_metrics.overall_score:.3f}
- Completeness: {quality_metrics.completeness:.3f}
- Coherence: {quality_metrics.coherence:.3f}
- Accuracy: {quality_metrics.accuracy:.3f}
- Citation Quality: {quality_metrics.citation_quality:.3f}

## Research Content

{chr(10).join(combined_content) if combined_content else "No content available in draft."}

## Conclusion

This report represents the current state of research on the topic "{draft.topic}". The research process involved iterative refinement and quality assessment to ensure comprehensive coverage of the subject matter.

---

*Report generated using Test-Time Diffusion Deep Researcher (TTD-DR) Framework*  
*Framework Version: 1.0*  
*Generation Method: Fallback (Kimi K2 synthesis unavailable)*
"""
        
        return fallback_report

    async def generate_executive_summary(self, report: str, max_length: int = 500) -> str:
        """
        Generate executive summary for the report using Kimi K2
        
        Args:
            report: Full report content
            max_length: Maximum length of summary in words
            
        Returns:
            Executive summary
        """
        try:
            summary_prompt = f"""Please create a concise executive summary for this research report. The summary should:

1. Capture the main findings and conclusions
2. Highlight key insights and recommendations
3. Be approximately {max_length} words or less
4. Use clear, professional language
5. Stand alone as a complete overview

Research Report:
{report[:4000]}  # Truncate for prompt limits

Executive Summary:"""

            response = await self.kimi_client.generate_text(
                summary_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return self._generate_basic_summary(report, max_length)
    
    def _generate_basic_summary(self, report: str, max_length: int) -> str:
        """Generate basic summary as fallback"""
        # Extract first few sentences as basic summary
        sentences = report.split('. ')
        summary_sentences = []
        word_count = 0
        
        for sentence in sentences[:5]:  # Max 5 sentences
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= max_length:
                summary_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else "Summary not available."

    async def validate_report_quality(self, report: str, original_draft: Draft) -> Dict[str, Any]:
        """
        Validate the quality of the synthesized report using Kimi K2
        
        Args:
            report: Final synthesized report
            original_draft: Original draft for comparison
            
        Returns:
            Quality validation results
        """
        try:
            validation_prompt = f"""Please evaluate the quality of this synthesized research report against the original draft. Assess:

1. Content completeness (0-1 scale)
2. Structural coherence (0-1 scale)
3. Professional formatting (0-1 scale)
4. Information accuracy (0-1 scale)
5. Overall synthesis quality (0-1 scale)

Original Topic: {original_draft.topic}
Original Quality Score: {original_draft.quality_score}

Synthesized Report:
{report[:3000]}  # Truncate for prompt limits

Please respond with a JSON object containing your assessment:
{{
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "formatting": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "overall_quality": 0.0-1.0,
    "improvement_over_draft": 0.0-1.0,
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"],
    "recommendation": "accept/revise/reject"
}}"""

            validation_schema = {
                "completeness": "number",
                "coherence": "number", 
                "formatting": "number",
                "accuracy": "number",
                "overall_quality": "number",
                "improvement_over_draft": "number",
                "strengths": "array",
                "areas_for_improvement": "array",
                "recommendation": "string"
            }
            
            validation_result = await self.kimi_client.generate_structured_response(
                validation_prompt,
                validation_schema,
                max_tokens=1000,
                temperature=0.2
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Report quality validation failed: {e}")
            return self._generate_basic_validation(report, original_draft)
    
    def _generate_basic_validation(self, report: str, original_draft: Draft) -> Dict[str, Any]:
        """Generate basic validation as fallback"""
        report_length = len(report.split())
        draft_length = sum(len(content.split()) for content in original_draft.content.values())
        
        # Basic heuristic assessment
        length_improvement = min(1.0, report_length / max(draft_length, 100))
        structure_score = 0.8 if "# " in report and "## " in report else 0.6
        
        return {
            "completeness": min(1.0, length_improvement),
            "coherence": structure_score,
            "formatting": structure_score,
            "accuracy": original_draft.quality_score,  # Assume maintained
            "overall_quality": (length_improvement + structure_score + original_draft.quality_score) / 3,
            "improvement_over_draft": max(0.0, length_improvement - original_draft.quality_score),
            "strengths": ["Content synthesis", "Basic structure"],
            "areas_for_improvement": ["Professional formatting", "Citation integration"],
            "recommendation": "accept" if length_improvement > 0.7 else "revise"
        }

    # Research Methodology Documentation Methods (Task 9.2)
    
    async def generate_research_methodology_documentation(self, 
                                                        state: Dict[str, Any],
                                                        workflow_log: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate comprehensive research methodology documentation using Kimi K2
        
        Args:
            state: Complete workflow state containing all research data
            workflow_log: Optional workflow execution log
            
        Returns:
            Comprehensive methodology documentation
        """
        logger.info("Generating research methodology documentation with Kimi K2")
        
        try:
            # Extract methodology components from state
            methodology_data = self._extract_methodology_data(state, workflow_log)
            
            # Build methodology documentation prompt
            methodology_prompt = self._build_methodology_prompt(methodology_data)
            
            # Generate methodology documentation using Kimi K2
            response = await self.kimi_client.generate_text(
                methodology_prompt,
                max_tokens=4000,
                temperature=0.2  # Low temperature for factual documentation
            )
            
            methodology_doc = response.content
            
            # Enhance with structured sections
            enhanced_methodology = await self._enhance_methodology_structure(methodology_doc, methodology_data)
            
            logger.info(f"Research methodology documentation generated - Length: {len(enhanced_methodology)} characters")
            return enhanced_methodology
            
        except Exception as e:
            logger.error(f"Methodology documentation generation failed: {e}")
            return self._generate_fallback_methodology(state, workflow_log)
    
    def _extract_methodology_data(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract methodology data from workflow state and logs
        
        Args:
            state: Complete workflow state
            workflow_log: Optional workflow execution log
            
        Returns:
            Dictionary containing methodology data
        """
        methodology_data = {
            "research_topic": state.get("topic", "Unknown Topic"),
            "requirements": state.get("requirements"),
            "workflow_stages": [],
            "iteration_count": state.get("iteration_count", 0),
            "quality_metrics": state.get("quality_metrics"),
            "evolution_history": state.get("evolution_history", []),
            "sources_used": [],
            "search_queries": [],
            "information_gaps": state.get("information_gaps", []),
            "retrieved_info": state.get("retrieved_info", []),
            "workflow_execution_time": None,
            "methodology_approach": "TTD-DR (Test-Time Diffusion Deep Researcher)"
        }
        
        # Extract workflow stages from state
        if state.get("current_draft"):
            methodology_data["workflow_stages"].append({
                "stage": "Draft Generation",
                "description": "Initial research skeleton creation",
                "output": f"Generated {len(state['current_draft'].structure.sections)} main sections"
            })
        
        if state.get("information_gaps"):
            methodology_data["workflow_stages"].append({
                "stage": "Gap Analysis", 
                "description": "Identification of information gaps",
                "output": f"Identified {len(state['information_gaps'])} information gaps"
            })
        
        if state.get("retrieved_info"):
            methodology_data["workflow_stages"].append({
                "stage": "Information Retrieval",
                "description": "Dynamic retrieval of external information",
                "output": f"Retrieved {len(state['retrieved_info'])} information sources"
            })
            
            # Extract sources and queries
            for info in state["retrieved_info"]:
                if hasattr(info, 'source') and info.source:
                    methodology_data["sources_used"].append({
                        "url": getattr(info.source, 'url', 'Unknown URL'),
                        "title": getattr(info.source, 'title', 'Unknown Title'),
                        "credibility_score": getattr(info, 'credibility_score', 0.0),
                        "relevance_score": getattr(info, 'relevance_score', 0.0)
                    })
        
        if state.get("quality_metrics"):
            methodology_data["workflow_stages"].append({
                "stage": "Quality Assessment",
                "description": "Evaluation of research quality and completeness",
                "output": f"Overall quality score: {state['quality_metrics'].overall_score:.3f}"
            })
        
        if state.get("evolution_history"):
            methodology_data["workflow_stages"].append({
                "stage": "Self-Evolution Enhancement",
                "description": "Adaptive improvement of research components",
                "output": f"Applied {len(state['evolution_history'])} evolution cycles"
            })
        
        # Extract search queries from gaps
        for gap in state.get("information_gaps", []):
            if hasattr(gap, 'search_queries'):
                methodology_data["search_queries"].extend(gap.search_queries)
        
        # Extract workflow execution data from log if available
        if workflow_log:
            methodology_data["workflow_execution_time"] = self._calculate_execution_time(workflow_log)
            methodology_data["workflow_log"] = workflow_log[-10:]  # Last 10 log entries
        
        return methodology_data
    
    def _calculate_execution_time(self, workflow_log: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate total workflow execution time from log"""
        try:
            if not workflow_log:
                return None
            
            start_time = None
            end_time = None
            
            for entry in workflow_log:
                timestamp = entry.get("timestamp")
                if timestamp:
                    if start_time is None or timestamp < start_time:
                        start_time = timestamp
                    if end_time is None or timestamp > end_time:
                        end_time = timestamp
            
            if start_time and end_time:
                return end_time - start_time
            
        except Exception as e:
            logger.warning(f"Could not calculate execution time: {e}")
        
        return None
    
    def _build_methodology_prompt(self, methodology_data: Dict[str, Any]) -> str:
        """
        Build comprehensive methodology documentation prompt for Kimi K2
        
        Args:
            methodology_data: Extracted methodology data
            
        Returns:
            Formatted prompt for methodology documentation
        """
        # Format workflow stages
        stages_text = ""
        for i, stage in enumerate(methodology_data["workflow_stages"], 1):
            stages_text += f"{i}. **{stage['stage']}**: {stage['description']}\n   - {stage['output']}\n"
        
        # Format sources
        sources_text = ""
        if methodology_data["sources_used"]:
            for i, source in enumerate(methodology_data["sources_used"][:10], 1):  # Limit to top 10
                sources_text += f"{i}. {source['title']} ({source['url']})\n   - Credibility: {source['credibility_score']:.2f}, Relevance: {source['relevance_score']:.2f}\n"
        else:
            sources_text = "No external sources were retrieved during this research process."
        
        # Format search queries
        queries_text = ""
        if methodology_data["search_queries"]:
            unique_queries = list(set(methodology_data["search_queries"][:15]))  # Limit and deduplicate
            for i, query in enumerate(unique_queries, 1):
                queries_text += f"{i}. \"{query}\"\n"
        else:
            queries_text = "No search queries were generated during this research process."
        
        # Format quality metrics
        quality_text = ""
        if methodology_data["quality_metrics"]:
            metrics = methodology_data["quality_metrics"]
            quality_text = f"""
- Overall Quality Score: {metrics.overall_score:.3f}
- Completeness: {metrics.completeness:.3f}
- Coherence: {metrics.coherence:.3f}
- Accuracy: {metrics.accuracy:.3f}
- Citation Quality: {metrics.citation_quality:.3f}
"""
        else:
            quality_text = "Quality metrics were not available for this research process."
        
        # Format evolution summary
        evolution_text = ""
        if methodology_data["evolution_history"]:
            total_improvement = sum(
                r.performance_after - r.performance_before 
                for r in methodology_data["evolution_history"]
            )
            evolution_text = f"""
- Total Evolution Cycles: {len(methodology_data["evolution_history"])}
- Cumulative Performance Improvement: {total_improvement:+.3f}
- Components Enhanced: {len(set(r.component for r in methodology_data["evolution_history"]))}
"""
        else:
            evolution_text = "No self-evolution enhancements were applied during this research process."
        
        # Format execution time
        execution_text = ""
        if methodology_data["workflow_execution_time"]:
            execution_text = f"Total Execution Time: {methodology_data['workflow_execution_time']:.2f} seconds"
        else:
            execution_text = "Execution time data not available."
        
        prompt = f"""You are tasked with creating comprehensive research methodology documentation for a research report generated using the TTD-DR (Test-Time Diffusion Deep Researcher) framework. This documentation should provide complete transparency about the research process, methods used, and quality assurance measures applied.

RESEARCH OVERVIEW:
Topic: {methodology_data["research_topic"]}
Methodology Framework: {methodology_data["methodology_approach"]}
Total Iterations: {methodology_data["iteration_count"]}
{execution_text}

WORKFLOW STAGES EXECUTED:
{stages_text}

INFORMATION SOURCES UTILIZED:
{sources_text}

SEARCH QUERIES GENERATED:
{queries_text}

QUALITY ASSESSMENT RESULTS:
{quality_text}

SELF-EVOLUTION ENHANCEMENTS:
{evolution_text}

DOCUMENTATION REQUIREMENTS:

Please generate a comprehensive research methodology documentation that includes:

1. **Research Framework Overview**
   - Explanation of the TTD-DR methodology
   - Theoretical foundation and approach
   - Key principles and innovations

2. **Research Process Documentation**
   - Detailed description of each workflow stage
   - Decision points and criteria used
   - Iteration logic and convergence criteria

3. **Information Retrieval Methodology**
   - Search strategy and query formulation
   - Source selection and validation criteria
   - Information integration approaches

4. **Quality Assurance Framework**
   - Quality metrics and assessment criteria
   - Validation procedures and thresholds
   - Continuous improvement mechanisms

5. **Self-Evolution Enhancement Process**
   - Adaptive learning algorithms applied
   - Performance optimization strategies
   - Component enhancement methodologies

6. **Limitations and Considerations**
   - Methodological limitations
   - Potential biases and mitigation strategies
   - Scope and applicability constraints

7. **Reproducibility Guidelines**
   - Steps for reproducing the research
   - Parameter settings and configurations
   - Data requirements and dependencies

The documentation should be written in a professional, academic style suitable for peer review and should provide sufficient detail for methodology replication and validation.

RESEARCH METHODOLOGY DOCUMENTATION:"""

        return prompt
    
    async def _enhance_methodology_structure(self, methodology_doc: str, methodology_data: Dict[str, Any]) -> str:
        """
        Enhance methodology documentation with structured sections and formatting
        
        Args:
            methodology_doc: Raw methodology documentation from Kimi K2
            methodology_data: Original methodology data
            
        Returns:
            Enhanced and structured methodology documentation
        """
        try:
            enhancement_prompt = f"""Please enhance this research methodology documentation by adding proper academic structure, formatting, and additional technical details. Ensure it follows standard research methodology documentation format with:

1. Clear section headers and subsections
2. Professional academic writing style
3. Proper citation format for the TTD-DR framework
4. Technical specifications and parameters
5. Appendices for detailed data

Original Research Topic: {methodology_data["research_topic"]}
Framework: {methodology_data["methodology_approach"]}

Raw Methodology Documentation:
{methodology_doc}

Please return the enhanced, professionally formatted methodology documentation:"""

            response = await self.kimi_client.generate_text(
                enhancement_prompt,
                max_tokens=5000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            enhanced_doc = response.content
            
            # Add technical appendix
            technical_appendix = self._generate_technical_appendix(methodology_data)
            
            return f"{enhanced_doc}\n\n{technical_appendix}"
            
        except Exception as e:
            logger.warning(f"Methodology enhancement failed, using original: {e}")
            return self._apply_basic_methodology_formatting(methodology_doc, methodology_data)
    
    def _generate_technical_appendix(self, methodology_data: Dict[str, Any]) -> str:
        """
        Generate technical appendix with detailed parameters and data
        
        Args:
            methodology_data: Methodology data
            
        Returns:
            Technical appendix content
        """
        appendix = f"""
## Technical Appendix

### A. Framework Parameters

**TTD-DR Configuration:**
- Research Topic: {methodology_data["research_topic"]}
- Total Iterations: {methodology_data["iteration_count"]}
- Quality Threshold: {methodology_data.get("requirements", {}).get("quality_threshold", "Not specified") if methodology_data.get("requirements") else "Not specified"}

### B. Workflow Execution Details

**Stage Execution Summary:**
"""
        
        for i, stage in enumerate(methodology_data["workflow_stages"], 1):
            appendix += f"{i}. {stage['stage']}: {stage['output']}\n"
        
        appendix += f"""
### C. Information Sources Summary

**Total Sources Retrieved:** {len(methodology_data["sources_used"])}
**Search Queries Generated:** {len(methodology_data["search_queries"])}

**Source Quality Distribution:**
"""
        
        if methodology_data["sources_used"]:
            high_quality = sum(1 for s in methodology_data["sources_used"] if s["credibility_score"] >= 0.8)
            medium_quality = sum(1 for s in methodology_data["sources_used"] if 0.5 <= s["credibility_score"] < 0.8)
            low_quality = sum(1 for s in methodology_data["sources_used"] if s["credibility_score"] < 0.5)
            
            appendix += f"""- High Quality (≥0.8): {high_quality} sources
- Medium Quality (0.5-0.8): {medium_quality} sources  
- Low Quality (<0.5): {low_quality} sources
"""
        else:
            appendix += "- No external sources were utilized\n"
        
        appendix += f"""
### D. Quality Metrics Details

"""
        
        if methodology_data["quality_metrics"]:
            metrics = methodology_data["quality_metrics"]
            appendix += f"""**Final Quality Assessment:**
- Overall Score: {metrics.overall_score:.4f}
- Completeness: {metrics.completeness:.4f}
- Coherence: {metrics.coherence:.4f}
- Accuracy: {metrics.accuracy:.4f}
- Citation Quality: {metrics.citation_quality:.4f}
"""
        else:
            appendix += "Quality metrics were not available for this research execution.\n"
        
        appendix += f"""
### E. Self-Evolution Enhancement Log

"""
        
        if methodology_data["evolution_history"]:
            appendix += f"**Total Enhancement Cycles:** {len(methodology_data['evolution_history'])}\n\n"
            
            for i, record in enumerate(methodology_data["evolution_history"][-5:], 1):  # Last 5 records
                improvement = record.performance_after - record.performance_before
                appendix += f"{i}. Component: {record.component}\n"
                appendix += f"   - Enhancement Type: {record.improvement_type}\n"
                appendix += f"   - Performance Change: {improvement:+.4f}\n"
                appendix += f"   - Description: {record.description}\n\n"
        else:
            appendix += "No self-evolution enhancements were applied during this research process.\n"
        
        appendix += f"""
### F. Reproducibility Information

**Framework Version:** TTD-DR v1.0
**Execution Environment:** Python-based LangGraph workflow
**External APIs Used:** 
- Kimi K2 Language Model
- Google Search API (if applicable)

**Key Dependencies:**
- LangGraph for workflow orchestration
- Kimi K2 client for AI-powered processing
- Custom TTD-DR components for research automation

---

*This technical appendix provides detailed information for research methodology validation and reproducibility.*
"""
        
        return appendix
    
    def _apply_basic_methodology_formatting(self, methodology_doc: str, methodology_data: Dict[str, Any]) -> str:
        """Apply basic formatting as fallback for methodology documentation"""
        formatted_doc = f"""# Research Methodology Documentation

**Research Topic:** {methodology_data["research_topic"]}  
**Framework:** {methodology_data["methodology_approach"]}  
**Date Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Total Iterations:** {methodology_data["iteration_count"]}

---

{methodology_doc}

---

## Technical Summary

**Workflow Stages Completed:** {len(methodology_data["workflow_stages"])}  
**Information Sources Used:** {len(methodology_data["sources_used"])}  
**Search Queries Generated:** {len(methodology_data["search_queries"])}  
**Evolution Cycles Applied:** {len(methodology_data["evolution_history"])}

*Methodology documentation generated by TTD-DR Framework*
"""
        return formatted_doc
    
    def _generate_fallback_methodology(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> str:
        """Generate fallback methodology documentation when Kimi K2 fails"""
        logger.info("Generating fallback methodology documentation")
        
        topic = state.get("topic", "Unknown Topic")
        iteration_count = state.get("iteration_count", 0)
        
        # Basic workflow summary
        workflow_summary = []
        if state.get("current_draft"):
            workflow_summary.append("1. Initial draft generation completed")
        if state.get("information_gaps"):
            workflow_summary.append(f"2. Information gap analysis identified {len(state['information_gaps'])} gaps")
        if state.get("retrieved_info"):
            workflow_summary.append(f"3. Information retrieval gathered {len(state['retrieved_info'])} sources")
        if state.get("quality_metrics"):
            workflow_summary.append(f"4. Quality assessment achieved {state['quality_metrics'].overall_score:.3f} score")
        if state.get("evolution_history"):
            workflow_summary.append(f"5. Self-evolution applied {len(state['evolution_history'])} enhancements")
        
        fallback_methodology = f"""# Research Methodology Documentation

**Research Topic:** {topic}  
**Framework:** TTD-DR (Test-Time Diffusion Deep Researcher)  
**Date Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Generation Method:** Fallback Documentation

## Methodology Overview

This research was conducted using the TTD-DR framework, which implements a diffusion-inspired approach to automated research report generation. The framework treats research as an iterative refinement process, starting with a preliminary draft and progressively improving it through multiple cycles of analysis, information retrieval, and integration.

## Research Process

The research process completed {iteration_count} iterations and included the following stages:

{chr(10).join(workflow_summary) if workflow_summary else "No detailed workflow information available."}

## Framework Components

The TTD-DR framework consists of the following key components:

1. **Draft Generator**: Creates initial research skeleton using AI-powered topic analysis
2. **Gap Analyzer**: Identifies areas requiring additional information or improvement
3. **Retrieval Engine**: Dynamically retrieves relevant information from external sources
4. **Information Integrator**: Seamlessly incorporates new information into existing draft
5. **Quality Assessor**: Evaluates research quality and determines iteration needs
6. **Self-Evolution Enhancer**: Applies adaptive learning to improve component performance
7. **Report Synthesizer**: Generates final polished research report

## Quality Assurance

The research process included continuous quality assessment to ensure:
- Content completeness and accuracy
- Logical coherence and flow
- Proper source integration and citation
- Professional formatting and presentation

## Limitations

This methodology documentation was generated using fallback procedures due to technical limitations in the primary documentation system. The content may lack the depth and detail typically provided by the full TTD-DR methodology documentation process.

## Conclusion

The TTD-DR framework provides a systematic approach to automated research that combines the benefits of iterative refinement with AI-powered content generation and quality assessment. This methodology enables the production of comprehensive research reports while maintaining transparency and reproducibility.

---

*Methodology documentation generated by TTD-DR Framework v1.0*  
*Documentation Method: Fallback Generation*  
*Total Iterations: {iteration_count}*
"""
        
        return fallback_methodology

    async def generate_source_bibliography(self, retrieved_info: List[Any], citation_style: str = "APA") -> str:
        """
        Generate formatted bibliography from retrieved sources using Kimi K2
        
        Args:
            retrieved_info: List of retrieved information with sources
            citation_style: Citation style (APA, MLA, Chicago, etc.)
            
        Returns:
            Formatted bibliography
        """
        logger.info(f"Generating source bibliography in {citation_style} style with Kimi K2")
        
        if not retrieved_info:
            return "No sources were retrieved during the research process."
        
        try:
            # Extract unique sources
            unique_sources = self._extract_unique_sources(retrieved_info)
            
            # Build bibliography prompt
            bibliography_prompt = self._build_bibliography_prompt(unique_sources, citation_style)
            
            # Generate bibliography using Kimi K2
            response = await self.kimi_client.generate_text(
                bibliography_prompt,
                max_tokens=3000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            bibliography = response.content
            
            # Validate and enhance bibliography
            enhanced_bibliography = await self._enhance_bibliography_formatting(bibliography, citation_style)
            
            logger.info(f"Bibliography generated with {len(unique_sources)} sources")
            return enhanced_bibliography
            
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return self._generate_fallback_bibliography(retrieved_info, citation_style)
    
    def _extract_unique_sources(self, retrieved_info: List[Any]) -> List[Dict[str, Any]]:
        """Extract unique sources from retrieved information"""
        unique_sources = []
        seen_urls = set()
        
        for info in retrieved_info:
            if hasattr(info, 'source') and info.source:
                source = info.source
                url = getattr(source, 'url', '')
                
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append({
                        'title': getattr(source, 'title', 'Unknown Title'),
                        'url': url,
                        'author': getattr(source, 'author', ''),
                        'publication_date': getattr(source, 'publication_date', ''),
                        'domain': getattr(source, 'domain', ''),
                        'credibility_score': getattr(info, 'credibility_score', 0.0),
                        'access_date': getattr(info, 'extraction_timestamp', datetime.now()).strftime("%Y-%m-%d")
                    })
        
        # Sort by credibility score (highest first)
        unique_sources.sort(key=lambda x: x['credibility_score'], reverse=True)
        
        return unique_sources
    
    def _build_bibliography_prompt(self, sources: List[Dict[str, Any]], citation_style: str) -> str:
        """Build bibliography generation prompt for Kimi K2"""
        
        sources_text = ""
        for i, source in enumerate(sources, 1):
            sources_text += f"""
{i}. Title: {source['title']}
   URL: {source['url']}
   Author: {source['author'] or 'Unknown Author'}
   Publication Date: {source['publication_date'] or 'Unknown Date'}
   Domain: {source['domain']}
   Access Date: {source['access_date']}
   Credibility Score: {source['credibility_score']:.2f}
"""
        
        prompt = f"""You are tasked with creating a properly formatted bibliography from the following sources using {citation_style} citation style. Please generate accurate, consistent citations following standard {citation_style} formatting guidelines.

SOURCES TO CITE:
{sources_text}

FORMATTING REQUIREMENTS:

1. Use proper {citation_style} citation format
2. Include all available information (author, title, URL, access date, etc.)
3. Handle missing information appropriately (use "Unknown Author", "n.d." for no date, etc.)
4. Sort entries alphabetically by author/title as appropriate for {citation_style}
5. Use proper indentation and formatting
6. Include DOI or URL as required by {citation_style}

Please generate a complete bibliography section with proper heading and formatting:

BIBLIOGRAPHY:"""

        return prompt
    
    async def _enhance_bibliography_formatting(self, bibliography: str, citation_style: str) -> str:
        """Enhance bibliography formatting and validation"""
        try:
            enhancement_prompt = f"""Please review and enhance this bibliography to ensure it strictly follows {citation_style} formatting standards. Check for:

1. Proper {citation_style} citation format
2. Consistent formatting throughout
3. Correct punctuation and capitalization
4. Proper handling of web sources
5. Alphabetical ordering (if required by {citation_style})

Original Bibliography:
{bibliography}

Please return the corrected and enhanced bibliography:"""

            response = await self.kimi_client.generate_text(
                enhancement_prompt,
                max_tokens=3000,
                temperature=0.1
            )
            
            return response.content
            
        except Exception as e:
            logger.warning(f"Bibliography enhancement failed: {e}")
            return self._apply_basic_bibliography_formatting(bibliography, citation_style)
    
    def _apply_basic_bibliography_formatting(self, bibliography: str, citation_style: str) -> str:
        """Apply basic bibliography formatting as fallback"""
        formatted_bibliography = f"""# Bibliography ({citation_style} Style)

{bibliography}

---

*Bibliography generated using TTD-DR Framework*  
*Citation Style: {citation_style}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        return formatted_bibliography
    
    def _generate_fallback_bibliography(self, retrieved_info: List[Any], citation_style: str) -> str:
        """Generate basic fallback bibliography"""
        logger.info("Generating fallback bibliography")
        
        unique_sources = self._extract_unique_sources(retrieved_info)
        
        if not unique_sources:
            return f"""# Bibliography ({citation_style} Style)

No sources were retrieved during the research process.

---

*Bibliography generated using TTD-DR Framework (Fallback)*  
*Citation Style: {citation_style}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        
        bibliography_entries = []
        for i, source in enumerate(unique_sources, 1):
            # Basic citation format (simplified)
            author = source['author'] or "Unknown Author"
            title = source['title']
            url = source['url']
            access_date = source['access_date']
            
            if citation_style.upper() == "APA":
                entry = f"{author}. ({source['publication_date'] or 'n.d.'}). {title}. Retrieved {access_date}, from {url}"
            elif citation_style.upper() == "MLA":
                entry = f"{author}. \"{title}.\" Web. {access_date}. <{url}>."
            else:
                entry = f"{author}. \"{title}.\" Accessed {access_date}. {url}."
            
            bibliography_entries.append(f"{i}. {entry}")
        
        fallback_bibliography = f"""# Bibliography ({citation_style} Style)

{chr(10).join(bibliography_entries)}

---

*Bibliography generated using TTD-DR Framework (Fallback)*  
*Citation Style: {citation_style}*  
*Total Sources: {len(unique_sources)}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        
        return fallback_bibliography

    async def generate_methodology_summary(self, state: Dict[str, Any]) -> str:
        """
        Generate concise methodology summary using Kimi K2
        
        Args:
            state: Complete workflow state
            
        Returns:
            Concise methodology summary
        """
        logger.info("Generating methodology summary with Kimi K2")
        
        try:
            # Extract key methodology points
            methodology_points = self._extract_methodology_summary_points(state)
            
            # Build summary prompt
            summary_prompt = self._build_methodology_summary_prompt(methodology_points)
            
            # Generate summary using Kimi K2
            response = await self.kimi_client.generate_text(
                summary_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            methodology_summary = response.content
            
            logger.info(f"Methodology summary generated - Length: {len(methodology_summary)} characters")
            return methodology_summary
            
        except Exception as e:
            logger.error(f"Methodology summary generation failed: {e}")
            return self._generate_fallback_methodology_summary(state)
    
    def _extract_methodology_summary_points(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key points for methodology summary"""
        return {
            "topic": state.get("topic", "Unknown Topic"),
            "framework": "TTD-DR (Test-Time Diffusion Deep Researcher)",
            "iterations": state.get("iteration_count", 0),
            "sources_count": len(state.get("retrieved_info", [])),
            "gaps_identified": len(state.get("information_gaps", [])),
            "quality_score": state.get("quality_metrics", {}).get("overall_score", 0.0) if state.get("quality_metrics") else 0.0,
            "evolution_cycles": len(state.get("evolution_history", [])),
            "has_final_report": bool(state.get("final_report"))
        }
    
    def _build_methodology_summary_prompt(self, methodology_points: Dict[str, Any]) -> str:
        """Build methodology summary prompt for Kimi K2"""
        prompt = f"""Please create a concise methodology summary for a research report generated using the TTD-DR framework. The summary should be 2-3 paragraphs and suitable for inclusion in the main research report.

RESEARCH DETAILS:
- Topic: {methodology_points["topic"]}
- Framework: {methodology_points["framework"]}
- Iterations Completed: {methodology_points["iterations"]}
- External Sources Retrieved: {methodology_points["sources_count"]}
- Information Gaps Identified: {methodology_points["gaps_identified"]}
- Final Quality Score: {methodology_points["quality_score"]:.3f}
- Self-Evolution Cycles: {methodology_points["evolution_cycles"]}

The summary should explain:
1. The TTD-DR methodology approach
2. Key research process steps
3. Quality assurance measures
4. Overall research rigor and reliability

Please write a professional, concise methodology summary:

METHODOLOGY SUMMARY:"""

        return prompt
    
    def _generate_fallback_methodology_summary(self, state: Dict[str, Any]) -> str:
        """Generate fallback methodology summary"""
        logger.info("Generating fallback methodology summary")
        
        methodology_points = self._extract_methodology_summary_points(state)
        
        fallback_summary = f"""## Methodology Summary

This research on "{methodology_points["topic"]}" was conducted using the TTD-DR (Test-Time Diffusion Deep Researcher) framework, an innovative approach that treats research report generation as a diffusion process. The methodology involved {methodology_points["iterations"]} iterations of refinement, starting with an initial research draft and progressively improving it through systematic gap analysis, information retrieval, and integration.

The research process identified {methodology_points["gaps_identified"]} information gaps and retrieved {methodology_points["sources_count"]} external sources to address these gaps. Quality assurance was maintained throughout the process, achieving a final quality score of {methodology_points["quality_score"]:.3f}. The framework also applied {methodology_points["evolution_cycles"]} self-evolution cycles to continuously improve the research components and strategies.

The TTD-DR methodology ensures research rigor through its iterative approach, comprehensive quality assessment, and adaptive learning mechanisms, providing a systematic and transparent approach to automated research report generation.
"""
        
        return fallback_summary

    # Research Methodology Documentation Methods (Task 9.2)
    
    async def generate_research_methodology_documentation(self, 
                                                        state: Dict[str, Any],
                                                        workflow_log: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate comprehensive research methodology documentation using Kimi K2
        
        Args:
            state: Complete workflow state containing all research data
            workflow_log: Optional workflow execution log
            
        Returns:
            Comprehensive methodology documentation
        """
        logger.info("Generating research methodology documentation with Kimi K2")
        
        try:
            # Extract methodology components from state
            methodology_data = self._extract_methodology_data(state, workflow_log)
            
            # Build methodology documentation prompt
            methodology_prompt = self._build_methodology_prompt(methodology_data)
            
            # Generate methodology documentation using Kimi K2
            response = await self.kimi_client.generate_text(
                methodology_prompt,
                max_tokens=4000,
                temperature=0.2  # Low temperature for factual documentation
            )
            
            methodology_doc = response.content
            
            # Enhance with structured sections
            enhanced_methodology = await self._enhance_methodology_structure(methodology_doc, methodology_data)
            
            logger.info(f"Research methodology documentation generated - Length: {len(enhanced_methodology)} characters")
            return enhanced_methodology
            
        except Exception as e:
            logger.error(f"Methodology documentation generation failed: {e}")
            return self._generate_fallback_methodology(state, workflow_log)
    
    def _extract_methodology_data(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract methodology data from workflow state and logs
        
        Args:
            state: Complete workflow state
            workflow_log: Optional workflow execution log
            
        Returns:
            Dictionary containing methodology data
        """
        methodology_data = {
            "research_topic": state.get("topic", "Unknown Topic"),
            "requirements": state.get("requirements"),
            "workflow_stages": [],
            "iteration_count": state.get("iteration_count", 0),
            "quality_metrics": state.get("quality_metrics"),
            "evolution_history": state.get("evolution_history", []),
            "sources_used": [],
            "search_queries": [],
            "information_gaps": state.get("information_gaps", []),
            "retrieved_info": state.get("retrieved_info", []),
            "workflow_execution_time": None,
            "methodology_approach": "TTD-DR (Test-Time Diffusion Deep Researcher)"
        }
        
        # Extract workflow stages from state
        if state.get("current_draft"):
            methodology_data["workflow_stages"].append({
                "stage": "Draft Generation",
                "description": "Initial research skeleton creation",
                "output": f"Generated {len(state['current_draft'].structure.sections)} main sections"
            })
        
        if state.get("information_gaps"):
            methodology_data["workflow_stages"].append({
                "stage": "Gap Analysis", 
                "description": "Identification of information gaps",
                "output": f"Identified {len(state['information_gaps'])} information gaps"
            })
        
        if state.get("retrieved_info"):
            methodology_data["workflow_stages"].append({
                "stage": "Information Retrieval",
                "description": "Dynamic retrieval of external information",
                "output": f"Retrieved {len(state['retrieved_info'])} information sources"
            })
            
            # Extract sources and queries
            for info in state["retrieved_info"]:
                if hasattr(info, 'source') and info.source:
                    methodology_data["sources_used"].append({
                        "url": getattr(info.source, 'url', 'Unknown URL'),
                        "title": getattr(info.source, 'title', 'Unknown Title'),
                        "credibility_score": getattr(info, 'credibility_score', 0.0),
                        "relevance_score": getattr(info, 'relevance_score', 0.0)
                    })
        
        if state.get("quality_metrics"):
            methodology_data["workflow_stages"].append({
                "stage": "Quality Assessment",
                "description": "Evaluation of research quality and completeness",
                "output": f"Overall quality score: {state['quality_metrics'].overall_score:.3f}"
            })
        
        if state.get("evolution_history"):
            methodology_data["workflow_stages"].append({
                "stage": "Self-Evolution Enhancement",
                "description": "Adaptive improvement of research components",
                "output": f"Applied {len(state['evolution_history'])} evolution cycles"
            })
        
        # Extract search queries from gaps
        for gap in state.get("information_gaps", []):
            if hasattr(gap, 'search_queries'):
                methodology_data["search_queries"].extend(gap.search_queries)
        
        # Extract workflow execution data from log if available
        if workflow_log:
            methodology_data["workflow_execution_time"] = self._calculate_execution_time(workflow_log)
            methodology_data["workflow_log"] = workflow_log[-10:]  # Last 10 log entries
        
        return methodology_data
    
    def _calculate_execution_time(self, workflow_log: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate total workflow execution time from log"""
        try:
            if not workflow_log:
                return None
            
            start_time = None
            end_time = None
            
            for entry in workflow_log:
                timestamp = entry.get("timestamp")
                if timestamp:
                    if start_time is None or timestamp < start_time:
                        start_time = timestamp
                    if end_time is None or timestamp > end_time:
                        end_time = timestamp
            
            if start_time and end_time:
                return end_time - start_time
            
        except Exception as e:
            logger.warning(f"Could not calculate execution time: {e}")
        
        return None
    
    def _build_methodology_prompt(self, methodology_data: Dict[str, Any]) -> str:
        """
        Build comprehensive methodology documentation prompt for Kimi K2
        
        Args:
            methodology_data: Extracted methodology data
            
        Returns:
            Formatted prompt for methodology documentation
        """
        # Format workflow stages
        stages_text = ""
        for i, stage in enumerate(methodology_data["workflow_stages"], 1):
            stages_text += f"{i}. **{stage['stage']}**: {stage['description']}\n   - {stage['output']}\n"
        
        # Format sources
        sources_text = ""
        if methodology_data["sources_used"]:
            for i, source in enumerate(methodology_data["sources_used"][:10], 1):  # Limit to top 10
                sources_text += f"{i}. {source['title']} ({source['url']})\n   - Credibility: {source['credibility_score']:.2f}, Relevance: {source['relevance_score']:.2f}\n"
        else:
            sources_text = "No external sources were retrieved during this research process."
        
        # Format search queries
        queries_text = ""
        if methodology_data["search_queries"]:
            unique_queries = list(set(methodology_data["search_queries"][:15]))  # Limit and deduplicate
            for i, query in enumerate(unique_queries, 1):
                queries_text += f"{i}. \"{query}\"\n"
        else:
            queries_text = "No search queries were generated during this research process."
        
        # Format quality metrics
        quality_text = ""
        if methodology_data["quality_metrics"]:
            metrics = methodology_data["quality_metrics"]
            quality_text = f"""
- Overall Quality Score: {metrics.overall_score:.3f}
- Completeness: {metrics.completeness:.3f}
- Coherence: {metrics.coherence:.3f}
- Accuracy: {metrics.accuracy:.3f}
- Citation Quality: {metrics.citation_quality:.3f}
"""
        else:
            quality_text = "Quality metrics were not available for this research process."
        
        # Format evolution summary
        evolution_text = ""
        if methodology_data["evolution_history"]:
            total_improvement = sum(
                r.performance_after - r.performance_before 
                for r in methodology_data["evolution_history"]
            )
            evolution_text = f"""
- Total Evolution Cycles: {len(methodology_data["evolution_history"])}
- Cumulative Performance Improvement: {total_improvement:+.3f}
- Components Enhanced: {len(set(r.component for r in methodology_data["evolution_history"]))}
"""
        else:
            evolution_text = "No self-evolution enhancements were applied during this research process."
        
        # Format execution time
        execution_text = ""
        if methodology_data["workflow_execution_time"]:
            execution_text = f"Total Execution Time: {methodology_data['workflow_execution_time']:.2f} seconds"
        else:
            execution_text = "Execution time data not available."
        
        prompt = f"""You are tasked with creating comprehensive research methodology documentation for a research report generated using the TTD-DR (Test-Time Diffusion Deep Researcher) framework. This documentation should provide complete transparency about the research process, methods used, and quality assurance measures applied.

RESEARCH OVERVIEW:
Topic: {methodology_data["research_topic"]}
Methodology Framework: {methodology_data["methodology_approach"]}
Total Iterations: {methodology_data["iteration_count"]}
{execution_text}

WORKFLOW STAGES EXECUTED:
{stages_text}

INFORMATION SOURCES UTILIZED:
{sources_text}

SEARCH QUERIES GENERATED:
{queries_text}

QUALITY ASSESSMENT RESULTS:
{quality_text}

SELF-EVOLUTION ENHANCEMENTS:
{evolution_text}

DOCUMENTATION REQUIREMENTS:

Please generate a comprehensive research methodology documentation that includes:

1. **Research Framework Overview**
   - Explanation of the TTD-DR methodology
   - Theoretical foundation and approach
   - Key principles and innovations

2. **Research Process Documentation**
   - Detailed description of each workflow stage
   - Decision points and criteria used
   - Iteration logic and convergence criteria

3. **Information Retrieval Methodology**
   - Search strategy and query formulation
   - Source selection and validation criteria
   - Information integration approaches

4. **Quality Assurance Framework**
   - Quality metrics and assessment criteria
   - Validation procedures and thresholds
   - Continuous improvement mechanisms

5. **Self-Evolution Enhancement Process**
   - Adaptive learning algorithms applied
   - Performance optimization strategies
   - Component enhancement methodologies

6. **Limitations and Considerations**
   - Methodological limitations
   - Potential biases and mitigation strategies
   - Scope and applicability constraints

7. **Reproducibility Guidelines**
   - Steps for reproducing the research
   - Parameter settings and configurations
   - Data requirements and dependencies

The documentation should be written in a professional, academic style suitable for peer review and should provide sufficient detail for methodology replication and validation.

RESEARCH METHODOLOGY DOCUMENTATION:"""

        return prompt
    
    async def _enhance_methodology_structure(self, methodology_doc: str, methodology_data: Dict[str, Any]) -> str:
        """
        Enhance methodology documentation with structured sections and formatting
        
        Args:
            methodology_doc: Raw methodology documentation from Kimi K2
            methodology_data: Original methodology data
            
        Returns:
            Enhanced and structured methodology documentation
        """
        try:
            enhancement_prompt = f"""Please enhance this research methodology documentation by adding proper academic structure, formatting, and additional technical details. Ensure it follows standard research methodology documentation format with:

1. Clear section headers and subsections
2. Professional academic writing style
3. Proper citation format for the TTD-DR framework
4. Technical specifications and parameters
5. Appendices for detailed data

Original Research Topic: {methodology_data["research_topic"]}
Framework: {methodology_data["methodology_approach"]}

Raw Methodology Documentation:
{methodology_doc}

Please return the enhanced, professionally formatted methodology documentation:"""

            response = await self.kimi_client.generate_text(
                enhancement_prompt,
                max_tokens=5000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            enhanced_doc = response.content
            
            # Add technical appendix
            technical_appendix = self._generate_technical_appendix(methodology_data)
            
            return f"{enhanced_doc}\n\n{technical_appendix}"
            
        except Exception as e:
            logger.warning(f"Methodology enhancement failed, using original: {e}")
            return self._apply_basic_methodology_formatting(methodology_doc, methodology_data)
    
    def _generate_technical_appendix(self, methodology_data: Dict[str, Any]) -> str:
        """
        Generate technical appendix with detailed parameters and data
        
        Args:
            methodology_data: Methodology data
            
        Returns:
            Technical appendix content
        """
        appendix = f"""
## Technical Appendix

### A. Framework Parameters

**TTD-DR Configuration:**
- Research Topic: {methodology_data["research_topic"]}
- Total Iterations: {methodology_data["iteration_count"]}
- Quality Threshold: {methodology_data.get("requirements", {}).get("quality_threshold", "Not specified") if methodology_data.get("requirements") else "Not specified"}

### B. Workflow Execution Details

**Stage Execution Summary:**
"""
        
        for i, stage in enumerate(methodology_data["workflow_stages"], 1):
            appendix += f"{i}. {stage['stage']}: {stage['output']}\n"
        
        appendix += f"""
### C. Information Sources Summary

**Total Sources Retrieved:** {len(methodology_data["sources_used"])}
**Search Queries Generated:** {len(methodology_data["search_queries"])}

**Source Quality Distribution:**
"""
        
        if methodology_data["sources_used"]:
            high_quality = sum(1 for s in methodology_data["sources_used"] if s["credibility_score"] >= 0.8)
            medium_quality = sum(1 for s in methodology_data["sources_used"] if 0.5 <= s["credibility_score"] < 0.8)
            low_quality = sum(1 for s in methodology_data["sources_used"] if s["credibility_score"] < 0.5)
            
            appendix += f"""- High Quality (≥0.8): {high_quality} sources
- Medium Quality (0.5-0.8): {medium_quality} sources  
- Low Quality (<0.5): {low_quality} sources
"""
        else:
            appendix += "- No external sources were utilized\n"
        
        appendix += f"""
### D. Quality Metrics Details

"""
        
        if methodology_data["quality_metrics"]:
            metrics = methodology_data["quality_metrics"]
            appendix += f"""**Final Quality Assessment:**
- Overall Score: {metrics.overall_score:.4f}
- Completeness: {metrics.completeness:.4f}
- Coherence: {metrics.coherence:.4f}
- Accuracy: {metrics.accuracy:.4f}
- Citation Quality: {metrics.citation_quality:.4f}
"""
        else:
            appendix += "Quality metrics were not available for this research execution.\n"
        
        appendix += f"""
### E. Self-Evolution Enhancement Log

"""
        
        if methodology_data["evolution_history"]:
            appendix += f"**Total Enhancement Cycles:** {len(methodology_data['evolution_history'])}\n\n"
            
            for i, record in enumerate(methodology_data["evolution_history"][-5:], 1):  # Last 5 records
                improvement = record.performance_after - record.performance_before
                appendix += f"{i}. Component: {record.component}\n"
                appendix += f"   - Enhancement Type: {record.improvement_type}\n"
                appendix += f"   - Performance Change: {improvement:+.4f}\n"
                appendix += f"   - Description: {record.description}\n\n"
        else:
            appendix += "No self-evolution enhancements were applied during this research process.\n"
        
        appendix += f"""
### F. Reproducibility Information

**Framework Version:** TTD-DR v1.0
**Execution Environment:** Python-based LangGraph workflow
**External APIs Used:** 
- Kimi K2 Language Model
- Google Search API (if applicable)

**Key Dependencies:**
- LangGraph for workflow orchestration
- Kimi K2 client for AI-powered processing
- Custom TTD-DR components for research automation

---

*This technical appendix provides detailed information for research methodology validation and reproducibility.*
"""
        
        return appendix
    
    def _apply_basic_methodology_formatting(self, methodology_doc: str, methodology_data: Dict[str, Any]) -> str:
        """Apply basic formatting as fallback for methodology documentation"""
        formatted_doc = f"""# Research Methodology Documentation

**Research Topic:** {methodology_data["research_topic"]}  
**Framework:** {methodology_data["methodology_approach"]}  
**Date Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Total Iterations:** {methodology_data["iteration_count"]}

---

{methodology_doc}

---

## Technical Summary

**Workflow Stages Completed:** {len(methodology_data["workflow_stages"])}  
**Information Sources Used:** {len(methodology_data["sources_used"])}  
**Search Queries Generated:** {len(methodology_data["search_queries"])}  
**Evolution Cycles Applied:** {len(methodology_data["evolution_history"])}

*Methodology documentation generated by TTD-DR Framework*
"""
        return formatted_doc
    
    def _generate_fallback_methodology(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> str:
        """Generate fallback methodology documentation when Kimi K2 fails"""
        logger.info("Generating fallback methodology documentation")
        
        topic = state.get("topic", "Unknown Topic")
        iteration_count = state.get("iteration_count", 0)
        
        # Basic workflow summary
        workflow_summary = []
        if state.get("current_draft"):
            workflow_summary.append("1. Initial draft generation completed")
        if state.get("information_gaps"):
            workflow_summary.append(f"2. Information gap analysis identified {len(state['information_gaps'])} gaps")
        if state.get("retrieved_info"):
            workflow_summary.append(f"3. Information retrieval gathered {len(state['retrieved_info'])} sources")
        if state.get("quality_metrics"):
            workflow_summary.append(f"4. Quality assessment achieved {state['quality_metrics'].overall_score:.3f} score")
        if state.get("evolution_history"):
            workflow_summary.append(f"5. Self-evolution applied {len(state['evolution_history'])} enhancements")
        
        fallback_methodology = f"""# Research Methodology Documentation

**Research Topic:** {topic}  
**Framework:** TTD-DR (Test-Time Diffusion Deep Researcher)  
**Date Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Generation Method:** Fallback Documentation

## Methodology Overview

This research was conducted using the TTD-DR framework, which implements a diffusion-inspired approach to automated research report generation. The framework treats research as an iterative refinement process, starting with a preliminary draft and progressively improving it through multiple cycles of analysis, information retrieval, and integration.

## Research Process

The research process completed {iteration_count} iterations and included the following stages:

{chr(10).join(workflow_summary) if workflow_summary else "No detailed workflow information available."}

## Framework Components

The TTD-DR framework consists of the following key components:

1. **Draft Generator**: Creates initial research skeleton using AI-powered topic analysis
2. **Gap Analyzer**: Identifies areas requiring additional information or improvement
3. **Retrieval Engine**: Dynamically retrieves relevant information from external sources
4. **Information Integrator**: Seamlessly incorporates new information into existing draft
5. **Quality Assessor**: Evaluates research quality and determines iteration needs
6. **Self-Evolution Enhancer**: Applies adaptive learning to improve component performance
7. **Report Synthesizer**: Generates final polished research report

## Quality Assurance

The research process included continuous quality assessment to ensure:
- Content completeness and accuracy
- Logical coherence and flow
- Proper source integration and citation
- Professional formatting and presentation

## Limitations

This methodology documentation was generated using fallback procedures due to technical limitations in the primary documentation system. The content may lack the depth and detail typically provided by the full TTD-DR methodology documentation process.

## Conclusion

The TTD-DR framework provides a systematic approach to automated research that combines the benefits of iterative refinement with AI-powered content generation and quality assessment. This methodology enables the production of comprehensive research reports while maintaining transparency and reproducibility.

---

*Methodology documentation generated by TTD-DR Framework v1.0*  
*Documentation Method: Fallback Generation*  
*Total Iterations: {iteration_count}*
"""
        
        return fallback_methodology

    async def generate_source_bibliography(self, retrieved_info: List[Any], citation_style: str = "APA") -> str:
        """
        Generate formatted bibliography from retrieved sources using Kimi K2
        
        Args:
            retrieved_info: List of retrieved information with sources
            citation_style: Citation style (APA, MLA, Chicago, etc.)
            
        Returns:
            Formatted bibliography
        """
        logger.info(f"Generating source bibliography in {citation_style} style with Kimi K2")
        
        if not retrieved_info:
            return "No sources were retrieved during the research process."
        
        try:
            # Extract unique sources
            unique_sources = self._extract_unique_sources(retrieved_info)
            
            # Build bibliography prompt
            bibliography_prompt = self._build_bibliography_prompt(unique_sources, citation_style)
            
            # Generate bibliography using Kimi K2
            response = await self.kimi_client.generate_text(
                bibliography_prompt,
                max_tokens=3000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            bibliography = response.content
            
            # Validate and enhance bibliography
            enhanced_bibliography = await self._enhance_bibliography_formatting(bibliography, citation_style)
            
            logger.info(f"Bibliography generated with {len(unique_sources)} sources")
            return enhanced_bibliography
            
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return self._generate_fallback_bibliography(retrieved_info, citation_style)
    
    def _extract_unique_sources(self, retrieved_info: List[Any]) -> List[Dict[str, Any]]:
        """Extract unique sources from retrieved information"""
        unique_sources = []
        seen_urls = set()
        
        for info in retrieved_info:
            if hasattr(info, 'source') and info.source:
                source = info.source
                url = getattr(source, 'url', '')
                
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append({
                        'title': getattr(source, 'title', 'Unknown Title'),
                        'url': url,
                        'author': getattr(source, 'author', ''),
                        'publication_date': getattr(source, 'publication_date', ''),
                        'domain': getattr(source, 'domain', ''),
                        'credibility_score': getattr(info, 'credibility_score', 0.0),
                        'access_date': getattr(info, 'extraction_timestamp', datetime.now()).strftime("%Y-%m-%d")
                    })
        
        # Sort by credibility score (highest first)
        unique_sources.sort(key=lambda x: x['credibility_score'], reverse=True)
        
        return unique_sources
    
    def _build_bibliography_prompt(self, sources: List[Dict[str, Any]], citation_style: str) -> str:
        """Build bibliography generation prompt for Kimi K2"""
        
        sources_text = ""
        for i, source in enumerate(sources, 1):
            sources_text += f"""
{i}. Title: {source['title']}
   URL: {source['url']}
   Author: {source['author'] or 'Unknown Author'}
   Publication Date: {source['publication_date'] or 'Unknown Date'}
   Domain: {source['domain']}
   Access Date: {source['access_date']}
   Credibility Score: {source['credibility_score']:.2f}
"""
        
        prompt = f"""You are tasked with creating a properly formatted bibliography from the following sources using {citation_style} citation style. Please generate accurate, consistent citations following standard {citation_style} formatting guidelines.

SOURCES TO CITE:
{sources_text}

FORMATTING REQUIREMENTS:

1. Use proper {citation_style} citation format
2. Include all available information (author, title, URL, access date, etc.)
3. Handle missing information appropriately (use "Unknown Author", "n.d." for no date, etc.)
4. Sort entries alphabetically by author/title as appropriate for {citation_style}
5. Use proper indentation and formatting
6. Include DOI or URL as required by {citation_style}

Please generate a complete bibliography section with proper heading and formatting:

BIBLIOGRAPHY:"""

        return prompt
    
    async def _enhance_bibliography_formatting(self, bibliography: str, citation_style: str) -> str:
        """Enhance bibliography formatting and validation"""
        try:
            enhancement_prompt = f"""Please review and enhance this bibliography to ensure it strictly follows {citation_style} formatting standards. Check for:

1. Proper {citation_style} citation format
2. Consistent formatting throughout
3. Correct punctuation and capitalization
4. Proper handling of web sources
5. Alphabetical ordering (if required by {citation_style})

Original Bibliography:
{bibliography}

Please return the corrected and enhanced bibliography:"""

            response = await self.kimi_client.generate_text(
                enhancement_prompt,
                max_tokens=3000,
                temperature=0.1
            )
            
            return response.content
            
        except Exception as e:
            logger.warning(f"Bibliography enhancement failed: {e}")
            return self._apply_basic_bibliography_formatting(bibliography, citation_style)
    
    def _apply_basic_bibliography_formatting(self, bibliography: str, citation_style: str) -> str:
        """Apply basic bibliography formatting as fallback"""
        formatted_bibliography = f"""# Bibliography ({citation_style} Style)

{bibliography}

---

*Bibliography generated using TTD-DR Framework*  
*Citation Style: {citation_style}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        return formatted_bibliography
    
    def _generate_fallback_bibliography(self, retrieved_info: List[Any], citation_style: str) -> str:
        """Generate basic fallback bibliography"""
        logger.info("Generating fallback bibliography")
        
        unique_sources = self._extract_unique_sources(retrieved_info)
        
        if not unique_sources:
            return f"""# Bibliography ({citation_style} Style)

No sources were retrieved during the research process.

---

*Bibliography generated using TTD-DR Framework (Fallback)*  
*Citation Style: {citation_style}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        
        bibliography_entries = []
        for i, source in enumerate(unique_sources, 1):
            # Basic citation format (simplified)
            author = source['author'] or "Unknown Author"
            title = source['title']
            url = source['url']
            access_date = source['access_date']
            
            if citation_style.upper() == "APA":
                entry = f"{author}. ({source['publication_date'] or 'n.d.'}). {title}. Retrieved {access_date}, from {url}"
            elif citation_style.upper() == "MLA":
                entry = f"{author}. \"{title}.\" Web. {access_date}. <{url}>."
            else:
                entry = f"{author}. \"{title}.\" Accessed {access_date}. {url}."
            
            bibliography_entries.append(f"{i}. {entry}")
        
        fallback_bibliography = f"""# Bibliography ({citation_style} Style)

{chr(10).join(bibliography_entries)}

---

*Bibliography generated using TTD-DR Framework (Fallback)*  
*Citation Style: {citation_style}*  
*Total Sources: {len(unique_sources)}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        
        return fallback_bibliography

    async def generate_methodology_summary(self, state: Dict[str, Any]) -> str:
        """
        Generate concise methodology summary using Kimi K2
        
        Args:
            state: Complete workflow state
            
        Returns:
            Concise methodology summary
        """
        logger.info("Generating methodology summary with Kimi K2")
        
        try:
            # Extract key methodology points
            methodology_points = self._extract_methodology_summary_points(state)
            
            # Build summary prompt
            summary_prompt = self._build_methodology_summary_prompt(methodology_points)
            
            # Generate summary using Kimi K2
            response = await self.kimi_client.generate_text(
                summary_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            methodology_summary = response.content
            
            logger.info(f"Methodology summary generated - Length: {len(methodology_summary)} characters")
            return methodology_summary
            
        except Exception as e:
            logger.error(f"Methodology summary generation failed: {e}")
            return self._generate_fallback_methodology_summary(state)
    
    def _extract_methodology_summary_points(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key points for methodology summary"""
        return {
            "topic": state.get("topic", "Unknown Topic"),
            "framework": "TTD-DR (Test-Time Diffusion Deep Researcher)",
            "iterations": state.get("iteration_count", 0),
            "sources_count": len(state.get("retrieved_info", [])),
            "gaps_identified": len(state.get("information_gaps", [])),
            "quality_score": state.get("quality_metrics", {}).get("overall_score", 0.0) if state.get("quality_metrics") else 0.0,
            "evolution_cycles": len(state.get("evolution_history", [])),
            "has_final_report": bool(state.get("final_report"))
        }
    
    def _build_methodology_summary_prompt(self, methodology_points: Dict[str, Any]) -> str:
        """Build methodology summary prompt for Kimi K2"""
        prompt = f"""Please create a concise methodology summary for a research report generated using the TTD-DR framework. The summary should be 2-3 paragraphs and suitable for inclusion in the main research report.

RESEARCH DETAILS:
- Topic: {methodology_points["topic"]}
- Framework: {methodology_points["framework"]}
- Iterations Completed: {methodology_points["iterations"]}
- External Sources Retrieved: {methodology_points["sources_count"]}
- Information Gaps Identified: {methodology_points["gaps_identified"]}
- Final Quality Score: {methodology_points["quality_score"]:.3f}
- Self-Evolution Cycles: {methodology_points["evolution_cycles"]}

The summary should explain:
1. The TTD-DR methodology approach
2. Key research process steps
3. Quality assurance measures
4. Overall research rigor and reliability

Please write a professional, concise methodology summary:

METHODOLOGY SUMMARY:"""

        return prompt
    
    def _generate_fallback_methodology_summary(self, state: Dict[str, Any]) -> str:
        """Generate fallback methodology summary"""
        logger.info("Generating fallback methodology summary")
        
        methodology_points = self._extract_methodology_summary_points(state)
        
        fallback_summary = f"""## Methodology Summary

This research on "{methodology_points["topic"]}" was conducted using the TTD-DR (Test-Time Diffusion Deep Researcher) framework, an innovative approach that treats research report generation as a diffusion process. The methodology involved {methodology_points["iterations"]} iterations of refinement, starting with an initial research draft and progressively improving it through systematic gap analysis, information retrieval, and integration.

The research process identified {methodology_points["gaps_identified"]} information gaps and retrieved {methodology_points["sources_count"]} external sources to address these gaps. Quality assurance was maintained throughout the process, achieving a final quality score of {methodology_points["quality_score"]:.3f}. The framework also applied {methodology_points["evolution_cycles"]} self-evolution cycles to continuously improve the research components and strategies.

The TTD-DR methodology ensures research rigor through its iterative approach, comprehensive quality assessment, and adaptive learning mechanisms, providing a systematic and transparent approach to automated research report generation.
"""
        
        return fallback_summary

    # Research Methodology Documentation Methods (Task 9.2)
    
    async def generate_research_methodology_documentation(self, 
                                                        state: Dict[str, Any],
                                                        workflow_log: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate comprehensive research methodology documentation using Kimi K2
        
        Args:
            state: Complete workflow state containing all research data
            workflow_log: Optional workflow execution log
            
        Returns:
            Comprehensive methodology documentation
        """
        logger.info("Generating research methodology documentation with Kimi K2")
        
        try:
            # Extract methodology components from state
            methodology_data = self._extract_methodology_data(state, workflow_log)
            
            # Build methodology documentation prompt
            methodology_prompt = self._build_methodology_prompt(methodology_data)
            
            # Generate methodology documentation using Kimi K2
            response = await self.kimi_client.generate_text(
                methodology_prompt,
                max_tokens=4000,
                temperature=0.2  # Low temperature for factual documentation
            )
            
            methodology_doc = response.content
            
            # Enhance with structured sections
            enhanced_methodology = await self._enhance_methodology_structure(methodology_doc, methodology_data)
            
            logger.info(f"Research methodology documentation generated - Length: {len(enhanced_methodology)} characters")
            return enhanced_methodology
            
        except Exception as e:
            logger.error(f"Methodology documentation generation failed: {e}")
            return self._generate_fallback_methodology(state, workflow_log)
    
    def _extract_methodology_data(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract methodology data from workflow state and logs
        
        Args:
            state: Complete workflow state
            workflow_log: Optional workflow execution log
            
        Returns:
            Dictionary containing methodology data
        """
        methodology_data = {
            "research_topic": state.get("topic", "Unknown Topic"),
            "requirements": state.get("requirements"),
            "workflow_stages": [],
            "iteration_count": state.get("iteration_count", 0),
            "quality_metrics": state.get("quality_metrics"),
            "evolution_history": state.get("evolution_history", []),
            "sources_used": [],
            "search_queries": [],
            "information_gaps": state.get("information_gaps", []),
            "retrieved_info": state.get("retrieved_info", []),
            "workflow_execution_time": None,
            "methodology_approach": "TTD-DR (Test-Time Diffusion Deep Researcher)"
        }
        
        # Extract workflow stages from state
        if state.get("current_draft"):
            methodology_data["workflow_stages"].append({
                "stage": "Draft Generation",
                "description": "Initial research skeleton creation",
                "output": f"Generated {len(state['current_draft'].structure.sections)} main sections"
            })
        
        if state.get("information_gaps"):
            methodology_data["workflow_stages"].append({
                "stage": "Gap Analysis", 
                "description": "Identification of information gaps",
                "output": f"Identified {len(state['information_gaps'])} information gaps"
            })
        
        if state.get("retrieved_info"):
            methodology_data["workflow_stages"].append({
                "stage": "Information Retrieval",
                "description": "Dynamic retrieval of external information",
                "output": f"Retrieved {len(state['retrieved_info'])} information sources"
            })
            
            # Extract sources and queries
            for info in state["retrieved_info"]:
                if hasattr(info, 'source') and info.source:
                    methodology_data["sources_used"].append({
                        "url": getattr(info.source, 'url', 'Unknown URL'),
                        "title": getattr(info.source, 'title', 'Unknown Title'),
                        "credibility_score": getattr(info, 'credibility_score', 0.0),
                        "relevance_score": getattr(info, 'relevance_score', 0.0)
                    })
        
        if state.get("quality_metrics"):
            methodology_data["workflow_stages"].append({
                "stage": "Quality Assessment",
                "description": "Evaluation of research quality and completeness",
                "output": f"Overall quality score: {state['quality_metrics'].overall_score:.3f}"
            })
        
        if state.get("evolution_history"):
            methodology_data["workflow_stages"].append({
                "stage": "Self-Evolution Enhancement",
                "description": "Adaptive improvement of research components",
                "output": f"Applied {len(state['evolution_history'])} evolution cycles"
            })
        
        # Extract search queries from gaps
        for gap in state.get("information_gaps", []):
            if hasattr(gap, 'search_queries'):
                methodology_data["search_queries"].extend(gap.search_queries)
        
        # Extract workflow execution data from log if available
        if workflow_log:
            methodology_data["workflow_execution_time"] = self._calculate_execution_time(workflow_log)
            methodology_data["workflow_log"] = workflow_log[-10:]  # Last 10 log entries
        
        return methodology_data
    
    def _calculate_execution_time(self, workflow_log: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate total workflow execution time from log"""
        try:
            if not workflow_log:
                return None
            
            start_time = None
            end_time = None
            
            for entry in workflow_log:
                timestamp = entry.get("timestamp")
                if timestamp:
                    if start_time is None or timestamp < start_time:
                        start_time = timestamp
                    if end_time is None or timestamp > end_time:
                        end_time = timestamp
            
            if start_time and end_time:
                return end_time - start_time
            
        except Exception as e:
            logger.warning(f"Could not calculate execution time: {e}")
        
        return None
    
    def _build_methodology_prompt(self, methodology_data: Dict[str, Any]) -> str:
        """
        Build comprehensive methodology documentation prompt for Kimi K2
        
        Args:
            methodology_data: Extracted methodology data
            
        Returns:
            Formatted prompt for methodology documentation
        """
        # Format workflow stages
        stages_text = ""
        for i, stage in enumerate(methodology_data["workflow_stages"], 1):
            stages_text += f"{i}. **{stage['stage']}**: {stage['description']}\n   - {stage['output']}\n"
        
        # Format sources
        sources_text = ""
        if methodology_data["sources_used"]:
            for i, source in enumerate(methodology_data["sources_used"][:10], 1):  # Limit to top 10
                sources_text += f"{i}. {source['title']} ({source['url']})\n   - Credibility: {source['credibility_score']:.2f}, Relevance: {source['relevance_score']:.2f}\n"
        else:
            sources_text = "No external sources were retrieved during this research process."
        
        # Format search queries
        queries_text = ""
        if methodology_data["search_queries"]:
            unique_queries = list(set(methodology_data["search_queries"][:15]))  # Limit and deduplicate
            for i, query in enumerate(unique_queries, 1):
                queries_text += f"{i}. \"{query}\"\n"
        else:
            queries_text = "No search queries were generated during this research process."
        
        # Format quality metrics
        quality_text = ""
        if methodology_data["quality_metrics"]:
            metrics = methodology_data["quality_metrics"]
            quality_text = f"""
- Overall Quality Score: {metrics.overall_score:.3f}
- Completeness: {metrics.completeness:.3f}
- Coherence: {metrics.coherence:.3f}
- Accuracy: {metrics.accuracy:.3f}
- Citation Quality: {metrics.citation_quality:.3f}
"""
        else:
            quality_text = "Quality metrics were not available for this research process."
        
        # Format evolution summary
        evolution_text = ""
        if methodology_data["evolution_history"]:
            total_improvement = sum(
                r.performance_after - r.performance_before 
                for r in methodology_data["evolution_history"]
            )
            evolution_text = f"""
- Total Evolution Cycles: {len(methodology_data["evolution_history"])}
- Cumulative Performance Improvement: {total_improvement:+.3f}
- Components Enhanced: {len(set(r.component for r in methodology_data["evolution_history"]))}
"""
        else:
            evolution_text = "No self-evolution enhancements were applied during this research process."
        
        # Format execution time
        execution_text = ""
        if methodology_data["workflow_execution_time"]:
            execution_text = f"Total Execution Time: {methodology_data['workflow_execution_time']:.2f} seconds"
        else:
            execution_text = "Execution time data not available."
        
        prompt = f"""You are tasked with creating comprehensive research methodology documentation for a research report generated using the TTD-DR (Test-Time Diffusion Deep Researcher) framework. This documentation should provide complete transparency about the research process, methods used, and quality assurance measures applied.

RESEARCH OVERVIEW:
Topic: {methodology_data["research_topic"]}
Methodology Framework: {methodology_data["methodology_approach"]}
Total Iterations: {methodology_data["iteration_count"]}
{execution_text}

WORKFLOW STAGES EXECUTED:
{stages_text}

INFORMATION SOURCES UTILIZED:
{sources_text}

SEARCH QUERIES GENERATED:
{queries_text}

QUALITY ASSESSMENT RESULTS:
{quality_text}

SELF-EVOLUTION ENHANCEMENTS:
{evolution_text}

DOCUMENTATION REQUIREMENTS:

Please generate a comprehensive research methodology documentation that includes:

1. **Research Framework Overview**
   - Explanation of the TTD-DR methodology
   - Theoretical foundation and approach
   - Key principles and innovations

2. **Research Process Documentation**
   - Detailed description of each workflow stage
   - Decision points and criteria used
   - Iteration logic and convergence criteria

3. **Information Retrieval Methodology**
   - Search strategy and query formulation
   - Source selection and validation criteria
   - Information integration approaches

4. **Quality Assurance Framework**
   - Quality metrics and assessment criteria
   - Validation procedures and thresholds
   - Continuous improvement mechanisms

5. **Self-Evolution Enhancement Process**
   - Adaptive learning algorithms applied
   - Performance optimization strategies
   - Component enhancement methodologies

6. **Limitations and Considerations**
   - Methodological limitations
   - Potential biases and mitigation strategies
   - Scope and applicability constraints

7. **Reproducibility Guidelines**
   - Steps for reproducing the research
   - Parameter settings and configurations
   - Data requirements and dependencies

The documentation should be written in a professional, academic style suitable for peer review and should provide sufficient detail for methodology replication and validation.

RESEARCH METHODOLOGY DOCUMENTATION:"""

        return prompt
    
    async def _enhance_methodology_structure(self, methodology_doc: str, methodology_data: Dict[str, Any]) -> str:
        """
        Enhance methodology documentation with structured sections and formatting
        
        Args:
            methodology_doc: Raw methodology documentation from Kimi K2
            methodology_data: Original methodology data
            
        Returns:
            Enhanced and structured methodology documentation
        """
        try:
            enhancement_prompt = f"""Please enhance this research methodology documentation by adding proper academic structure, formatting, and additional technical details. Ensure it follows standard research methodology documentation format with:

1. Clear section headers and subsections
2. Professional academic writing style
3. Proper citation format for the TTD-DR framework
4. Technical specifications and parameters
5. Appendices for detailed data

Original Research Topic: {methodology_data["research_topic"]}
Framework: {methodology_data["methodology_approach"]}

Raw Methodology Documentation:
{methodology_doc}

Please return the enhanced, professionally formatted methodology documentation:"""

            response = await self.kimi_client.generate_text(
                enhancement_prompt,
                max_tokens=5000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            enhanced_doc = response.content
            
            # Add technical appendix
            technical_appendix = self._generate_technical_appendix(methodology_data)
            
            return f"{enhanced_doc}\n\n{technical_appendix}"
            
        except Exception as e:
            logger.warning(f"Methodology enhancement failed, using original: {e}")
            return self._apply_basic_methodology_formatting(methodology_doc, methodology_data)
    
    def _generate_technical_appendix(self, methodology_data: Dict[str, Any]) -> str:
        """
        Generate technical appendix with detailed parameters and data
        
        Args:
            methodology_data: Methodology data
            
        Returns:
            Technical appendix content
        """
        appendix = f"""
## Technical Appendix

### A. Framework Parameters

**TTD-DR Configuration:**
- Research Topic: {methodology_data["research_topic"]}
- Total Iterations: {methodology_data["iteration_count"]}
- Quality Threshold: {methodology_data.get("requirements", {}).get("quality_threshold", "Not specified") if methodology_data.get("requirements") else "Not specified"}

### B. Workflow Execution Details

**Stage Execution Summary:**
"""
        
        for i, stage in enumerate(methodology_data["workflow_stages"], 1):
            appendix += f"{i}. {stage['stage']}: {stage['output']}\n"
        
        appendix += f"""
### C. Information Sources Summary

**Total Sources Retrieved:** {len(methodology_data["sources_used"])}
**Search Queries Generated:** {len(methodology_data["search_queries"])}

**Source Quality Distribution:**
"""
        
        if methodology_data["sources_used"]:
            high_quality = sum(1 for s in methodology_data["sources_used"] if s["credibility_score"] >= 0.8)
            medium_quality = sum(1 for s in methodology_data["sources_used"] if 0.5 <= s["credibility_score"] < 0.8)
            low_quality = sum(1 for s in methodology_data["sources_used"] if s["credibility_score"] < 0.5)
            
            appendix += f"""- High Quality (≥0.8): {high_quality} sources
- Medium Quality (0.5-0.8): {medium_quality} sources  
- Low Quality (<0.5): {low_quality} sources
"""
        else:
            appendix += "- No external sources were utilized\n"
        
        appendix += f"""
### D. Quality Metrics Details

"""
        
        if methodology_data["quality_metrics"]:
            metrics = methodology_data["quality_metrics"]
            appendix += f"""**Final Quality Assessment:**
- Overall Score: {metrics.overall_score:.4f}
- Completeness: {metrics.completeness:.4f}
- Coherence: {metrics.coherence:.4f}
- Accuracy: {metrics.accuracy:.4f}
- Citation Quality: {metrics.citation_quality:.4f}
"""
        else:
            appendix += "Quality metrics were not available for this research execution.\n"
        
        appendix += f"""
### E. Self-Evolution Enhancement Log

"""
        
        if methodology_data["evolution_history"]:
            appendix += f"**Total Enhancement Cycles:** {len(methodology_data['evolution_history'])}\n\n"
            
            for i, record in enumerate(methodology_data["evolution_history"][-5:], 1):  # Last 5 records
                improvement = record.performance_after - record.performance_before
                appendix += f"{i}. Component: {record.component}\n"
                appendix += f"   - Enhancement Type: {record.improvement_type}\n"
                appendix += f"   - Performance Change: {improvement:+.4f}\n"
                appendix += f"   - Description: {record.description}\n\n"
        else:
            appendix += "No self-evolution enhancements were applied during this research process.\n"
        
        appendix += f"""
### F. Reproducibility Information

**Framework Version:** TTD-DR v1.0
**Execution Environment:** Python-based LangGraph workflow
**External APIs Used:** 
- Kimi K2 Language Model
- Google Search API (if applicable)

**Key Dependencies:**
- LangGraph for workflow orchestration
- Kimi K2 client for AI-powered processing
- Custom TTD-DR components for research automation

---

*This technical appendix provides detailed information for research methodology validation and reproducibility.*
"""
        
        return appendix
    
    def _apply_basic_methodology_formatting(self, methodology_doc: str, methodology_data: Dict[str, Any]) -> str:
        """Apply basic formatting as fallback for methodology documentation"""
        formatted_doc = f"""# Research Methodology Documentation

**Research Topic:** {methodology_data["research_topic"]}  
**Framework:** {methodology_data["methodology_approach"]}  
**Date Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Total Iterations:** {methodology_data["iteration_count"]}

---

{methodology_doc}

---

## Technical Summary

**Workflow Stages Completed:** {len(methodology_data["workflow_stages"])}  
**Information Sources Used:** {len(methodology_data["sources_used"])}  
**Search Queries Generated:** {len(methodology_data["search_queries"])}  
**Evolution Cycles Applied:** {len(methodology_data["evolution_history"])}

*Methodology documentation generated by TTD-DR Framework*
"""
        return formatted_doc
    
    def _generate_fallback_methodology(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> str:
        """Generate fallback methodology documentation when Kimi K2 fails"""
        logger.info("Generating fallback methodology documentation")
        
        topic = state.get("topic", "Unknown Topic")
        iteration_count = state.get("iteration_count", 0)
        
        # Basic workflow summary
        workflow_summary = []
        if state.get("current_draft"):
            workflow_summary.append("1. Initial draft generation completed")
        if state.get("information_gaps"):
            workflow_summary.append(f"2. Information gap analysis identified {len(state['information_gaps'])} gaps")
        if state.get("retrieved_info"):
            workflow_summary.append(f"3. Information retrieval gathered {len(state['retrieved_info'])} sources")
        if state.get("quality_metrics"):
            workflow_summary.append(f"4. Quality assessment achieved {state['quality_metrics'].overall_score:.3f} score")
        if state.get("evolution_history"):
            workflow_summary.append(f"5. Self-evolution applied {len(state['evolution_history'])} enhancements")
        
        fallback_methodology = f"""# Research Methodology Documentation

**Research Topic:** {topic}  
**Framework:** TTD-DR (Test-Time Diffusion Deep Researcher)  
**Date Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Generation Method:** Fallback Documentation

## Methodology Overview

This research on "{topic}" was conducted using the TTD-DR framework, which implements a diffusion-inspired approach to automated research report generation. The framework treats research as an iterative refinement process, starting with a preliminary draft and progressively improving it through multiple cycles of analysis, information retrieval, and integration.

## Research Process

The research process completed {iteration_count} iterations and included the following stages:

{chr(10).join(workflow_summary) if workflow_summary else "No detailed workflow information available."}

## Framework Components

The TTD-DR framework consists of the following key components:

1. **Draft Generator**: Creates initial research skeleton using AI-powered topic analysis
2. **Gap Analyzer**: Identifies areas requiring additional information or improvement
3. **Retrieval Engine**: Dynamically retrieves relevant information from external sources
4. **Information Integrator**: Seamlessly incorporates new information into existing draft
5. **Quality Assessor**: Evaluates research quality and determines iteration needs
6. **Self-Evolution Enhancer**: Applies adaptive learning to improve component performance
7. **Report Synthesizer**: Generates final polished research report

## Quality Assurance

The research process included continuous quality assessment to ensure:
- Content completeness and accuracy
- Logical coherence and flow
- Proper source integration and citation
- Professional formatting and presentation

## Limitations

This methodology documentation was generated using fallback procedures due to technical limitations in the primary documentation system. The content may lack the depth and detail typically provided by the full TTD-DR methodology documentation process.

## Conclusion

The TTD-DR framework provides a systematic approach to automated research that combines the benefits of iterative refinement with AI-powered content generation and quality assessment. This methodology enables the production of comprehensive research reports while maintaining transparency and reproducibility.

---

*Methodology documentation generated by TTD-DR Framework v1.0*  
*Documentation Method: Fallback Generation*  
*Total Iterations: {iteration_count}*
"""
        
        return fallback_methodology

    async def generate_source_bibliography(self, retrieved_info: List[Any], citation_style: str = "APA") -> str:
        """
        Generate formatted bibliography from retrieved sources using Kimi K2
        
        Args:
            retrieved_info: List of retrieved information with sources
            citation_style: Citation style (APA, MLA, Chicago, etc.)
            
        Returns:
            Formatted bibliography
        """
        logger.info(f"Generating source bibliography in {citation_style} style with Kimi K2")
        
        if not retrieved_info:
            return "No sources were retrieved during the research process."
        
        try:
            # Extract unique sources
            unique_sources = self._extract_unique_sources(retrieved_info)
            
            # Build bibliography prompt
            bibliography_prompt = self._build_bibliography_prompt(unique_sources, citation_style)
            
            # Generate bibliography using Kimi K2
            response = await self.kimi_client.generate_text(
                bibliography_prompt,
                max_tokens=3000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            bibliography = response.content
            
            # Validate and enhance bibliography
            enhanced_bibliography = await self._enhance_bibliography_formatting(bibliography, citation_style)
            
            logger.info(f"Bibliography generated with {len(unique_sources)} sources")
            return enhanced_bibliography
            
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return self._generate_fallback_bibliography(retrieved_info, citation_style)
    
    def _extract_unique_sources(self, retrieved_info: List[Any]) -> List[Dict[str, Any]]:
        """Extract unique sources from retrieved information"""
        unique_sources = []
        seen_urls = set()
        
        for info in retrieved_info:
            if hasattr(info, 'source') and info.source:
                source = info.source
                url = getattr(source, 'url', '')
                
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append({
                        'title': getattr(source, 'title', 'Unknown Title'),
                        'url': url,
                        'author': getattr(source, 'author', ''),
                        'publication_date': getattr(source, 'publication_date', ''),
                        'domain': getattr(source, 'domain', ''),
                        'credibility_score': getattr(info, 'credibility_score', 0.0),
                        'access_date': getattr(info, 'extraction_timestamp', datetime.now()).strftime("%Y-%m-%d")
                    })
        
        # Sort by credibility score (highest first)
        unique_sources.sort(key=lambda x: x['credibility_score'], reverse=True)
        
        return unique_sources
    
    def _build_bibliography_prompt(self, sources: List[Dict[str, Any]], citation_style: str) -> str:
        """Build bibliography generation prompt for Kimi K2"""
        
        sources_text = ""
        for i, source in enumerate(sources, 1):
            sources_text += f"""
{i}. Title: {source['title']}
   URL: {source['url']}
   Author: {source['author'] or 'Unknown Author'}
   Publication Date: {source['publication_date'] or 'Unknown Date'}
   Domain: {source['domain']}
   Access Date: {source['access_date']}
   Credibility Score: {source['credibility_score']:.2f}
"""
        
        prompt = f"""You are tasked with creating a properly formatted bibliography from the following sources using {citation_style} citation style. Please generate accurate, consistent citations following standard {citation_style} formatting guidelines.

SOURCES TO CITE:
{sources_text}

FORMATTING REQUIREMENTS:

1. Use proper {citation_style} citation format
2. Include all available information (author, title, URL, access date, etc.)
3. Handle missing information appropriately (use "Unknown Author", "n.d." for no date, etc.)
4. Sort entries alphabetically by author/title as appropriate for {citation_style}
5. Use proper indentation and formatting
6. Include DOI or URL as required by {citation_style}

Please generate a complete bibliography section with proper heading and formatting:

BIBLIOGRAPHY:"""

        return prompt
    
    async def _enhance_bibliography_formatting(self, bibliography: str, citation_style: str) -> str:
        """Enhance bibliography formatting and validation"""
        try:
            enhancement_prompt = f"""Please review and enhance this bibliography to ensure it strictly follows {citation_style} formatting standards. Check for:

1. Proper {citation_style} citation format
2. Consistent formatting throughout
3. Correct punctuation and capitalization
4. Proper handling of web sources
5. Alphabetical ordering (if required by {citation_style})

Original Bibliography:
{bibliography}

Please return the corrected and enhanced bibliography:"""

            response = await self.kimi_client.generate_text(
                enhancement_prompt,
                max_tokens=3000,
                temperature=0.1
            )
            
            return response.content
            
        except Exception as e:
            logger.warning(f"Bibliography enhancement failed: {e}")
            return self._apply_basic_bibliography_formatting(bibliography, citation_style)
    
    def _apply_basic_bibliography_formatting(self, bibliography: str, citation_style: str) -> str:
        """Apply basic bibliography formatting as fallback"""
        formatted_bibliography = f"""# Bibliography ({citation_style} Style)

{bibliography}

---

*Bibliography generated using TTD-DR Framework*  
*Citation Style: {citation_style}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        return formatted_bibliography
    
    def _generate_fallback_bibliography(self, retrieved_info: List[Any], citation_style: str) -> str:
        """Generate basic fallback bibliography"""
        logger.info("Generating fallback bibliography")
        
        unique_sources = self._extract_unique_sources(retrieved_info)
        
        if not unique_sources:
            return f"""# Bibliography ({citation_style} Style)

No sources were retrieved during the research process.

---

*Bibliography generated using TTD-DR Framework (Fallback)*  
*Citation Style: {citation_style}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        
        bibliography_entries = []
        for i, source in enumerate(unique_sources, 1):
            # Basic citation format (simplified)
            author = source['author'] or "Unknown Author"
            title = source['title']
            url = source['url']
            access_date = source['access_date']
            
            if citation_style.upper() == "APA":
                entry = f"{author}. ({source['publication_date'] or 'n.d.'}). {title}. Retrieved {access_date}, from {url}"
            elif citation_style.upper() == "MLA":
                entry = f"{author}. \"{title}.\" Web. {access_date}. <{url}>."
            else:
                entry = f"{author}. \"{title}.\" Accessed {access_date}. {url}."
            
            bibliography_entries.append(f"{i}. {entry}")
        
        fallback_bibliography = f"""# Bibliography ({citation_style} Style)

{chr(10).join(bibliography_entries)}

---

*Bibliography generated using TTD-DR Framework (Fallback)*  
*Citation Style: {citation_style}*  
*Total Sources: {len(unique_sources)}*  
*Generated: {datetime.now().strftime("%B %d, %Y")}*
"""
        
        return fallback_bibliography

    async def generate_methodology_summary(self, state: Dict[str, Any]) -> str:
        """
        Generate concise methodology summary using Kimi K2
        
        Args:
            state: Complete workflow state
            
        Returns:
            Concise methodology summary
        """
        logger.info("Generating methodology summary with Kimi K2")
        
        try:
            # Extract key methodology points
            methodology_points = self._extract_methodology_summary_points(state)
            
            # Build summary prompt
            summary_prompt = self._build_methodology_summary_prompt(methodology_points)
            
            # Generate summary using Kimi K2
            response = await self.kimi_client.generate_text(
                summary_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            methodology_summary = response.content
            
            logger.info(f"Methodology summary generated - Length: {len(methodology_summary)} characters")
            return methodology_summary
            
        except Exception as e:
            logger.error(f"Methodology summary generation failed: {e}")
            return self._generate_fallback_methodology_summary(state)
    
    def _extract_methodology_summary_points(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key points for methodology summary"""
        return {
            "topic": state.get("topic", "Unknown Topic"),
            "framework": "TTD-DR (Test-Time Diffusion Deep Researcher)",
            "iterations": state.get("iteration_count", 0),
            "sources_count": len(state.get("retrieved_info", [])),
            "gaps_identified": len(state.get("information_gaps", [])),
            "quality_score": state.get("quality_metrics", {}).get("overall_score", 0.0) if state.get("quality_metrics") else 0.0,
            "evolution_cycles": len(state.get("evolution_history", [])),
            "has_final_report": bool(state.get("final_report"))
        }
    
    def _build_methodology_summary_prompt(self, methodology_points: Dict[str, Any]) -> str:
        """Build methodology summary prompt for Kimi K2"""
        prompt = f"""Please create a concise methodology summary for a research report generated using the TTD-DR framework. The summary should be 2-3 paragraphs and suitable for inclusion in the main research report.

RESEARCH DETAILS:
- Topic: {methodology_points["topic"]}
- Framework: {methodology_points["framework"]}
- Iterations Completed: {methodology_points["iterations"]}
- External Sources Retrieved: {methodology_points["sources_count"]}
- Information Gaps Identified: {methodology_points["gaps_identified"]}
- Final Quality Score: {methodology_points["quality_score"]:.3f}
- Self-Evolution Cycles: {methodology_points["evolution_cycles"]}

The summary should explain:
1. The TTD-DR methodology approach
2. Key research process steps
3. Quality assurance measures
4. Overall research rigor and reliability

Please write a professional, concise methodology summary:

METHODOLOGY SUMMARY:"""

        return prompt
    
    def _generate_fallback_methodology_summary(self, state: Dict[str, Any]) -> str:
        """Generate fallback methodology summary"""
        logger.info("Generating fallback methodology summary")
        
        methodology_points = self._extract_methodology_summary_points(state)
        
        fallback_summary = f"""## Methodology Summary

This research on "{methodology_points["topic"]}" was conducted using the TTD-DR (Test-Time Diffusion Deep Researcher) framework, an innovative approach that treats research report generation as a diffusion process. The methodology involved {methodology_points["iterations"]} iterations of refinement, starting with an initial research draft and progressively improving it through systematic gap analysis, information retrieval, and integration.

The research process identified {methodology_points["gaps_identified"]} information gaps and retrieved {methodology_points["sources_count"]} external sources to address these gaps. Quality assurance was maintained throughout the process, achieving a final quality score of {methodology_points["quality_score"]:.3f}. The framework also applied {methodology_points["evolution_cycles"]} self-evolution cycles to continuously improve the research components and strategies.

The TTD-DR methodology ensures research rigor through its iterative approach, comprehensive quality assessment, and adaptive learning mechanisms, providing a systematic and transparent approach to automated research report generation.
"""
        
        return fallback_summary  
  # Research Methodology Documentation Methods (Task 9.2)
    
    async def generate_research_methodology_documentation(self, 
                                                        state: Dict[str, Any],
                                                        workflow_log: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate comprehensive research methodology documentation using Kimi K2
        
        Args:
            state: Complete workflow state containing all research data
            workflow_log: Optional workflow execution log
            
        Returns:
            Comprehensive methodology documentation
        """
        logger.info("Generating research methodology documentation with Kimi K2")
        
        try:
            # Extract methodology components from state
            methodology_data = self._extract_methodology_data(state, workflow_log)
            
            # Build methodology documentation prompt
            methodology_prompt = self._build_methodology_prompt(methodology_data)
            
            # Generate methodology documentation using Kimi K2
            response = await self.kimi_client.generate_text(
                methodology_prompt,
                max_tokens=4000,
                temperature=0.2  # Low temperature for factual documentation
            )
            
            methodology_doc = response.content
            
            # Enhance with structured sections
            enhanced_methodology = await self._enhance_methodology_structure(methodology_doc, methodology_data)
            
            logger.info(f"Research methodology documentation generated - Length: {len(enhanced_methodology)} characters")
            return enhanced_methodology
            
        except Exception as e:
            logger.error(f"Methodology documentation generation failed: {e}")
            return self._generate_fallback_methodology(state, workflow_log)
    
    def _extract_methodology_data(self, state: Dict[str, Any], workflow_log: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract methodology data from workflow state and logs"""
        methodology_data = {
            "research_topic": state.get("topic", "Unknown Topic"),
            "requirements": state.get("requirements"),
            "workflow_stages": [],
            "iteration_count": state.get("iteration_count", 0),
            "quality_metrics": state.get("quality_metrics"),
            "evolution_history": state.get("evolution_history", []),
            "sources_used": [],
            "search_queries": [],
            "information_gaps": state.get("information_gaps", []),
            "retrieved_info": state.get("retrieved_info", []),
            "workflow_execution_time": None,
            "methodology_approach": "TTD-DR (Test-Time Diffusion Deep Researcher)"
        }
        
        # Extract workflow stages from state
        if state.get("current_draft"):
            methodology_data["workflow_stages"].append({
                "stage": "Draft Generation",
                "description": "Initial research skeleton creation",
                "output": f"Generated {len(state['current_draft'].structure.sections)} main sections"
            })
        
        if state.get("information_gaps"):
            methodology_data["workflow_stages"].append({
                "stage": "Gap Analysis", 
                "description": "Identification of information gaps",
                "output": f"Identified {len(state['information_gaps'])} information gaps"
            })
        
        if state.get("retrieved_info"):
            methodology_data["workflow_stages"].append({
                "stage": "Information Retrieval",
                "description": "Dynamic retrieval of external information",
                "output": f"Retrieved {len(state['retrieved_info'])} information sources"
            })
            
            # Extract sources and queries
            for info in state["retrieved_info"]:
                if hasattr(info, 'source') and info.source:
                    methodology_data["sources_used"].append({
                        "url": getattr(info.source, 'url', 'Unknown URL'),
                        "title": getattr(info.source, 'title', 'Unknown Title'),
                        "credibility_score": getattr(info, 'credibility_score', 0.0),
                        "relevance_score": getattr(info, 'relevance_score', 0.0)
                    })
        
        if state.get("quality_metrics"):
            methodology_data["workflow_stages"].append({
                "stage": "Quality Assessment",
                "description": "Evaluation of research quality and completeness",
                "output": f"Overall quality score: {state['quality_metrics'].overall_score:.3f}"
            })
        
        if state.get("evolution_history"):
            methodology_data["workflow_stages"].append({
                "stage": "Self-Evolution Enhancement",
                "description": "Adaptive improvement of research components",
                "output": f"Applied {len(state['evolution_history'])} evolution cycles"
            })
        
        # Extract search queries from gaps
        for gap in state.get("information_gaps", []):
            if hasattr(gap, 'search_queries'):
                methodology_data["search_queries"].extend(gap.search_queries)
        
        # Extract workflow execution data from log if available
        if workflow_log:
            methodology_data["workflow_execution_time"] = self._calculate_execution_time(workflow_log)
            methodology_data["workflow_log"] = workflow_log[-10:]  # Last 10 log entries
        
        return methodology_data
    
    def _calculate_execution_time(self, workflow_log: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate total workflow execution time from log"""
        try:
            if not workflow_log:
                return None
            
            start_time = None
            end_time = None
            
            for entry in workflow_log:
                timestamp = entry.get("timestamp")
                if timestamp:
                    if start_time is None or timestamp < start_time:
                        start_time = timestamp
                    if end_time is None or timestamp > end_time:
                        end_time = timestamp
            
            if start_time and end_time:
                return end_time - start_time
            
        except Exception as e:
            logger.warning(f"Could not calculate execution time: {e}")
        
        return None

    async def generate_source_bibliography(self, retrieved_info: List[Any], citation_style: str = "APA") -> str:
        """Generate formatted bibliography from retrieved sources using Kimi K2"""
        logger.info(f"Generating source bibliography in {citation_style} style with Kimi K2")
        
        if not retrieved_info:
            return "No sources were retrieved during the research process."
        
        try:
            # Extract unique sources
            unique_sources = self._extract_unique_sources(retrieved_info)
            
            # Build bibliography prompt
            bibliography_prompt = self._build_bibliography_prompt(unique_sources, citation_style)
            
            # Generate bibliography using Kimi K2
            response = await self.kimi_client.generate_text(
                bibliography_prompt,
                max_tokens=3000,
                temperature=0.1  # Very low temperature for consistent formatting
            )
            
            bibliography = response.content
            
            # Validate and enhance bibliography
            enhanced_bibliography = await self._enhance_bibliography_formatting(bibliography, citation_style)
            
            logger.info(f"Bibliography generated with {len(unique_sources)} sources")
            return enhanced_bibliography
            
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return self._generate_fallback_bibliography(retrieved_info, citation_style)
    
    def _extract_unique_sources(self, retrieved_info: List[Any]) -> List[Dict[str, Any]]:
        """Extract unique sources from retrieved information"""
        unique_sources = []
        seen_urls = set()
        
        for info in retrieved_info:
            if hasattr(info, 'source') and info.source:
                source = info.source
                url = getattr(source, 'url', '')
                
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append({
                        'title': getattr(source, 'title', 'Unknown Title'),
                        'url': url,
                        'author': getattr(source, 'author', ''),
                        'publication_date': getattr(source, 'publication_date', ''),
                        'domain': getattr(source, 'domain', ''),
                        'credibility_score': getattr(info, 'credibility_score', 0.0),
                        'access_date': getattr(info, 'extraction_timestamp', datetime.now()).strftime("%Y-%m-%d")
                    })
        
        # Sort by credibility score (highest first)
        unique_sources.sort(key=lambda x: x['credibility_score'], reverse=True)
        
        return unique_sources

    async def generate_methodology_summary(self, state: Dict[str, Any]) -> str:
        """Generate concise methodology summary using Kimi K2"""
        logger.info("Generating methodology summary with Kimi K2")
        
        try:
            # Extract key methodology points
            methodology_points = self._extract_methodology_summary_points(state)
            
            # Build summary prompt
            summary_prompt = self._build_methodology_summary_prompt(methodology_points)
            
            # Generate summary using Kimi K2
            response = await self.kimi_client.generate_text(
                summary_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            methodology_summary = response.content
            
            logger.info(f"Methodology summary generated - Length: {len(methodology_summary)} characters")
            return methodology_summary
            
        except Exception as e:
            logger.error(f"Methodology summary generation failed: {e}")
            return self._generate_fallback_methodology_summary(state)
    
    def _extract_methodology_summary_points(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key points for methodology summary"""
        return {
            "topic": state.get("topic", "Unknown Topic"),
            "framework": "TTD-DR (Test-Time Diffusion Deep Researcher)",
            "iterations": state.get("iteration_count", 0),
            "sources_count": len(state.get("retrieved_info", [])),
            "gaps_identified": len(state.get("information_gaps", [])),
            "quality_score": state.get("quality_metrics", {}).get("overall_score", 0.0) if state.get("quality_metrics") else 0.0,
            "evolution_cycles": len(state.get("evolution_history", [])),
            "has_final_report": bool(state.get("final_report"))
        }