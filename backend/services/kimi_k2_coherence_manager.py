"""
Kimi K2-powered coherence maintenance and citation management service.
Handles logical flow consistency, citation tracking, and bibliography management.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any, Set
import json
import re
from datetime import datetime
from dataclasses import dataclass

from models.core import (
    Draft, RetrievedInfo, Section, Source
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Represents a citation in the document"""
    id: str
    source: Source
    page_number: Optional[str] = None
    quote: Optional[str] = None
    context: str = ""
    citation_style: str = "APA"
    
@dataclass
class CoherenceIssue:
    """Represents a coherence issue in the document"""
    section_id: str
    issue_type: str  # "logical_gap", "inconsistency", "poor_transition", "redundancy"
    description: str
    severity: str  # "low", "medium", "high", "critical"
    suggested_fix: str
    affected_text: str

@dataclass
class CoherenceReport:
    """Report on document coherence"""
    overall_score: float
    issues: List[CoherenceIssue]
    strengths: List[str]
    improvement_suggestions: List[str]
    section_scores: Dict[str, float]

class KimiK2CoherenceManager:
    """Kimi K2-powered coherence maintenance and citation management"""
    
    def __init__(self):
        self.kimi_client = KimiK2Client()
        self.citations: Dict[str, Citation] = {}
        self.coherence_history: List[Dict[str, Any]] = []
        
    async def maintain_coherence(self, draft: Draft) -> Tuple[Draft, CoherenceReport]:
        """
        Maintain coherence across the entire draft using Kimi K2 intelligence
        
        Args:
            draft: Current research draft
            
        Returns:
            Tuple of updated draft and coherence report
        """
        logger.info("Starting coherence maintenance analysis")
        
        try:
            # Analyze current coherence
            coherence_report = await self._analyze_coherence(draft)
            
            # Apply coherence improvements
            updated_draft = await self._apply_coherence_improvements(draft, coherence_report)
            
            # Re-analyze to verify improvements
            final_report = await self._analyze_coherence(updated_draft)
            
            # Record coherence maintenance history
            self.coherence_history.append({
                "timestamp": datetime.now(),
                "initial_score": coherence_report.overall_score,
                "final_score": final_report.overall_score,
                "issues_resolved": len(coherence_report.issues) - len(final_report.issues),
                "improvements_applied": len(coherence_report.improvement_suggestions)
            })
            
            logger.info(f"Coherence maintenance completed. Score: {coherence_report.overall_score:.2f} -> {final_report.overall_score:.2f}")
            
            return updated_draft, final_report
            
        except Exception as e:
            logger.error(f"Coherence maintenance failed: {e}")
            # Return original draft with empty report
            return draft, CoherenceReport(
                overall_score=0.5,
                issues=[],
                strengths=[],
                improvement_suggestions=[],
                section_scores={}
            )
    
    async def _analyze_coherence(self, draft: Draft) -> CoherenceReport:
        """Analyze document coherence using Kimi K2"""
        
        try:
            # Build coherence analysis prompt
            analysis_prompt = self._build_coherence_analysis_prompt(draft)
            
            # Get Kimi K2 analysis
            response = await self.kimi_client.generate_structured_response(
                analysis_prompt,
                {
                    "overall_score": "number",
                    "issues": [
                        {
                            "section_id": "string",
                            "issue_type": "string",
                            "description": "string",
                            "severity": "string",
                            "suggested_fix": "string",
                            "affected_text": "string"
                        }
                    ],
                    "strengths": ["string"],
                    "improvement_suggestions": ["string"],
                    "section_scores": {"string": "number"}
                }
            )
            
            # Parse response into CoherenceReport
            issues = [
                CoherenceIssue(
                    section_id=issue.get("section_id", ""),
                    issue_type=issue.get("issue_type", "unknown"),
                    description=issue.get("description", ""),
                    severity=issue.get("severity", "medium"),
                    suggested_fix=issue.get("suggested_fix", ""),
                    affected_text=issue.get("affected_text", "")
                )
                for issue in response.get("issues", [])
            ]
            
            return CoherenceReport(
                overall_score=response.get("overall_score", 0.5),
                issues=issues,
                strengths=response.get("strengths", []),
                improvement_suggestions=response.get("improvement_suggestions", []),
                section_scores=response.get("section_scores", {})
            )
            
        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            return self._fallback_coherence_analysis(draft)
    
    def _build_coherence_analysis_prompt(self, draft: Draft) -> str:
        """Build prompt for coherence analysis"""
        
        # Compile full document content
        full_content = f"# {draft.topic}\n\n"
        for section in draft.structure.sections:
            content = draft.content.get(section.id, "")
            full_content += f"## {section.title}\n{content}\n\n"
        
        return f"""Analyze the coherence and logical flow of this research document. Evaluate how well the sections connect, the consistency of arguments, and the overall narrative structure.

**Document Title:** {draft.topic}

**Full Document Content:**
{full_content}

**Analysis Instructions:**
1. Evaluate overall logical flow and coherence (score 0.0-1.0)
2. Identify specific coherence issues:
   - logical_gap: Missing logical connections
   - inconsistency: Contradictory statements
   - poor_transition: Abrupt section changes
   - redundancy: Repeated information
3. Note document strengths
4. Suggest specific improvements
5. Score each section individually (0.0-1.0)

**Evaluation Criteria:**
- Logical progression of ideas
- Smooth transitions between sections
- Consistency in terminology and concepts
- Absence of contradictions
- Clear argument structure
- Appropriate level of detail

Please provide your analysis in the requested JSON format with specific, actionable feedback.
"""
    
    def _fallback_coherence_analysis(self, draft: Draft) -> CoherenceReport:
        """Fallback coherence analysis when Kimi K2 fails"""
        
        issues = []
        section_scores = {}
        
        # Simple heuristic analysis
        for section in draft.structure.sections:
            content = draft.content.get(section.id, "")
            
            # Basic checks
            if len(content) < 50:
                issues.append(CoherenceIssue(
                    section_id=section.id,
                    issue_type="logical_gap",
                    description=f"Section '{section.title}' is too brief",
                    severity="medium",
                    suggested_fix="Add more detailed content",
                    affected_text=content[:100]
                ))
                section_scores[section.id] = 0.3
            else:
                section_scores[section.id] = 0.7
        
        overall_score = sum(section_scores.values()) / len(section_scores) if section_scores else 0.5
        
        return CoherenceReport(
            overall_score=overall_score,
            issues=issues,
            strengths=["Document has clear section structure"],
            improvement_suggestions=["Add more detailed content to brief sections"],
            section_scores=section_scores
        )
    
    async def _apply_coherence_improvements(self, draft: Draft, report: CoherenceReport) -> Draft:
        """Apply coherence improvements based on analysis"""
        
        if not report.issues:
            return draft
        
        updated_draft = self._copy_draft(draft)
        
        # Group issues by section
        issues_by_section = {}
        for issue in report.issues:
            if issue.section_id not in issues_by_section:
                issues_by_section[issue.section_id] = []
            issues_by_section[issue.section_id].append(issue)
        
        # Apply improvements section by section
        for section_id, section_issues in issues_by_section.items():
            if section_id in updated_draft.content:
                try:
                    improved_content = await self._improve_section_coherence(
                        updated_draft.content[section_id],
                        section_issues,
                        draft.topic
                    )
                    updated_draft.content[section_id] = improved_content
                except Exception as e:
                    logger.error(f"Failed to improve section {section_id}: {e}")
        
        return updated_draft
    
    async def _improve_section_coherence(self, content: str, issues: List[CoherenceIssue], topic: str) -> str:
        """Improve coherence of a specific section"""
        
        try:
            # Build improvement prompt
            improvement_prompt = self._build_improvement_prompt(content, issues, topic)
            
            # Get Kimi K2 improvement
            response = await self.kimi_client.generate_text(improvement_prompt)
            
            # Parse and clean the response
            improved_content = self._parse_improvement_response(response.content)
            
            return improved_content
            
        except Exception as e:
            logger.error(f"Section improvement failed: {e}")
            return content  # Return original content if improvement fails
    
    def _build_improvement_prompt(self, content: str, issues: List[CoherenceIssue], topic: str) -> str:
        """Build prompt for section improvement"""
        
        issues_description = "\n".join([
            f"- {issue.issue_type}: {issue.description} (Suggested fix: {issue.suggested_fix})"
            for issue in issues
        ])
        
        return f"""Improve the coherence and flow of this section from a research document on "{topic}".

**Current Content:**
{content}

**Identified Issues:**
{issues_description}

**Improvement Instructions:**
1. Address each identified issue specifically
2. Maintain the original meaning and key information
3. Improve logical flow and transitions
4. Ensure consistency in terminology
5. Remove redundancy while preserving important details
6. Add connecting phrases where needed

Please provide the improved section content. Do not include explanations or meta-commentary - just return the enhanced content.
"""
    
    def _parse_improvement_response(self, response_content: str) -> str:
        """Parse Kimi K2 improvement response"""
        
        # Clean up the response
        improved_content = response_content.strip()
        
        # Remove any meta-commentary
        lines = improved_content.split('\n')
        content_lines = []
        
        for line in lines:
            # Skip lines that look like meta-commentary
            if (line.startswith('**') or 
                line.startswith('Improvement:') or 
                line.startswith('Changes made:') or
                line.startswith('Note:')):
                continue
            content_lines.append(line)
        
        return '\n'.join(content_lines).strip()
    
    async def manage_citations(self, draft: Draft, retrieved_info: List[RetrievedInfo]) -> Tuple[Draft, List[Citation]]:
        """
        Manage citations and bibliography using Kimi K2
        
        Args:
            draft: Current research draft
            retrieved_info: List of retrieved information with sources
            
        Returns:
            Tuple of updated draft with proper citations and list of citations
        """
        logger.info(f"Managing citations for {len(retrieved_info)} sources")
        
        try:
            # Extract and organize citations
            citations = self._extract_citations(retrieved_info)
            
            # Apply proper citation formatting using Kimi K2
            updated_draft = await self._apply_citation_formatting(draft, citations)
            
            # Generate bibliography
            bibliography = await self._generate_bibliography(citations)
            
            # Add bibliography to draft
            updated_draft = self._add_bibliography_section(updated_draft, bibliography)
            
            logger.info(f"Citation management completed. {len(citations)} citations processed")
            
            return updated_draft, citations
            
        except Exception as e:
            logger.error(f"Citation management failed: {e}")
            return draft, []
    
    def _extract_citations(self, retrieved_info: List[RetrievedInfo]) -> List[Citation]:
        """Extract citations from retrieved information"""
        
        citations = []
        
        for i, info in enumerate(retrieved_info):
            citation_id = f"cite_{i+1}"
            
            citation = Citation(
                id=citation_id,
                source=info.source,
                context=info.content[:200] + "..." if len(info.content) > 200 else info.content,
                citation_style="APA"
            )
            
            citations.append(citation)
            self.citations[citation_id] = citation
        
        return citations
    
    async def _apply_citation_formatting(self, draft: Draft, citations: List[Citation]) -> Draft:
        """Apply proper citation formatting using Kimi K2"""
        
        updated_draft = self._copy_draft(draft)
        
        try:
            # Build citation formatting prompt
            formatting_prompt = self._build_citation_formatting_prompt(draft, citations)
            
            # Get Kimi K2 formatting suggestions
            response = await self.kimi_client.generate_structured_response(
                formatting_prompt,
                {
                    "formatted_sections": {
                        "string": "string"  # section_id -> formatted_content
                    },
                    "citation_style": "string",
                    "formatting_notes": ["string"]
                }
            )
            
            # Apply formatting suggestions
            formatted_sections = response.get("formatted_sections", {})
            for section_id, formatted_content in formatted_sections.items():
                if section_id in updated_draft.content:
                    updated_draft.content[section_id] = formatted_content
            
            return updated_draft
            
        except Exception as e:
            logger.error(f"Citation formatting failed: {e}")
            return self._fallback_citation_formatting(draft, citations)
    
    def _build_citation_formatting_prompt(self, draft: Draft, citations: List[Citation]) -> str:
        """Build prompt for citation formatting"""
        
        # Compile content with source information
        content_with_sources = ""
        for section in draft.structure.sections:
            content = draft.content.get(section.id, "")
            content_with_sources += f"## {section.title}\n{content}\n\n"
        
        # List available citations
        citations_list = "\n".join([
            f"[{cite.id}] {cite.source.title} - {cite.source.url}"
            for cite in citations
        ])
        
        return f"""Format this research document with proper in-text citations using APA style. Add appropriate citations where information from external sources is used.

**Document Content:**
{content_with_sources}

**Available Citations:**
{citations_list}

**Formatting Instructions:**
1. Add in-text citations in APA format (Author, Year) or (Source, Year)
2. Ensure every claim from external sources is properly cited
3. Use consistent citation style throughout
4. Place citations appropriately within sentences
5. Avoid over-citation while ensuring proper attribution

Please provide the formatted content for each section in the requested JSON format.
"""
    
    def _fallback_citation_formatting(self, draft: Draft, citations: List[Citation]) -> Draft:
        """Fallback citation formatting when Kimi K2 fails"""
        
        updated_draft = self._copy_draft(draft)
        
        # Simple citation insertion
        for section_id, content in updated_draft.content.items():
            # Find potential citation points (end of sentences with factual claims)
            sentences = content.split('.')
            formatted_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    # Simple heuristic: add citation to sentences that seem to contain facts
                    if any(word in sentence.lower() for word in ['research', 'study', 'found', 'shows', 'indicates']):
                        # Add first available citation
                        if citations:
                            sentence += f" ({citations[0].source.title.split()[0]}, {datetime.now().year})"
                    formatted_sentences.append(sentence)
            
            updated_draft.content[section_id] = '. '.join(formatted_sentences)
        
        return updated_draft
    
    async def _generate_bibliography(self, citations: List[Citation]) -> str:
        """Generate bibliography using Kimi K2"""
        
        try:
            # Build bibliography generation prompt
            bibliography_prompt = self._build_bibliography_prompt(citations)
            
            # Get Kimi K2 bibliography
            response = await self.kimi_client.generate_text(bibliography_prompt)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return self._fallback_bibliography(citations)
    
    def _build_bibliography_prompt(self, citations: List[Citation]) -> str:
        """Build prompt for bibliography generation"""
        
        sources_info = "\n".join([
            f"- Title: {cite.source.title}\n  URL: {cite.source.url}\n  Domain: {cite.source.domain}\n  Accessed: {cite.source.last_accessed.strftime('%Y-%m-%d')}"
            for cite in citations
        ])
        
        return f"""Generate a properly formatted bibliography in APA style for these sources:

**Sources:**
{sources_info}

**Instructions:**
1. Use proper APA format for web sources
2. Include all required elements (author if available, date, title, URL, access date)
3. Sort alphabetically by title if no author is available
4. Use consistent formatting throughout
5. Include "Retrieved from" for web sources

Please provide only the formatted bibliography entries, one per line.
"""
    
    def _fallback_bibliography(self, citations: List[Citation]) -> str:
        """Fallback bibliography generation"""
        
        bibliography_entries = []
        
        for citation in citations:
            # Simple APA-style format
            entry = f"{citation.source.title}. Retrieved from {citation.source.url}"
            bibliography_entries.append(entry)
        
        return "\n".join(sorted(bibliography_entries))
    
    def _add_bibliography_section(self, draft: Draft, bibliography: str) -> Draft:
        """Add bibliography section to the draft"""
        
        # Add bibliography as a new section
        bibliography_section = Section(
            id="bibliography",
            title="References",
            content=bibliography,
            estimated_length=len(bibliography)
        )
        
        # Add to structure if not already present
        if not any(section.id == "bibliography" for section in draft.structure.sections):
            draft.structure.sections.append(bibliography_section)
        
        # Add to content
        draft.content["bibliography"] = bibliography
        
        return draft
    
    async def resolve_citation_conflicts(self, conflicting_sources: List[RetrievedInfo]) -> str:
        """Resolve conflicts between contradictory sources using Kimi K2"""
        
        try:
            # Build conflict resolution prompt
            conflict_prompt = self._build_conflict_resolution_prompt(conflicting_sources)
            
            # Get Kimi K2 resolution
            response = await self.kimi_client.generate_text(conflict_prompt)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Citation conflict resolution failed: {e}")
            return self._fallback_conflict_resolution(conflicting_sources)
    
    def _build_conflict_resolution_prompt(self, conflicting_sources: List[RetrievedInfo]) -> str:
        """Build prompt for citation conflict resolution"""
        
        sources_info = "\n\n".join([
            f"**Source {i+1}:** {info.source.title} (Credibility: {info.credibility_score:.2f})\n{info.content}"
            for i, info in enumerate(conflicting_sources)
        ])
        
        return f"""Resolve the conflict between these contradictory sources by providing a balanced, objective synthesis.

**Conflicting Sources:**
{sources_info}

**Resolution Instructions:**
1. Identify the specific points of disagreement
2. Evaluate the credibility and evidence quality of each source
3. Provide a balanced synthesis that acknowledges different perspectives
4. Use appropriate qualifiers (e.g., "according to", "some research suggests")
5. Maintain academic objectivity
6. Include appropriate citations for each perspective

Please provide a coherent resolution that addresses the conflict appropriately.
"""
    
    def _fallback_conflict_resolution(self, conflicting_sources: List[RetrievedInfo]) -> str:
        """Fallback conflict resolution"""
        
        # Simple approach: present both perspectives
        resolution = "There are differing perspectives on this topic:\n\n"
        
        for i, source in enumerate(conflicting_sources):
            resolution += f"According to {source.source.title}, {source.content[:200]}...\n\n"
        
        resolution += "Further research may be needed to resolve these differences."
        
        return resolution
    
    def _copy_draft(self, draft: Draft) -> Draft:
        """Create a copy of the draft for modification"""
        
        from models.core import DraftMetadata
        
        return Draft(
            id=draft.id,
            topic=draft.topic,
            structure=draft.structure,
            content=draft.content.copy(),
            metadata=DraftMetadata(
                created_at=draft.metadata.created_at,
                updated_at=datetime.now(),
                author=draft.metadata.author,
                version=draft.metadata.version,
                word_count=draft.metadata.word_count
            ),
            quality_score=draft.quality_score,
            iteration=draft.iteration
        )
    
    def get_coherence_statistics(self) -> Dict[str, Any]:
        """Get statistics about coherence maintenance operations"""
        
        if not self.coherence_history:
            return {"total_operations": 0}
        
        scores_before = [entry["initial_score"] for entry in self.coherence_history]
        scores_after = [entry["final_score"] for entry in self.coherence_history]
        issues_resolved = [entry["issues_resolved"] for entry in self.coherence_history]
        
        return {
            "total_operations": len(self.coherence_history),
            "average_initial_score": sum(scores_before) / len(scores_before),
            "average_final_score": sum(scores_after) / len(scores_after),
            "average_improvement": (sum(scores_after) - sum(scores_before)) / len(scores_before),
            "total_issues_resolved": sum(issues_resolved),
            "total_citations_managed": len(self.citations)
        }