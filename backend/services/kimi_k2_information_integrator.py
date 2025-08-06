"""
Kimi K2-powered information integration service for TTD-DR framework.
Handles intelligent content integration, contextual placement, and conflict resolution.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
import json
import uuid
from datetime import datetime

from models.core import (
    Draft, RetrievedInfo, InformationGap, Section, 
    ResearchStructure, DraftMetadata
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error

logger = logging.getLogger(__name__)

class IntegrationContext:
    """Context information for content integration"""
    
    def __init__(self, section: Section, surrounding_content: str, 
                 related_sections: List[Section], topic: str):
        self.section = section
        self.surrounding_content = surrounding_content
        self.related_sections = related_sections
        self.topic = topic

class ConflictResolution:
    """Represents a resolved content conflict"""
    
    def __init__(self, original_content: str, new_content: str, 
                 resolved_content: str, resolution_strategy: str):
        self.original_content = original_content
        self.new_content = new_content
        self.resolved_content = resolved_content
        self.resolution_strategy = resolution_strategy
        self.timestamp = datetime.now()

class KimiK2InformationIntegrator:
    """Kimi K2-powered information integration service"""
    
    def __init__(self):
        self.kimi_client = KimiK2Client()
        self.integration_history: List[Dict[str, Any]] = []
        
    async def integrate_information(self, draft: Draft, retrieved_info: List[RetrievedInfo], 
                                  gaps: List[InformationGap]) -> Draft:
        """
        Integrate retrieved information into the current draft using Kimi K2 intelligence
        
        Args:
            draft: Current research draft
            retrieved_info: List of retrieved information to integrate
            gaps: List of information gaps being addressed
            
        Returns:
            Updated draft with integrated information
        """
        logger.info(f"Starting information integration for {len(retrieved_info)} items")
        
        try:
            # Create a copy of the draft for modification
            updated_draft = self._copy_draft(draft)
            
            # Group retrieved info by gap/section
            info_by_gap = self._group_info_by_gap(retrieved_info, gaps)
            
            # Process each gap and its associated information
            for gap in gaps:
                if gap.id in info_by_gap:
                    gap_info = info_by_gap[gap.id]
                    logger.info(f"Integrating {len(gap_info)} items for gap: {gap.description[:50]}...")
                    
                    # Integrate information for this gap
                    updated_draft = await self._integrate_gap_information(
                        updated_draft, gap, gap_info
                    )
            
            # Perform final coherence check and cleanup
            updated_draft = await self._ensure_global_coherence(updated_draft)
            
            # Update metadata
            updated_draft.metadata.updated_at = datetime.now()
            updated_draft.metadata.word_count = self._calculate_word_count(updated_draft)
            
            logger.info("Information integration completed successfully")
            return updated_draft
            
        except Exception as e:
            logger.error(f"Information integration failed: {e}")
            raise
    
    async def _integrate_gap_information(self, draft: Draft, gap: InformationGap, 
                                       info_list: List[RetrievedInfo]) -> Draft:
        """Integrate information for a specific gap"""
        
        # Find the target section
        target_section = self._find_section_by_id(draft.structure.sections, gap.section_id)
        if not target_section:
            logger.warning(f"Target section {gap.section_id} not found")
            return draft
        
        # Get current content for the section
        current_content = draft.content.get(gap.section_id, "")
        
        # Create integration context
        context = self._create_integration_context(draft, target_section)
        
        # Integrate each piece of information
        integrated_content = current_content
        for info in info_list:
            integrated_content = await self._integrate_single_info(
                integrated_content, info, gap, context
            )
        
        # Update draft content
        draft.content[gap.section_id] = integrated_content
        
        return draft
    
    async def _integrate_single_info(self, current_content: str, info: RetrievedInfo,
                                   gap: InformationGap, context: IntegrationContext) -> str:
        """Integrate a single piece of retrieved information"""
        
        try:
            # Build integration prompt
            integration_prompt = self._build_integration_prompt(
                current_content, info, gap, context
            )
            
            # Get Kimi K2 integration suggestion
            response = await self.kimi_client.generate_text(integration_prompt)
            
            # Parse the response to extract integrated content
            integrated_content = self._parse_integration_response(response.content)
            
            # Record integration history
            self.integration_history.append({
                "timestamp": datetime.now(),
                "gap_id": gap.id,
                "section_id": gap.section_id,
                "source_url": info.source.url,
                "integration_strategy": "kimi_k2_contextual",
                "content_length_before": len(current_content),
                "content_length_after": len(integrated_content)
            })
            
            return integrated_content
            
        except KimiK2Error as e:
            logger.error(f"Kimi K2 integration failed: {e}")
            # Fallback to simple append
            return self._fallback_integration(current_content, info)
        except Exception as e:
            logger.error(f"Integration error: {e}")
            return current_content
    
    def _build_integration_prompt(self, current_content: str, info: RetrievedInfo,
                                gap: InformationGap, context: IntegrationContext) -> str:
        """Build prompt for Kimi K2 content integration"""
        
        return f"""You are an expert research assistant helping to integrate new information into an existing research draft. Your task is to seamlessly incorporate the retrieved information while maintaining coherence and flow.

**Research Topic:** {context.topic}

**Target Section:** {context.section.title}

**Information Gap Being Addressed:** {gap.description}

**Current Section Content:**
{current_content if current_content else "[Empty section - needs initial content]"}

**New Information to Integrate:**
Source: {info.source.title} ({info.source.url})
Relevance Score: {info.relevance_score:.2f}
Credibility Score: {info.credibility_score:.2f}

Content:
{info.content}

**Integration Instructions:**
1. Integrate the new information naturally into the existing content
2. Maintain logical flow and coherence
3. Avoid redundancy - don't repeat information already present
4. Use appropriate academic/research writing style
5. Include proper attribution to the source
6. If there are conflicts with existing content, resolve them intelligently
7. Ensure the integrated content directly addresses the information gap

**Surrounding Context:**
{context.surrounding_content[:500]}...

Please provide the complete updated section content with the new information properly integrated. Do not include any explanations or meta-commentary - just return the integrated content.
"""
    
    def _parse_integration_response(self, response_content: str) -> str:
        """Parse Kimi K2 response to extract integrated content"""
        
        # Clean up the response
        integrated_content = response_content.strip()
        
        # Remove any meta-commentary that might have been included
        lines = integrated_content.split('\n')
        content_lines = []
        
        for line in lines:
            # Skip lines that look like meta-commentary
            if (line.startswith('**') or 
                line.startswith('Note:') or 
                line.startswith('Explanation:') or
                line.startswith('Integration strategy:')):
                continue
            content_lines.append(line)
        
        return '\n'.join(content_lines).strip()
    
    def _fallback_integration(self, current_content: str, info: RetrievedInfo) -> str:
        """Fallback integration method when Kimi K2 fails"""
        
        if not current_content:
            return f"{info.content}\n\nSource: {info.source.title} ({info.source.url})"
        
        # Simple append with source attribution
        return f"{current_content}\n\n{info.content}\n\nSource: {info.source.title} ({info.source.url})"
    
    async def resolve_conflicts(self, existing_content: str, new_info: RetrievedInfo,
                              context: IntegrationContext) -> ConflictResolution:
        """Resolve conflicts between existing content and new information using Kimi K2"""
        
        try:
            conflict_prompt = self._build_conflict_resolution_prompt(
                existing_content, new_info, context
            )
            
            response = await self.kimi_client.generate_text(conflict_prompt)
            resolved_content = self._parse_integration_response(response.content)
            
            return ConflictResolution(
                original_content=existing_content,
                new_content=new_info.content,
                resolved_content=resolved_content,
                resolution_strategy="kimi_k2_intelligent"
            )
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            # Fallback to credibility-based resolution
            return self._fallback_conflict_resolution(existing_content, new_info)
    
    def _build_conflict_resolution_prompt(self, existing_content: str, 
                                        new_info: RetrievedInfo,
                                        context: IntegrationContext) -> str:
        """Build prompt for conflict resolution"""
        
        return f"""You are resolving a potential conflict between existing research content and new information. Your task is to create a coherent, accurate resolution that incorporates the best aspects of both sources.

**Research Topic:** {context.topic}
**Section:** {context.section.title}

**Existing Content:**
{existing_content}

**New Information:**
Source: {new_info.source.title} (Credibility: {new_info.credibility_score:.2f})
Content: {new_info.content}

**Resolution Instructions:**
1. Identify any factual conflicts or contradictions
2. Evaluate the credibility and recency of both sources
3. Create a balanced, accurate synthesis
4. Maintain academic objectivity
5. Include appropriate qualifications or caveats if needed
6. Ensure the resolution maintains the section's focus and flow

Please provide the resolved content that addresses the conflict appropriately.
"""
    
    def _fallback_conflict_resolution(self, existing_content: str, 
                                    new_info: RetrievedInfo) -> ConflictResolution:
        """Fallback conflict resolution based on credibility scores"""
        
        # Simple strategy: if new info has higher credibility, prefer it
        if new_info.credibility_score > 0.7:
            resolved_content = f"{new_info.content}\n\n{existing_content}"
            strategy = "credibility_based_new_preferred"
        else:
            resolved_content = f"{existing_content}\n\nAdditional perspective: {new_info.content}"
            strategy = "credibility_based_existing_preferred"
        
        return ConflictResolution(
            original_content=existing_content,
            new_content=new_info.content,
            resolved_content=resolved_content,
            resolution_strategy=strategy
        )
    
    async def _ensure_global_coherence(self, draft: Draft) -> Draft:
        """Ensure global coherence across all sections using Kimi K2"""
        
        try:
            # Build coherence check prompt
            coherence_prompt = self._build_coherence_prompt(draft)
            
            response = await self.kimi_client.generate_structured_response(
                coherence_prompt,
                {
                    "coherence_issues": ["string"],
                    "suggested_improvements": ["string"],
                    "overall_coherence_score": "number"
                }
            )
            
            # Apply suggested improvements if coherence is low
            if response.get("overall_coherence_score", 0) < 0.7:
                logger.info("Applying coherence improvements")
                draft = await self._apply_coherence_improvements(draft, response)
            
            return draft
            
        except Exception as e:
            logger.error(f"Coherence check failed: {e}")
            return draft
    
    def _build_coherence_prompt(self, draft: Draft) -> str:
        """Build prompt for coherence assessment"""
        
        full_content = ""
        for section in draft.structure.sections:
            content = draft.content.get(section.id, "")
            full_content += f"\n\n## {section.title}\n{content}"
        
        return f"""Analyze the coherence and flow of this research draft on "{draft.topic}".

**Full Draft Content:**
{full_content}

**Assessment Instructions:**
1. Identify any logical inconsistencies or flow issues
2. Check for smooth transitions between sections
3. Ensure consistent terminology and style
4. Identify redundant or contradictory information
5. Assess overall narrative coherence

Please provide your assessment in the requested JSON format.
"""
    
    async def _apply_coherence_improvements(self, draft: Draft, 
                                          coherence_analysis: Dict[str, Any]) -> Draft:
        """Apply coherence improvements based on Kimi K2 analysis"""
        
        # This is a simplified implementation
        # In practice, this would involve more sophisticated content restructuring
        
        improvements = coherence_analysis.get("suggested_improvements", [])
        
        for improvement in improvements[:3]:  # Limit to top 3 improvements
            logger.info(f"Applying improvement: {improvement}")
            # Implementation would depend on the specific improvement type
        
        return draft
    
    def _create_integration_context(self, draft: Draft, section: Section) -> IntegrationContext:
        """Create integration context for a section"""
        
        # Get surrounding content from adjacent sections
        surrounding_content = ""
        section_index = next((i for i, s in enumerate(draft.structure.sections) 
                            if s.id == section.id), -1)
        
        if section_index > 0:
            prev_section = draft.structure.sections[section_index - 1]
            prev_content = draft.content.get(prev_section.id, "")
            surrounding_content += f"Previous section ({prev_section.title}): {prev_content[-200:]}\n\n"
        
        if section_index < len(draft.structure.sections) - 1:
            next_section = draft.structure.sections[section_index + 1]
            next_content = draft.content.get(next_section.id, "")
            surrounding_content += f"Next section ({next_section.title}): {next_content[:200:]}"
        
        # Get related sections (simplified - could be more sophisticated)
        related_sections = [s for s in draft.structure.sections if s.id != section.id][:2]
        
        return IntegrationContext(
            section=section,
            surrounding_content=surrounding_content,
            related_sections=related_sections,
            topic=draft.topic
        )
    
    def _group_info_by_gap(self, retrieved_info: List[RetrievedInfo], 
                          gaps: List[InformationGap]) -> Dict[str, List[RetrievedInfo]]:
        """Group retrieved information by the gaps they address"""
        
        info_by_gap = {}
        
        for info in retrieved_info:
            gap_id = info.gap_id
            if gap_id:
                if gap_id not in info_by_gap:
                    info_by_gap[gap_id] = []
                info_by_gap[gap_id].append(info)
        
        return info_by_gap
    
    def _find_section_by_id(self, sections: List[Section], section_id: str) -> Optional[Section]:
        """Find a section by its ID"""
        
        for section in sections:
            if section.id == section_id:
                return section
            # Check subsections recursively
            if section.subsections:
                found = self._find_section_by_id(section.subsections, section_id)
                if found:
                    return found
        return None
    
    def _copy_draft(self, draft: Draft) -> Draft:
        """Create a deep copy of the draft for modification"""
        
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
            iteration=draft.iteration + 1
        )
    
    def _calculate_word_count(self, draft: Draft) -> int:
        """Calculate total word count for the draft"""
        
        total_words = 0
        for content in draft.content.values():
            total_words += len(content.split())
        return total_words