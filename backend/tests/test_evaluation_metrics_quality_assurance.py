"""
Evaluation metrics and quality assurance testing for TTD-DR framework.
Implements task 12.2: Factual accuracy validation, coherence assessment,
citation completeness validation, and evaluation metric reliability testing.

This module provides comprehensive quality assurance testing with advanced
evaluation metrics for the TTD-DR framework.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
import re
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, asdict
import uuid

from backend.models.core import (
    TTDRState, ResearchRequirements, ResearchDomain, ComplexityLevel,
    Draft, ResearchStructure, Section, QualityMetrics, Source,
    RetrievedInfo, Citation
)


@dataclass
class FactualAccuracyResult:
    """Factual accuracy validation result"""
    claim_id: str
    claim_text: str
    accuracy_score: float
    verification_sources: List[str]
    confidence_level: float
    fact_check_status: str  # 'verified', 'disputed', 'unverifiable'
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    timestamp: datetime


@dataclass
class CoherenceAssessmentResult:
    """Coherence assessment result"""
    section_id: str
    section_title: str
    coherence_score: float
    logical_flow_score: float
    transition_quality: float
    argument_consistency: float
    topic_relevance: float
    coherence_issues: List[str]
    improvement_suggestions: List[str]
    timestamp: datetime


@dataclass
class CitationValidationResult:
    """Citation validation result"""
    citation_id: str
    citation_text: str
    source_url: str
    source_type: str
    credibility_score: float
    accessibility_status: str  # 'accessible', 'broken', 'restricted'
    publication_date: Optional[datetime]
    author_credibility: float
    source_relevance: float
    citation_format_correct: bool
    validation_issues: List[str]
    timestamp: datetime


@dataclass
class ReadabilityAssessmentResult:
    """Readability assessment result"""
    text_id: str
    readability_score: float
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    complex_word_ratio: float
    readability_level: str  # 'elementary', 'middle_school', 'high_school', 'college', 'graduate'
    improvement_suggestions: List[str]
    timestamp: datetime


@dataclass
class QualityMetricsValidationResult:
    """Quality metrics validation result"""
    metric_name: str
    expected_range: Tuple[float, float]
    actual_value: float
    within_range: bool
    reliability_score: float
    consistency_score: float
    validation_method: str
    test_scenarios: List[str]
    accuracy_assessment: Dict[str, float]
    timestamp: datetime


class FactualAccuracyValidator:
    """Advanced factual accuracy validation system"""
    
    def __init__(self):
        self.fact_check_patterns = self._initialize_fact_check_patterns()
        self.verification_sources = self._initialize_verification_sources()
    
    def _initialize_fact_check_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for identifying factual claims"""
        return {
            'statistical_claims': [
                r'\d+(\.\d+)?%',  # Percentages
                r'\d+(\.\d+)?\s*(million|billion|thousand)',  # Large numbers
                r'increased by \d+',  # Growth claims
                r'decreased by \d+'   # Decline claims
            ],
            'temporal_claims': [
                r'in \d{4}',  # Year references
                r'since \d{4}',  # Since year
                r'by \d{4}',  # By year
                r'(last|past) \d+ (years?|months?|days?)'  # Time periods
            ],
            'comparative_claims': [
                r'(higher|lower|greater|less) than',
                r'(more|fewer) than \d+',
                r'(increased|decreased) from .* to',
                r'compared to'
            ],
            'definitive_claims': [
                r'(is|are) the (first|only|largest|smallest)',
                r'(always|never|all|none)',
                r'(proven|demonstrated|established) that'
            ]
        }
    
    def _initialize_verification_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize trusted verification sources"""
        return {
            'academic': {
                'domains': ['edu', 'org'],
                'credibility_weight': 0.9,
                'verification_methods': ['peer_review', 'citation_analysis']
            },
            'government': {
                'domains': ['gov', 'mil'],
                'credibility_weight': 0.85,
                'verification_methods': ['official_data', 'regulatory_compliance']
            },
            'industry': {
                'domains': ['com'],
                'credibility_weight': 0.7,
                'verification_methods': ['industry_standards', 'market_data']
            },
            'news': {
                'domains': ['com', 'org'],
                'credibility_weight': 0.6,
                'verification_methods': ['journalistic_standards', 'fact_checking']
            }
        }
    
    async def validate_factual_accuracy(self, text: str, sources: List[Source]) -> List[FactualAccuracyResult]:
        """
        Validate factual accuracy of claims in text
        
        Args:
            text: Text to validate
            sources: Available sources for verification
            
        Returns:
            List of factual accuracy results
        """
        results = []
        
        # Extract factual claims
        claims = self._extract_factual_claims(text)
        
        for claim in claims:
            # Verify each claim against sources
            accuracy_result = await self._verify_claim(claim, sources)
            results.append(accuracy_result)
        
        return results
    
    def _extract_factual_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual claims from text"""
        claims = []
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            claim_types = []
            
            # Check for different types of factual claims
            for claim_type, patterns in self.fact_check_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        claim_types.append(claim_type)
                        break
            
            if claim_types:
                claims.append({
                    'id': f"claim_{i}",
                    'text': sentence.strip(),
                    'types': claim_types,
                    'position': i
                })
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - in production, use more sophisticated NLP
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _verify_claim(self, claim: Dict[str, Any], sources: List[Source]) -> FactualAccuracyResult:
        """Verify a specific claim against sources"""
        
        # Simulate claim verification process
        verification_score = 0.0
        supporting_sources = []
        contradicting_sources = []
        confidence = 0.0
        
        # Check claim against each source
        for source in sources:
            source_relevance = self._calculate_source_relevance(claim['text'], source)
            source_credibility = self._calculate_source_credibility(source)
            
            if source_relevance > 0.5:
                # Simulate fact checking against source
                fact_check_result = self._simulate_fact_check(claim, source)
                
                if fact_check_result['supports']:
                    verification_score += source_credibility * source_relevance
                    supporting_sources.append(source.url)
                elif fact_check_result['contradicts']:
                    verification_score -= source_credibility * source_relevance * 0.5
                    contradicting_sources.append(source.url)
                
                confidence += source_credibility * 0.1
        
        # Normalize scores
        verification_score = max(0.0, min(1.0, verification_score))
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine fact check status
        if verification_score > 0.7 and len(supporting_sources) >= 2:
            status = 'verified'
        elif verification_score < 0.3 or len(contradicting_sources) > len(supporting_sources):
            status = 'disputed'
        else:
            status = 'unverifiable'
        
        return FactualAccuracyResult(
            claim_id=claim['id'],
            claim_text=claim['text'],
            accuracy_score=verification_score,
            verification_sources=supporting_sources + contradicting_sources,
            confidence_level=confidence,
            fact_check_status=status,
            supporting_evidence=supporting_sources,
            contradicting_evidence=contradicting_sources,
            timestamp=datetime.now()
        )
    
    def _calculate_source_relevance(self, claim_text: str, source: Source) -> float:
        """Calculate relevance of source to claim"""
        # Simplified relevance calculation
        claim_words = set(claim_text.lower().split())
        source_words = set((source.title + " " + source.description).lower().split())
        
        if not claim_words or not source_words:
            return 0.0
        
        intersection = claim_words.intersection(source_words)
        union = claim_words.union(source_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_source_credibility(self, source: Source) -> float:
        """Calculate credibility score for source"""
        # Simplified credibility calculation
        base_credibility = 0.5
        
        # Adjust based on source type
        if source.source_type in self.verification_sources:
            source_config = self.verification_sources[source.source_type]
            base_credibility = source_config['credibility_weight']
        
        # Adjust based on publication date (newer is generally better)
        if source.publication_date:
            days_old = (datetime.now() - source.publication_date).days
            if days_old < 365:  # Less than 1 year
                base_credibility *= 1.1
            elif days_old > 1825:  # More than 5 years
                base_credibility *= 0.9
        
        return max(0.0, min(1.0, base_credibility))
    
    def _simulate_fact_check(self, claim: Dict[str, Any], source: Source) -> Dict[str, bool]:
        """Simulate fact checking process"""
        # This is a simplified simulation
        # In reality, this would involve complex NLP and knowledge base queries
        
        # Simulate based on claim types and source credibility
        source_credibility = self._calculate_source_credibility(source)
        
        # Higher credibility sources are more likely to support claims
        support_probability = source_credibility * 0.8
        contradict_probability = (1 - source_credibility) * 0.3
        
        import random
        random.seed(hash(claim['text'] + source.url))  # Deterministic for testing
        
        rand_val = random.random()
        
        if rand_val < support_probability:
            return {'supports': True, 'contradicts': False}
        elif rand_val < support_probability + contradict_probability:
            return {'supports': False, 'contradicts': True}
        else:
            return {'supports': False, 'contradicts': False}


class CoherenceAssessmentEngine:
    """Advanced coherence and readability assessment engine"""
    
    def __init__(self):
        self.coherence_indicators = self._initialize_coherence_indicators()
        self.transition_words = self._initialize_transition_words()
    
    def _initialize_coherence_indicators(self) -> Dict[str, List[str]]:
        """Initialize coherence indicators"""
        return {
            'logical_connectors': [
                'therefore', 'thus', 'consequently', 'as a result',
                'because', 'since', 'due to', 'owing to',
                'however', 'nevertheless', 'nonetheless', 'although'
            ],
            'sequence_indicators': [
                'first', 'second', 'third', 'finally',
                'initially', 'subsequently', 'meanwhile',
                'before', 'after', 'during', 'while'
            ],
            'emphasis_markers': [
                'importantly', 'significantly', 'notably',
                'particularly', 'especially', 'specifically'
            ],
            'comparison_indicators': [
                'similarly', 'likewise', 'in contrast',
                'on the other hand', 'compared to', 'unlike'
            ]
        }
    
    def _initialize_transition_words(self) -> Set[str]:
        """Initialize transition words set"""
        all_transitions = set()
        for indicators in self.coherence_indicators.values():
            all_transitions.update(indicators)
        return all_transitions
    
    async def assess_coherence(self, draft: Draft) -> List[CoherenceAssessmentResult]:
        """
        Assess coherence of draft sections
        
        Args:
            draft: Draft to assess
            
        Returns:
            List of coherence assessment results
        """
        results = []
        
        for section in draft.structure.sections:
            section_content = draft.content.get(section.id, "")
            
            if section_content:
                coherence_result = await self._assess_section_coherence(
                    section, section_content
                )
                results.append(coherence_result)
        
        return results
    
    async def _assess_section_coherence(self, section: Section, content: str) -> CoherenceAssessmentResult:
        """Assess coherence of a specific section"""
        
        # Calculate various coherence metrics
        logical_flow_score = self._calculate_logical_flow(content)
        transition_quality = self._calculate_transition_quality(content)
        argument_consistency = self._calculate_argument_consistency(content)
        topic_relevance = self._calculate_topic_relevance(section, content)
        
        # Overall coherence score
        coherence_score = (
            logical_flow_score * 0.3 +
            transition_quality * 0.25 +
            argument_consistency * 0.25 +
            topic_relevance * 0.2
        )
        
        # Identify coherence issues
        issues = self._identify_coherence_issues(content, {
            'logical_flow': logical_flow_score,
            'transitions': transition_quality,
            'consistency': argument_consistency,
            'relevance': topic_relevance
        })
        
        # Generate improvement suggestions
        suggestions = self._generate_coherence_suggestions(issues)
        
        return CoherenceAssessmentResult(
            section_id=section.id,
            section_title=section.title,
            coherence_score=coherence_score,
            logical_flow_score=logical_flow_score,
            transition_quality=transition_quality,
            argument_consistency=argument_consistency,
            topic_relevance=topic_relevance,
            coherence_issues=issues,
            improvement_suggestions=suggestions,
            timestamp=datetime.now()
        )
    
    def _calculate_logical_flow(self, content: str) -> float:
        """Calculate logical flow score"""
        sentences = self._split_into_sentences(content)
        if len(sentences) < 2:
            return 1.0
        
        # Count logical connectors
        connector_count = 0
        for sentence in sentences:
            for connector in self.coherence_indicators['logical_connectors']:
                if connector.lower() in sentence.lower():
                    connector_count += 1
                    break
        
        # Calculate ratio of sentences with logical connectors
        connector_ratio = connector_count / len(sentences)
        
        # Optimal ratio is around 0.3-0.5
        if 0.3 <= connector_ratio <= 0.5:
            return 1.0
        elif connector_ratio < 0.3:
            return 0.5 + connector_ratio
        else:
            return max(0.5, 1.5 - connector_ratio)
    
    def _calculate_transition_quality(self, content: str) -> float:
        """Calculate transition quality score"""
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2:
            return 1.0
        
        transition_count = 0
        for paragraph in paragraphs[1:]:  # Skip first paragraph
            first_sentence = paragraph.split('.')[0].lower()
            
            # Check for transition words at paragraph beginnings
            for transition in self.transition_words:
                if transition in first_sentence:
                    transition_count += 1
                    break
        
        # Calculate transition ratio
        transition_ratio = transition_count / (len(paragraphs) - 1)
        
        # Good transition ratio is around 0.4-0.7
        if 0.4 <= transition_ratio <= 0.7:
            return 1.0
        elif transition_ratio < 0.4:
            return 0.3 + transition_ratio * 1.5
        else:
            return max(0.3, 1.3 - transition_ratio)
    
    def _calculate_argument_consistency(self, content: str) -> float:
        """Calculate argument consistency score"""
        # Simplified consistency check
        sentences = self._split_into_sentences(content)
        
        # Look for contradictory statements
        contradiction_indicators = [
            ('however', 'but'), ('although', 'despite'),
            ('while', 'whereas'), ('nevertheless', 'nonetheless')
        ]
        
        contradiction_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator_pair in contradiction_indicators:
                if any(indicator in sentence_lower for indicator in indicator_pair):
                    # Check if contradiction is properly resolved
                    if not self._is_contradiction_resolved(sentence):
                        contradiction_count += 1
        
        # Calculate consistency score
        if len(sentences) == 0:
            return 1.0
        
        contradiction_ratio = contradiction_count / len(sentences)
        return max(0.0, 1.0 - contradiction_ratio * 2)
    
    def _calculate_topic_relevance(self, section: Section, content: str) -> float:
        """Calculate topic relevance score"""
        # Extract key terms from section title and description
        section_terms = set((section.title + " " + section.description).lower().split())
        content_terms = set(content.lower().split())
        
        if not section_terms or not content_terms:
            return 0.5
        
        # Calculate term overlap
        overlap = section_terms.intersection(content_terms)
        relevance_score = len(overlap) / len(section_terms)
        
        return min(1.0, relevance_score * 2)  # Scale up to account for synonyms
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_contradiction_resolved(self, sentence: str) -> bool:
        """Check if a contradiction in a sentence is properly resolved"""
        # Simplified check for resolution indicators
        resolution_indicators = [
            'therefore', 'thus', 'consequently', 'as a result',
            'ultimately', 'in conclusion', 'overall'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in resolution_indicators)
    
    def _identify_coherence_issues(self, content: str, scores: Dict[str, float]) -> List[str]:
        """Identify specific coherence issues"""
        issues = []
        
        if scores['logical_flow'] < 0.6:
            issues.append("Insufficient logical connectors between ideas")
        
        if scores['transitions'] < 0.5:
            issues.append("Poor transitions between paragraphs")
        
        if scores['consistency'] < 0.7:
            issues.append("Potential contradictions or inconsistencies in arguments")
        
        if scores['relevance'] < 0.6:
            issues.append("Content may not be sufficiently relevant to section topic")
        
        return issues
    
    def _generate_coherence_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues"""
        suggestions = []
        
        for issue in issues:
            if "logical connectors" in issue:
                suggestions.append("Add more logical connectors (therefore, because, however) to link ideas")
            elif "transitions" in issue:
                suggestions.append("Improve paragraph transitions with connecting phrases")
            elif "contradictions" in issue:
                suggestions.append("Review arguments for consistency and resolve contradictions")
            elif "relevance" in issue:
                suggestions.append("Ensure content directly addresses the section topic")
        
        return suggestions


class CitationValidationSystem:
    """Comprehensive citation validation and source credibility system"""
    
    def __init__(self):
        self.citation_formats = self._initialize_citation_formats()
        self.credible_domains = self._initialize_credible_domains()
    
    def _initialize_citation_formats(self) -> Dict[str, str]:
        """Initialize citation format patterns"""
        return {
            'apa': r'\([A-Za-z]+,?\s+\d{4}\)',
            'mla': r'[A-Za-z]+\s+\d+',
            'chicago': r'[A-Za-z]+,?\s+"[^"]+",?\s+\d{4}',
            'url': r'https?://[^\s]+',
            'doi': r'doi:\s*10\.\d+/[^\s]+'
        }
    
    def _initialize_credible_domains(self) -> Dict[str, float]:
        """Initialize credible domain scores"""
        return {
            '.edu': 0.9,
            '.gov': 0.85,
            '.org': 0.7,
            'arxiv.org': 0.85,
            'pubmed.ncbi.nlm.nih.gov': 0.9,
            'scholar.google.com': 0.8,
            'jstor.org': 0.85,
            'springer.com': 0.8,
            'nature.com': 0.9,
            'science.org': 0.9
        }
    
    async def validate_citations(self, draft: Draft, sources: List[Source]) -> List[CitationValidationResult]:
        """
        Validate citations in draft
        
        Args:
            draft: Draft containing citations
            sources: Available sources
            
        Returns:
            List of citation validation results
        """
        results = []
        
        # Extract citations from draft content
        citations = self._extract_citations(draft)
        
        for citation in citations:
            # Find corresponding source
            source = self._find_corresponding_source(citation, sources)
            
            if source:
                validation_result = await self._validate_citation(citation, source)
                results.append(validation_result)
        
        return results
    
    def _extract_citations(self, draft: Draft) -> List[Dict[str, Any]]:
        """Extract citations from draft content"""
        citations = []
        citation_id = 0
        
        for section_id, content in draft.content.items():
            # Look for various citation formats
            for format_name, pattern in self.citation_formats.items():
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    citations.append({
                        'id': f"citation_{citation_id}",
                        'text': match.group(),
                        'format': format_name,
                        'section_id': section_id,
                        'position': match.start()
                    })
                    citation_id += 1
        
        return citations
    
    def _find_corresponding_source(self, citation: Dict[str, Any], sources: List[Source]) -> Optional[Source]:
        """Find source corresponding to citation"""
        citation_text = citation['text'].lower()
        
        # Try to match citation to source
        for source in sources:
            # Check URL match
            if citation['format'] == 'url' and source.url in citation_text:
                return source
            
            # Check author/title match for other formats
            if source.title.lower() in citation_text or any(
                author.lower() in citation_text for author in getattr(source, 'authors', [])
            ):
                return source
        
        return None
    
    async def _validate_citation(self, citation: Dict[str, Any], source: Source) -> CitationValidationResult:
        """Validate a specific citation"""
        
        # Check citation format correctness
        format_correct = self._check_citation_format(citation)
        
        # Calculate source credibility
        credibility_score = self._calculate_source_credibility(source)
        
        # Check source accessibility
        accessibility_status = await self._check_source_accessibility(source)
        
        # Calculate author credibility (simplified)
        author_credibility = self._calculate_author_credibility(source)
        
        # Calculate source relevance
        source_relevance = self._calculate_source_relevance(citation, source)
        
        # Identify validation issues
        issues = self._identify_citation_issues(
            citation, source, format_correct, credibility_score, accessibility_status
        )
        
        return CitationValidationResult(
            citation_id=citation['id'],
            citation_text=citation['text'],
            source_url=source.url,
            source_type=source.source_type,
            credibility_score=credibility_score,
            accessibility_status=accessibility_status,
            publication_date=source.publication_date,
            author_credibility=author_credibility,
            source_relevance=source_relevance,
            citation_format_correct=format_correct,
            validation_issues=issues,
            timestamp=datetime.now()
        )
    
    def _check_citation_format(self, citation: Dict[str, Any]) -> bool:
        """Check if citation format is correct"""
        # Simplified format checking
        format_name = citation['format']
        citation_text = citation['text']
        
        if format_name in self.citation_formats:
            pattern = self.citation_formats[format_name]
            return bool(re.match(pattern, citation_text))
        
        return False
    
    def _calculate_source_credibility(self, source: Source) -> float:
        """Calculate source credibility score"""
        base_score = 0.5
        
        # Check domain credibility
        for domain, score in self.credible_domains.items():
            if domain in source.url:
                base_score = max(base_score, score)
                break
        
        # Adjust based on source type
        type_adjustments = {
            'academic': 0.1,
            'government': 0.05,
            'industry': 0.0,
            'news': -0.05
        }
        
        if source.source_type in type_adjustments:
            base_score += type_adjustments[source.source_type]
        
        # Adjust based on publication date
        if source.publication_date:
            days_old = (datetime.now() - source.publication_date).days
            if days_old < 365:  # Recent
                base_score += 0.05
            elif days_old > 1825:  # Old
                base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    async def _check_source_accessibility(self, source: Source) -> str:
        """Check if source is accessible"""
        # Simplified accessibility check
        # In reality, this would make HTTP requests
        
        # Simulate based on URL patterns
        if 'doi.org' in source.url or 'arxiv.org' in source.url:
            return 'accessible'
        elif 'paywall' in source.url or 'subscription' in source.url:
            return 'restricted'
        elif 'broken' in source.url or '404' in source.url:
            return 'broken'
        else:
            return 'accessible'
    
    def _calculate_author_credibility(self, source: Source) -> float:
        """Calculate author credibility score"""
        # Simplified author credibility calculation
        base_score = 0.5
        
        # Check if source has author information
        if hasattr(source, 'authors') and source.authors:
            base_score += 0.2
        
        # Check institutional affiliation (simplified)
        if any(domain in source.url for domain in ['.edu', '.gov', 'research']):
            base_score += 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_source_relevance(self, citation: Dict[str, Any], source: Source) -> float:
        """Calculate source relevance to citation context"""
        # Simplified relevance calculation
        # In reality, this would analyze the surrounding text context
        
        base_relevance = 0.7  # Assume reasonable relevance
        
        # Adjust based on source type matching citation context
        if citation['format'] == 'url' and source.source_type == 'news':
            base_relevance += 0.1
        elif citation['format'] in ['apa', 'mla', 'chicago'] and source.source_type == 'academic':
            base_relevance += 0.2
        
        return max(0.0, min(1.0, base_relevance))
    
    def _identify_citation_issues(self, citation: Dict[str, Any], source: Source,
                                format_correct: bool, credibility_score: float,
                                accessibility_status: str) -> List[str]:
        """Identify citation validation issues"""
        issues = []
        
        if not format_correct:
            issues.append(f"Incorrect citation format for {citation['format']} style")
        
        if credibility_score < 0.5:
            issues.append("Low source credibility score")
        
        if accessibility_status == 'broken':
            issues.append("Source URL is not accessible")
        elif accessibility_status == 'restricted':
            issues.append("Source requires subscription or payment")
        
        if source.publication_date and (datetime.now() - source.publication_date).days > 1825:
            issues.append("Source is more than 5 years old")
        
        return issues


class QualityMetricsValidator:
    """Validator for quality metrics accuracy and reliability"""
    
    def __init__(self):
        self.metric_ranges = self._initialize_metric_ranges()
        self.test_scenarios = self._initialize_test_scenarios()
    
    def _initialize_metric_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialize expected ranges for quality metrics"""
        return {
            'overall_score': (0.0, 1.0),
            'completeness': (0.0, 1.0),
            'coherence': (0.0, 1.0),
            'accuracy': (0.0, 1.0),
            'citation_quality': (0.0, 1.0),
            'readability': (0.0, 1.0)
        }
    
    def _initialize_test_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize test scenarios for metric validation"""
        return [
            {
                'name': 'high_quality_text',
                'expected_scores': {
                    'overall_score': (0.8, 1.0),
                    'completeness': (0.85, 1.0),
                    'coherence': (0.8, 1.0),
                    'accuracy': (0.9, 1.0),
                    'citation_quality': (0.8, 1.0),
                    'readability': (0.7, 1.0)
                }
            },
            {
                'name': 'medium_quality_text',
                'expected_scores': {
                    'overall_score': (0.5, 0.8),
                    'completeness': (0.6, 0.8),
                    'coherence': (0.5, 0.8),
                    'accuracy': (0.6, 0.8),
                    'citation_quality': (0.5, 0.8),
                    'readability': (0.5, 0.8)
                }
            },
            {
                'name': 'low_quality_text',
                'expected_scores': {
                    'overall_score': (0.0, 0.5),
                    'completeness': (0.0, 0.5),
                    'coherence': (0.0, 0.5),
                    'accuracy': (0.0, 0.5),
                    'citation_quality': (0.0, 0.5),
                    'readability': (0.0, 0.5)
                }
            }
        ]
    
    async def validate_quality_metrics(self, quality_metrics: QualityMetrics,
                                     test_scenario: str) -> QualityMetricsValidationResult:
        """
        Validate quality metrics against expected ranges
        
        Args:
            quality_metrics: Quality metrics to validate
            test_scenario: Test scenario name
            
        Returns:
            Quality metrics validation result
        """
        
        # Find expected ranges for scenario
        scenario_config = next(
            (s for s in self.test_scenarios if s['name'] == test_scenario),
            None
        )
        
        if not scenario_config:
            raise ValueError(f"Unknown test scenario: {test_scenario}")
        
        validation_results = {}
        overall_accuracy = 0.0
        metrics_tested = 0
        
        # Validate each metric
        for metric_name, expected_range in scenario_config['expected_scores'].items():
            actual_value = getattr(quality_metrics, metric_name, None)
            
            if actual_value is not None:
                within_range = expected_range[0] <= actual_value <= expected_range[1]
                
                # Calculate accuracy (how close to expected range)
                if within_range:
                    accuracy = 1.0
                else:
                    # Calculate distance from range
                    if actual_value < expected_range[0]:
                        distance = expected_range[0] - actual_value
                    else:
                        distance = actual_value - expected_range[1]
                    
                    accuracy = max(0.0, 1.0 - distance)
                
                validation_results[metric_name] = {
                    'expected_range': expected_range,
                    'actual_value': actual_value,
                    'within_range': within_range,
                    'accuracy': accuracy
                }
                
                overall_accuracy += accuracy
                metrics_tested += 1
        
        # Calculate overall accuracy
        if metrics_tested > 0:
            overall_accuracy /= metrics_tested
        
        # Calculate reliability and consistency scores
        reliability_score = self._calculate_reliability_score(validation_results)
        consistency_score = self._calculate_consistency_score(validation_results)
        
        return QualityMetricsValidationResult(
            metric_name=f"quality_metrics_{test_scenario}",
            expected_range=(0.0, 1.0),  # Overall range
            actual_value=quality_metrics.overall_score,
            within_range=all(r['within_range'] for r in validation_results.values()),
            reliability_score=reliability_score,
            consistency_score=consistency_score,
            validation_method="range_validation",
            test_scenarios=[test_scenario],
            accuracy_assessment=validation_results,
            timestamp=datetime.now()
        )
    
    def _calculate_reliability_score(self, validation_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate reliability score based on validation results"""
        if not validation_results:
            return 0.0
        
        accuracy_scores = [r['accuracy'] for r in validation_results.values()]
        return statistics.mean(accuracy_scores)
    
    def _calculate_consistency_score(self, validation_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate consistency score based on validation results"""
        if not validation_results:
            return 0.0
        
        accuracy_scores = [r['accuracy'] for r in validation_results.values()]
        
        if len(accuracy_scores) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_accuracy = statistics.mean(accuracy_scores)
        std_accuracy = statistics.stdev(accuracy_scores)
        
        if mean_accuracy == 0:
            return 0.0
        
        cv = std_accuracy / mean_accuracy
        return max(0.0, 1.0 - cv)


class TestEvaluationMetricsQualityAssurance:
    """Test suite for evaluation metrics and quality assurance"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def factual_accuracy_validator(self):
        """Create factual accuracy validator"""
        return FactualAccuracyValidator()
    
    @pytest.fixture
    def coherence_assessment_engine(self):
        """Create coherence assessment engine"""
        return CoherenceAssessmentEngine()
    
    @pytest.fixture
    def citation_validation_system(self):
        """Create citation validation system"""
        return CitationValidationSystem()
    
    @pytest.fixture
    def quality_metrics_validator(self):
        """Create quality metrics validator"""
        return QualityMetricsValidator()
    
    @pytest.fixture
    def sample_draft(self):
        """Create sample draft for testing"""
        from backend.models.core import Draft, ResearchStructure, Section, DraftMetadata
        
        sections = [
            Section(
                id="introduction",
                title="Introduction",
                description="Introduction to the topic",
                estimated_length=500
            ),
            Section(
                id="analysis",
                title="Analysis",
                description="Detailed analysis",
                estimated_length=1000
            )
        ]
        
        structure = ResearchStructure(
            sections=sections,
            total_estimated_length=1500,
            domain=ResearchDomain.TECHNOLOGY
        )
        
        content = {
            "introduction": "This research examines AI applications in healthcare. According to recent studies, AI has improved diagnostic accuracy by 25% since 2020. However, implementation challenges remain significant.",
            "analysis": "The analysis shows that machine learning algorithms demonstrate superior performance. Therefore, healthcare institutions should consider adoption. Nevertheless, training requirements are substantial."
        }
        
        return Draft(
            id="test_draft",
            topic="AI in Healthcare",
            structure=structure,
            content=content,
            metadata=DraftMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                version=1
            ),
            quality_score=0.75,
            iteration=1
        )
    
    @pytest.fixture
    def sample_sources(self):
        """Create sample sources for testing"""
        return [
            Source(
                url="https://example.edu/ai-healthcare-study",
                title="AI Applications in Healthcare Diagnostics",
                description="Comprehensive study on AI diagnostic accuracy",
                source_type="academic",
                publication_date=datetime(2023, 1, 15),
                credibility_score=0.9
            ),
            Source(
                url="https://healthtech.gov/ai-implementation",
                title="Government Report on AI Implementation",
                description="Official report on AI adoption in healthcare",
                source_type="government",
                publication_date=datetime(2023, 6, 1),
                credibility_score=0.85
            )
        ]
    
    @pytest.mark.asyncio
    async def test_factual_accuracy_validation(self, factual_accuracy_validator, sample_sources):
        """Test factual accuracy validation system"""
        
        print(f"\n🔍 FACTUAL ACCURACY VALIDATION TESTING")
        print("=" * 50)
        
        test_text = """
        AI has improved diagnostic accuracy by 25% since 2020.
        The healthcare industry invested $2.1 billion in AI technologies in 2023.
        Machine learning algorithms are 100% accurate in all medical diagnoses.
        Studies show that 85% of hospitals have adopted AI systems.
        """
        
        # Validate factual accuracy
        accuracy_results = await factual_accuracy_validator.validate_factual_accuracy(
            test_text, sample_sources
        )
        
        print(f"📊 Analyzed {len(accuracy_results)} factual claims")
        
        verified_claims = 0
        disputed_claims = 0
        unverifiable_claims = 0
        
        for result in accuracy_results:
            print(f"\n📝 Claim: {result.claim_text[:50]}...")
            print(f"   🎯 Accuracy Score: {result.accuracy_score:.2f}")
            print(f"   📋 Status: {result.fact_check_status}")
            print(f"   🔗 Supporting Sources: {len(result.supporting_evidence)}")
            print(f"   ⚠️  Contradicting Sources: {len(result.contradicting_evidence)}")
            
            if result.fact_check_status == 'verified':
                verified_claims += 1
            elif result.fact_check_status == 'disputed':
                disputed_claims += 1
            else:
                unverifiable_claims += 1
            
            # Assertions
            assert 0.0 <= result.accuracy_score <= 1.0, "Accuracy score must be between 0 and 1"
            assert result.fact_check_status in ['verified', 'disputed', 'unverifiable'], \
                "Invalid fact check status"
            assert 0.0 <= result.confidence_level <= 1.0, "Confidence level must be between 0 and 1"
        
        print(f"\n✅ Verified: {verified_claims}")
        print(f"❌ Disputed: {disputed_claims}")
        print(f"❓ Unverifiable: {unverifiable_claims}")
        
        # Verify that the system can identify different types of claims
        assert len(accuracy_results) > 0, "Should identify factual claims"
        assert verified_claims + disputed_claims + unverifiable_claims == len(accuracy_results), \
            "All claims should be categorized"
    
    @pytest.mark.asyncio
    async def test_coherence_assessment(self, coherence_assessment_engine, sample_draft):
        """Test coherence assessment system"""
        
        print(f"\n🧠 COHERENCE ASSESSMENT TESTING")
        print("=" * 40)
        
        # Assess coherence
        coherence_results = await coherence_assessment_engine.assess_coherence(sample_draft)
        
        print(f"📊 Assessed {len(coherence_results)} sections")
        
        for result in coherence_results:
            print(f"\n📝 Section: {result.section_title}")
            print(f"   🎯 Overall Coherence: {result.coherence_score:.2f}")
            print(f"   🔄 Logical Flow: {result.logical_flow_score:.2f}")
            print(f"   🔗 Transition Quality: {result.transition_quality:.2f}")
            print(f"   ⚖️  Argument Consistency: {result.argument_consistency:.2f}")
            print(f"   🎯 Topic Relevance: {result.topic_relevance:.2f}")
            print(f"   ⚠️  Issues: {len(result.coherence_issues)}")
            print(f"   💡 Suggestions: {len(result.improvement_suggestions)}")
            
            # Assertions
            assert 0.0 <= result.coherence_score <= 1.0, "Coherence score must be between 0 and 1"
            assert 0.0 <= result.logical_flow_score <= 1.0, "Logical flow score must be between 0 and 1"
            assert 0.0 <= result.transition_quality <= 1.0, "Transition quality must be between 0 and 1"
            assert 0.0 <= result.argument_consistency <= 1.0, "Argument consistency must be between 0 and 1"
            assert 0.0 <= result.topic_relevance <= 1.0, "Topic relevance must be between 0 and 1"
            
            # Check that issues and suggestions are provided when scores are low
            if result.coherence_score < 0.7:
                assert len(result.coherence_issues) > 0, "Should identify issues for low coherence scores"
                assert len(result.improvement_suggestions) > 0, "Should provide suggestions for improvement"
        
        print(f"\n✅ Coherence assessment completed successfully")
    
    @pytest.mark.asyncio
    async def test_citation_validation(self, citation_validation_system, sample_draft, sample_sources):
        """Test citation validation system"""
        
        print(f"\n📚 CITATION VALIDATION TESTING")
        print("=" * 35)
        
        # Add some citations to the draft content
        sample_draft.content["introduction"] += " (Smith, 2023). See https://example.edu/ai-healthcare-study for details."
        sample_draft.content["analysis"] += " According to Johnson et al. (2022), implementation is challenging."
        
        # Validate citations
        citation_results = await citation_validation_system.validate_citations(
            sample_draft, sample_sources
        )
        
        print(f"📊 Validated {len(citation_results)} citations")
        
        accessible_citations = 0
        credible_citations = 0
        format_correct_citations = 0
        
        for result in citation_results:
            print(f"\n📝 Citation: {result.citation_text}")
            print(f"   🔗 Source URL: {result.source_url}")
            print(f"   🏆 Credibility: {result.credibility_score:.2f}")
            print(f"   🌐 Accessibility: {result.accessibility_status}")
            print(f"   ✅ Format Correct: {result.citation_format_correct}")
            print(f"   🎯 Relevance: {result.source_relevance:.2f}")
            print(f"   ⚠️  Issues: {len(result.validation_issues)}")
            
            if result.accessibility_status == 'accessible':
                accessible_citations += 1
            if result.credibility_score >= 0.7:
                credible_citations += 1
            if result.citation_format_correct:
                format_correct_citations += 1
            
            # Assertions
            assert 0.0 <= result.credibility_score <= 1.0, "Credibility score must be between 0 and 1"
            assert result.accessibility_status in ['accessible', 'broken', 'restricted'], \
                "Invalid accessibility status"
            assert 0.0 <= result.author_credibility <= 1.0, "Author credibility must be between 0 and 1"
            assert 0.0 <= result.source_relevance <= 1.0, "Source relevance must be between 0 and 1"
        
        print(f"\n✅ Accessible: {accessible_citations}/{len(citation_results)}")
        print(f"🏆 Credible: {credible_citations}/{len(citation_results)}")
        print(f"📝 Format Correct: {format_correct_citations}/{len(citation_results)}")
        
        # Verify that citations are found and validated
        assert len(citation_results) > 0, "Should find citations in the draft"
    
    @pytest.mark.asyncio
    async def test_quality_metrics_validation(self, quality_metrics_validator):
        """Test quality metrics validation system"""
        
        print(f"\n📊 QUALITY METRICS VALIDATION TESTING")
        print("=" * 45)
        
        # Test different quality scenarios
        test_scenarios = [
            {
                'name': 'high_quality_text',
                'metrics': QualityMetrics(
                    overall_score=0.85,
                    completeness=0.9,
                    coherence=0.85,
                    accuracy=0.95,
                    citation_quality=0.8,
                    readability=0.75
                )
            },
            {
                'name': 'medium_quality_text',
                'metrics': QualityMetrics(
                    overall_score=0.65,
                    completeness=0.7,
                    coherence=0.6,
                    accuracy=0.7,
                    citation_quality=0.6,
                    readability=0.65
                )
            },
            {
                'name': 'low_quality_text',
                'metrics': QualityMetrics(
                    overall_score=0.35,
                    completeness=0.4,
                    coherence=0.3,
                    accuracy=0.4,
                    citation_quality=0.3,
                    readability=0.35
                )
            }
        ]
        
        validation_results = []
        
        for scenario in test_scenarios:
            print(f"\n🧪 Testing scenario: {scenario['name']}")
            
            # Validate quality metrics
            validation_result = await quality_metrics_validator.validate_quality_metrics(
                scenario['metrics'], scenario['name']
            )
            
            validation_results.append(validation_result)
            
            print(f"   🎯 Within Range: {validation_result.within_range}")
            print(f"   🔄 Reliability: {validation_result.reliability_score:.2f}")
            print(f"   📊 Consistency: {validation_result.consistency_score:.2f}")
            print(f"   📈 Accuracy Details: {len(validation_result.accuracy_assessment)} metrics")
            
            # Assertions
            assert 0.0 <= validation_result.reliability_score <= 1.0, \
                "Reliability score must be between 0 and 1"
            assert 0.0 <= validation_result.consistency_score <= 1.0, \
                "Consistency score must be between 0 and 1"
            assert len(validation_result.accuracy_assessment) > 0, \
                "Should have accuracy assessment for metrics"
            
            # Check that high quality scenarios have high reliability
            if scenario['name'] == 'high_quality_text':
                assert validation_result.reliability_score >= 0.8, \
                    "High quality scenario should have high reliability"
        
        print(f"\n✅ Quality metrics validation completed for {len(validation_results)} scenarios")
        
        # Verify that different scenarios produce different validation results
        reliability_scores = [r.reliability_score for r in validation_results]
        assert len(set(reliability_scores)) > 1, "Different scenarios should produce different reliability scores"


# Export main classes
__all__ = [
    "FactualAccuracyValidator",
    "CoherenceAssessmentEngine", 
    "CitationValidationSystem",
    "QualityMetricsValidator",
    "TestEvaluationMetricsQualityAssurance",
    "FactualAccuracyResult",
    "CoherenceAssessmentResult",
    "CitationValidationResult",
    "ReadabilityAssessmentResult",
    "QualityMetricsValidationResult"
]