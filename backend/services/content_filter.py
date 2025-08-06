"""
Advanced content filtering system for TTD-DR framework.
Provides sophisticated content quality assessment and filtering capabilities.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import RetrievedInfo, Source, InformationGap, GapType, ResearchDomain
from services.google_search_client import GoogleSearchResult

logger = logging.getLogger(__name__)

@dataclass
class ContentQualityMetrics:
    """Metrics for assessing content quality"""
    readability_score: float = 0.0
    information_density: float = 0.0
    factual_indicators: float = 0.0
    source_authority: float = 0.0
    recency_score: float = 0.0
    relevance_score: float = 0.0
    overall_score: float = 0.0

@dataclass
class DuplicationSignature:
    """Signature for detecting content duplication"""
    content_hash: str
    title_hash: str
    domain: str
    similarity_threshold: float = 0.8

class ContentQualityAssessor:
    """Assesses the quality of retrieved content"""
    
    # Quality indicators for different content types
    QUALITY_INDICATORS = {
        "academic": [
            "research", "study", "analysis", "methodology", "findings",
            "data", "statistics", "peer-reviewed", "journal", "publication",
            "hypothesis", "experiment", "results", "conclusion", "abstract"
        ],
        "technical": [
            "implementation", "algorithm", "framework", "architecture",
            "specification", "documentation", "tutorial", "guide",
            "best practices", "performance", "optimization", "testing"
        ],
        "news": [
            "report", "investigation", "interview", "source", "statement",
            "official", "confirmed", "according to", "spokesperson",
            "breaking", "update", "development", "announcement"
        ],
        "reference": [
            "definition", "explanation", "overview", "introduction",
            "background", "history", "concept", "principle", "theory",
            "example", "illustration", "comparison", "classification"
        ]
    }
    
    # Low quality indicators
    LOW_QUALITY_INDICATORS = [
        "click here", "read more", "subscribe", "advertisement",
        "sponsored", "affiliate", "buy now", "limited time",
        "amazing", "incredible", "unbelievable", "shocking",
        "you won't believe", "doctors hate", "one weird trick"
    ]
    
    def assess_content_quality(self, content: str, source: Source, 
                             gap: InformationGap) -> ContentQualityMetrics:
        """
        Assess the overall quality of content
        
        Args:
            content: The content to assess
            source: Source information
            gap: Information gap this content addresses
            
        Returns:
            ContentQualityMetrics with detailed quality scores
        """
        metrics = ContentQualityMetrics()
        
        # Assess different quality dimensions
        metrics.readability_score = self._assess_readability(content)
        metrics.information_density = self._assess_information_density(content)
        metrics.factual_indicators = self._assess_factual_indicators(content, gap)
        metrics.source_authority = source.credibility_score
        metrics.recency_score = self._assess_recency(source)
        metrics.relevance_score = self._assess_relevance(content, gap)
        
        # Calculate overall score
        weights = {
            "readability": 0.15,
            "information_density": 0.20,
            "factual_indicators": 0.25,
            "source_authority": 0.20,
            "recency": 0.10,
            "relevance": 0.10
        }
        
        metrics.overall_score = (
            metrics.readability_score * weights["readability"] +
            metrics.information_density * weights["information_density"] +
            metrics.factual_indicators * weights["factual_indicators"] +
            metrics.source_authority * weights["source_authority"] +
            metrics.recency_score * weights["recency"] +
            metrics.relevance_score * weights["relevance"]
        )
        
        return metrics
    
    def _assess_readability(self, content: str) -> float:
        """Assess content readability"""
        if not content:
            return 0.0
        
        # Simple readability metrics
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            readability = 1.0
        elif 5 <= avg_sentence_length <= 35:
            readability = 0.8
        else:
            readability = 0.5
        
        # Check for proper structure
        if content.count('\n') > 0:  # Has paragraphs
            readability += 0.1
        
        # Check for excessive capitalization or poor formatting
        caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
        if caps_ratio > 0.3:  # Too much capitalization
            readability -= 0.2
        
        return min(1.0, max(0.0, readability))
    
    def _assess_information_density(self, content: str) -> float:
        """Assess how information-dense the content is"""
        if not content:
            return 0.0
        
        words = content.split()
        if len(words) < 10:
            return 0.3  # Too short to be informative
        
        # Look for informative words
        informative_words = [
            "data", "research", "study", "analysis", "result", "finding",
            "evidence", "fact", "statistic", "number", "percent", "rate",
            "increase", "decrease", "trend", "pattern", "correlation",
            "cause", "effect", "impact", "influence", "significant"
        ]
        
        informative_count = sum(1 for word in words 
                              if word.lower() in informative_words)
        
        density = informative_count / len(words)
        
        # Normalize to 0-1 scale
        if density > 0.1:
            return min(1.0, density * 5)
        else:
            return density * 2
    
    def _assess_factual_indicators(self, content: str, gap: InformationGap) -> float:
        """Assess presence of factual indicators"""
        content_lower = content.lower()
        
        # Determine content type based on gap and content
        content_type = self._determine_content_type(content, gap)
        
        # Get relevant quality indicators
        relevant_indicators = self.QUALITY_INDICATORS.get(content_type, [])
        
        # Count quality indicators
        quality_count = sum(1 for indicator in relevant_indicators 
                          if indicator in content_lower)
        
        # Count low quality indicators
        low_quality_count = sum(1 for indicator in self.LOW_QUALITY_INDICATORS 
                              if indicator in content_lower)
        
        # Calculate score
        if len(relevant_indicators) > 0:
            quality_ratio = quality_count / len(relevant_indicators)
        else:
            quality_ratio = 0.5
        
        # Penalize low quality indicators
        penalty = min(0.5, low_quality_count * 0.1)
        
        return max(0.0, min(1.0, quality_ratio - penalty))
    
    def _assess_recency(self, source: Source) -> float:
        """Assess content recency based on source information"""
        # Simple recency assessment based on last accessed time
        # In a real implementation, this would parse actual publication dates
        
        time_diff = datetime.now() - source.last_accessed
        days_old = time_diff.days
        
        if days_old <= 30:
            return 1.0
        elif days_old <= 90:
            return 0.8
        elif days_old <= 365:
            return 0.6
        elif days_old <= 730:
            return 0.4
        else:
            return 0.2
    
    def _assess_relevance(self, content: str, gap: InformationGap) -> float:
        """Assess content relevance to the information gap"""
        content_lower = content.lower()
        gap_terms = gap.description.lower().split()
        
        # Count matching terms
        matches = sum(1 for term in gap_terms if term in content_lower)
        
        if len(gap_terms) == 0:
            return 0.5
        
        relevance = matches / len(gap_terms)
        
        # Bonus for gap type specific content
        gap_type_indicators = {
            GapType.CONTENT: ["information", "details", "description", "overview"],
            GapType.EVIDENCE: ["evidence", "proof", "data", "research", "study"],
            GapType.CITATION: ["source", "reference", "citation", "bibliography"],
            GapType.ANALYSIS: ["analysis", "interpretation", "conclusion", "insight"]
        }
        
        type_indicators = gap_type_indicators.get(gap.gap_type, [])
        type_matches = sum(1 for indicator in type_indicators 
                          if indicator in content_lower)
        
        if type_indicators:
            relevance += (type_matches / len(type_indicators)) * 0.2
        
        return min(1.0, relevance)
    
    def _determine_content_type(self, content: str, gap: InformationGap) -> str:
        """Determine the type of content for quality assessment"""
        content_lower = content.lower()
        
        # Check for academic indicators
        academic_score = sum(1 for indicator in self.QUALITY_INDICATORS["academic"] 
                           if indicator in content_lower)
        
        # Check for technical indicators
        technical_score = sum(1 for indicator in self.QUALITY_INDICATORS["technical"] 
                            if indicator in content_lower)
        
        # Check for news indicators
        news_score = sum(1 for indicator in self.QUALITY_INDICATORS["news"] 
                       if indicator in content_lower)
        
        # Check for reference indicators
        reference_score = sum(1 for indicator in self.QUALITY_INDICATORS["reference"] 
                            if indicator in content_lower)
        
        scores = {
            "academic": academic_score,
            "technical": technical_score,
            "news": news_score,
            "reference": reference_score
        }
        
        # Return type with highest score, default to reference
        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else "reference"

class AdvancedDuplicateDetector:
    """Advanced duplicate detection with similarity analysis"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.seen_signatures: List[DuplicationSignature] = []
        self.content_hashes: Set[str] = set()
        self.url_patterns: Set[str] = set()
    
    def is_duplicate(self, content: str, title: str, url: str) -> bool:
        """
        Check if content is a duplicate using multiple methods
        
        Args:
            content: Content to check
            title: Title of the content
            url: Source URL
            
        Returns:
            True if content is considered a duplicate
        """
        # Quick hash-based check
        content_hash = self._hash_content(content)
        if content_hash in self.content_hashes:
            return True
        
        # URL pattern check
        url_pattern = self._normalize_url(url)
        if url_pattern in self.url_patterns:
            return True
        
        # Similarity-based check
        if self._is_similar_content(content, title, url):
            return True
        
        # Add to seen content
        self.content_hashes.add(content_hash)
        self.url_patterns.add(url_pattern)
        
        signature = DuplicationSignature(
            content_hash=content_hash,
            title_hash=self._hash_content(title),
            domain=urlparse(url).netloc
        )
        self.seen_signatures.append(signature)
        
        return False
    
    def _hash_content(self, content: str) -> str:
        """Create a hash of content for quick comparison"""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for pattern matching"""
        parsed = urlparse(url)
        # Remove query parameters and fragments
        normalized = f"{parsed.netloc}{parsed.path}"
        return normalized.lower().rstrip('/')
    
    def _is_similar_content(self, content: str, title: str, url: str) -> bool:
        """Check if content is similar to previously seen content"""
        current_signature = DuplicationSignature(
            content_hash=self._hash_content(content),
            title_hash=self._hash_content(title),
            domain=urlparse(url).netloc
        )
        
        for signature in self.seen_signatures:
            # Check domain similarity
            if signature.domain == current_signature.domain:
                # Same domain, check title similarity
                if self._calculate_similarity(title, signature.title_hash) > 0.9:
                    return True
            
            # Check content similarity using simplified approach
            if self._calculate_content_similarity(content, signature.content_hash) > self.similarity_threshold:
                return True
        
        return False
    
    def _calculate_similarity(self, text: str, hash_to_compare: str) -> float:
        """Calculate similarity between text and a hash (simplified)"""
        # This is a simplified similarity calculation
        # In a real implementation, you might use more sophisticated methods
        text_hash = self._hash_content(text)
        return 1.0 if text_hash == hash_to_compare else 0.0
    
    def _calculate_content_similarity(self, content: str, hash_to_compare: str) -> float:
        """Calculate content similarity (simplified)"""
        content_hash = self._hash_content(content)
        return 1.0 if content_hash == hash_to_compare else 0.0

class ContentFilteringPipeline:
    """Main pipeline for filtering and processing search results"""
    
    def __init__(self):
        self.quality_assessor = ContentQualityAssessor()
        self.duplicate_detector = AdvancedDuplicateDetector()
        
    def filter_search_results(self, results: List[GoogleSearchResult], 
                            gaps: List[InformationGap],
                            min_quality_score: float = 0.5) -> List[Tuple[GoogleSearchResult, ContentQualityMetrics]]:
        """
        Filter search results based on quality and duplication
        
        Args:
            results: List of Google search results
            gaps: Information gaps being addressed
            min_quality_score: Minimum quality score threshold
            
        Returns:
            List of filtered results with quality metrics
        """
        filtered_results = []
        
        for result in results:
            # Check for duplicates
            if self.duplicate_detector.is_duplicate(
                result.snippet, result.title, result.link
            ):
                logger.debug(f"Filtered duplicate: {result.title}")
                continue
            
            # Assess quality for each relevant gap
            best_quality = None
            best_gap = None
            
            for gap in gaps:
                # Create temporary source for quality assessment
                temp_source = Source(
                    url=result.link,
                    title=result.title,
                    domain=urlparse(result.link).netloc,
                    credibility_score=0.7  # Default, will be calculated properly
                )
                
                quality_metrics = self.quality_assessor.assess_content_quality(
                    result.snippet, temp_source, gap
                )
                
                if best_quality is None or quality_metrics.overall_score > best_quality.overall_score:
                    best_quality = quality_metrics
                    best_gap = gap
            
            # Filter based on quality threshold
            if best_quality and best_quality.overall_score >= min_quality_score:
                filtered_results.append((result, best_quality))
            else:
                logger.debug(f"Filtered low quality result: {result.title} (score: {best_quality.overall_score if best_quality else 0})")
        
        # Sort by quality score
        filtered_results.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return filtered_results
    
    def process_retrieved_info(self, retrieved_info: List[RetrievedInfo],
                             min_quality_score: float = 0.5) -> List[RetrievedInfo]:
        """
        Post-process retrieved information for final filtering
        
        Args:
            retrieved_info: List of retrieved information
            min_quality_score: Minimum quality score threshold
            
        Returns:
            Filtered and processed retrieved information
        """
        processed_info = []
        
        for info in retrieved_info:
            # Re-assess quality with full content
            gap = InformationGap(
                id="temp",
                section_id="temp",
                gap_type=GapType.CONTENT,
                description="temp",
                priority=Priority.MEDIUM
            )  # Simplified for processing
            
            quality_metrics = self.quality_assessor.assess_content_quality(
                info.content, info.source, gap
            )
            
            # Update relevance score with quality assessment
            info.relevance_score = max(info.relevance_score, quality_metrics.relevance_score)
            
            # Filter based on combined quality
            combined_score = (quality_metrics.overall_score + info.credibility_score) / 2
            
            if combined_score >= min_quality_score:
                processed_info.append(info)
        
        return processed_info
    
    def get_filtering_statistics(self) -> Dict[str, int]:
        """Get statistics about the filtering process"""
        return {
            "total_signatures": len(self.duplicate_detector.seen_signatures),
            "unique_domains": len(set(sig.domain for sig in self.duplicate_detector.seen_signatures)),
            "content_hashes": len(self.duplicate_detector.content_hashes),
            "url_patterns": len(self.duplicate_detector.url_patterns)
        }

# Singleton instance for global use
content_filtering_pipeline = ContentFilteringPipeline()