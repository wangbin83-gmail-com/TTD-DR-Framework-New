"""
Dynamic Retrieval Engine for TTD-DR framework.
Integrates Google Search API with intelligent content processing and filtering.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from urllib.parse import urlparse
import re
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import (
    InformationGap, RetrievedInfo, Source, SearchQuery, 
    Priority, GapType, ResearchDomain
)
from services.google_search_client import (
    GoogleSearchClient, GoogleSearchResponse, GoogleSearchResult, GoogleSearchError
)
from services.kimi_k2_client import KimiK2Client, KimiK2Error
from services.content_filter import ContentFilteringPipeline, ContentQualityAssessor

logger = logging.getLogger(__name__)

@dataclass
class CredibilityFactors:
    """Factors used to assess source credibility"""
    domain_authority: float = 0.5
    search_ranking: float = 0.5
    content_quality: float = 0.5
    recency: float = 0.5
    source_type: str = "unknown"

class SourceCredibilityScorer:
    """Scores source credibility based on various factors"""
    
    # Domain authority scores (simplified)
    DOMAIN_SCORES = {
        # Academic and educational
        ".edu": 0.9,
        ".ac.": 0.9,
        "scholar.google.com": 0.95,
        "arxiv.org": 0.9,
        "pubmed.ncbi.nlm.nih.gov": 0.95,
        "jstor.org": 0.9,
        
        # Government and official
        ".gov": 0.9,
        ".mil": 0.85,
        "who.int": 0.9,
        "cdc.gov": 0.9,
        "fda.gov": 0.9,
        
        # Reputable news and media
        "reuters.com": 0.85,
        "bbc.com": 0.85,
        "npr.org": 0.8,
        "apnews.com": 0.85,
        "nytimes.com": 0.8,
        "wsj.com": 0.8,
        "economist.com": 0.8,
        
        # Professional and industry
        "ieee.org": 0.85,
        "acm.org": 0.85,
        "nature.com": 0.9,
        "science.org": 0.9,
        "nejm.org": 0.9,
        
        # Technology
        "stackoverflow.com": 0.7,
        "github.com": 0.7,
        "medium.com": 0.6,
        
        # General reference
        "wikipedia.org": 0.7,
        "britannica.com": 0.75,
        
        # Lower credibility
        "blog": 0.4,
        "wordpress.com": 0.4,
        "blogspot.com": 0.4,
        "tumblr.com": 0.3,
    }
    
    def score_domain_authority(self, url: str) -> float:
        """Score domain authority based on URL"""
        domain = urlparse(url).netloc.lower()
        
        # Check exact matches first
        for known_domain, score in self.DOMAIN_SCORES.items():
            if known_domain in domain:
                return score
        
        # Check TLD patterns
        if domain.endswith('.edu') or domain.endswith('.gov'):
            return 0.9
        elif domain.endswith('.org'):
            return 0.7
        elif domain.endswith('.com'):
            return 0.6
        else:
            return 0.5
    
    def score_search_ranking(self, position: int, total_results: int) -> float:
        """Score based on search result position"""
        if position <= 3:
            return 0.9
        elif position <= 10:
            return 0.8
        elif position <= 20:
            return 0.7
        else:
            return max(0.3, 1.0 - (position / max(total_results, 100)))
    
    def score_content_quality(self, result: GoogleSearchResult) -> float:
        """Score content quality based on snippet and metadata"""
        score = 0.5
        
        # Check snippet quality
        if result.snippet:
            snippet_length = len(result.snippet)
            if snippet_length > 100:
                score += 0.1
            if snippet_length > 200:
                score += 0.1
            
            # Look for quality indicators
            quality_indicators = [
                "research", "study", "analysis", "report", "findings",
                "data", "statistics", "evidence", "methodology"
            ]
            snippet_lower = result.snippet.lower()
            matches = sum(1 for indicator in quality_indicators if indicator in snippet_lower)
            score += min(0.2, matches * 0.05)
        
        # Check title quality
        if result.title:
            title_lower = result.title.lower()
            if any(word in title_lower for word in ["study", "research", "analysis", "report"]):
                score += 0.1
        
        return min(1.0, score)
    
    def determine_source_type(self, url: str, result: GoogleSearchResult) -> str:
        """Determine the type of source"""
        domain = urlparse(url).netloc.lower()
        
        if any(edu in domain for edu in ['.edu', '.ac.', 'scholar.google']):
            return "academic"
        elif any(gov in domain for gov in ['.gov', '.mil']):
            return "government"
        elif any(news in domain for news in ['reuters', 'bbc', 'npr', 'nytimes', 'wsj']):
            return "news"
        elif any(ref in domain for ref in ['wikipedia', 'britannica']):
            return "reference"
        elif any(tech in domain for tech in ['stackoverflow', 'github']):
            return "technical"
        elif any(blog in domain for blog in ['blog', 'wordpress', 'medium']):
            return "blog"
        else:
            return "general"
    
    def calculate_credibility_score(self, result: GoogleSearchResult, position: int, 
                                  total_results: int) -> CredibilityFactors:
        """Calculate overall credibility score"""
        domain_authority = self.score_domain_authority(result.link)
        search_ranking = self.score_search_ranking(position, total_results)
        content_quality = self.score_content_quality(result)
        source_type = self.determine_source_type(result.link, result)
        
        # Simple recency scoring (would need actual date parsing for real implementation)
        recency = 0.7  # Default assumption of reasonably recent
        
        return CredibilityFactors(
            domain_authority=domain_authority,
            search_ranking=search_ranking,
            content_quality=content_quality,
            recency=recency,
            source_type=source_type
        )

class ContentProcessor:
    """Processes and filters search result content"""
    
    def __init__(self):
        self.kimi_client = KimiK2Client()
    
    def extract_relevant_content(self, result: GoogleSearchResult, gap: InformationGap) -> str:
        """Extract relevant content from search result"""
        # Start with the snippet
        content = result.snippet
        
        # Add HTML snippet if available and different
        if result.html_snippet and result.html_snippet != result.snippet:
            # Clean HTML tags
            clean_html = re.sub(r'<[^>]+>', '', result.html_snippet)
            if clean_html not in content:
                content += f"\n\n{clean_html}"
        
        # Add structured data if available
        if result.page_map:
            # Extract useful structured data
            for key, value in result.page_map.items():
                if key in ['metatags', 'cse_thumbnail', 'cse_image']:
                    continue
                if isinstance(value, list) and value:
                    content += f"\n\n{key}: {value[0]}"
        
        return content.strip()
    
    def calculate_relevance_score(self, content: str, gap: InformationGap, 
                                query: SearchQuery) -> float:
        """Calculate relevance score for content"""
        score = 0.5
        
        # Check for query terms
        query_terms = query.query.lower().split()
        content_lower = content.lower()
        
        term_matches = sum(1 for term in query_terms if term in content_lower)
        if query_terms:
            score += 0.3 * (term_matches / len(query_terms))
        
        # Check for gap-specific terms
        gap_terms = gap.description.lower().split()
        gap_matches = sum(1 for term in gap_terms if term in content_lower)
        if gap_terms:
            score += 0.2 * (gap_matches / len(gap_terms))
        
        # Bonus for longer, more detailed content
        if len(content) > 200:
            score += 0.1
        if len(content) > 500:
            score += 0.1
        
        return min(1.0, score)
    
    async def enhance_content_with_kimi(self, content: str, gap: InformationGap) -> str:
        """Use Kimi K2 to enhance and summarize content"""
        try:
            prompt = f"""
Please analyze and enhance the following content for research purposes.

Research Gap: {gap.description}
Gap Type: {gap.gap_type}

Content to analyze:
{content}

Please provide:
1. A concise summary of the key information
2. How this information addresses the research gap
3. Any important details or data points
4. Potential limitations or considerations

Keep the response focused and research-oriented.
"""
            
            response = await self.kimi_client.generate_text(prompt, max_tokens=1000)
            return response.content
            
        except Exception as e:
            logger.warning(f"Failed to enhance content with Kimi K2: {e}")
            return content

class DuplicateDetector:
    """Detects and handles duplicate content"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.seen_urls: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
    
    def is_duplicate_url(self, url: str) -> bool:
        """Check if URL has been seen before"""
        normalized_url = self._normalize_url(url)
        if normalized_url in self.seen_urls:
            return True
        self.seen_urls.add(normalized_url)
        return False
    
    def is_duplicate_content(self, content: str) -> bool:
        """Check if content is substantially similar to previously seen content"""
        content_hash = hash(content.lower().strip())
        if content_hash in self.seen_content_hashes:
            return True
        self.seen_content_hashes.add(content_hash)
        return False
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison"""
        # Remove common URL parameters and fragments
        parsed = urlparse(url)
        normalized = f"{parsed.netloc}{parsed.path}"
        return normalized.lower().rstrip('/')

class DynamicRetrievalEngine:
    """Main retrieval engine that orchestrates Google Search API integration"""
    
    def __init__(self):
        self.google_client = GoogleSearchClient()
        self.credibility_scorer = SourceCredibilityScorer()
        self.content_processor = ContentProcessor()
        self.duplicate_detector = DuplicateDetector()
        self.content_filter = ContentFilteringPipeline()
        
    async def retrieve_information(self, gaps: List[InformationGap], 
                                 max_results_per_gap: int = 10) -> List[RetrievedInfo]:
        """
        Retrieve information for multiple information gaps
        
        Args:
            gaps: List of information gaps to address
            max_results_per_gap: Maximum results to retrieve per gap
            
        Returns:
            List of retrieved information items
        """
        all_retrieved_info = []
        
        for gap in gaps:
            logger.info(f"Retrieving information for gap: {gap.id}")
            
            try:
                gap_info = await self._retrieve_for_single_gap(gap, max_results_per_gap)
                all_retrieved_info.extend(gap_info)
                
            except Exception as e:
                logger.error(f"Failed to retrieve information for gap {gap.id}: {e}")
                continue
        
        # Sort by relevance and credibility
        all_retrieved_info.sort(
            key=lambda x: (x.relevance_score + x.credibility_score) / 2,
            reverse=True
        )
        
        return all_retrieved_info
    
    async def _retrieve_for_single_gap(self, gap: InformationGap, 
                                     max_results: int) -> List[RetrievedInfo]:
        """Retrieve information for a single gap"""
        retrieved_info = []
        
        # Process each search query for the gap
        for query in gap.search_queries[:3]:  # Limit to top 3 queries per gap
            try:
                query_results = await self._search_and_process(query, gap, max_results // len(gap.search_queries))
                retrieved_info.extend(query_results)
                
            except GoogleSearchError as e:
                logger.error(f"Google Search failed for query '{query.query}': {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing query '{query.query}': {e}")
                continue
        
        return retrieved_info
    
    async def _search_and_process(self, query: SearchQuery, gap: InformationGap, 
                                max_results: int) -> List[RetrievedInfo]:
        """Search and process results for a single query"""
        # Perform Google search
        search_response = await self.google_client.search(
            query.query,
            num_results=min(max_results, 10),
            safe="medium"
        )
        
        # Apply advanced content filtering to search results
        filtered_results = self.content_filter.filter_search_results(
            search_response.items, [gap], min_quality_score=0.4
        )
        
        logger.info(f"Filtered {len(search_response.items)} results to {len(filtered_results)} high-quality results")
        
        retrieved_info = []
        
        for result, quality_metrics in filtered_results:
            # Calculate credibility using existing scorer
            credibility_factors = self.credibility_scorer.calculate_credibility_score(
                result, len(retrieved_info) + 1, search_response.total_results
            )
            
            # Extract and process content
            content = self.content_processor.extract_relevant_content(result, gap)
            
            # Calculate relevance score
            relevance_score = self.content_processor.calculate_relevance_score(
                content, gap, query
            )
            
            # Use the higher of calculated relevance or quality assessment relevance
            final_relevance = max(relevance_score, quality_metrics.relevance_score)
            
            # Enhance content with Kimi K2 if relevance is high
            if final_relevance > 0.7:
                try:
                    content = await self.content_processor.enhance_content_with_kimi(content, gap)
                except Exception as e:
                    logger.warning(f"Failed to enhance content: {e}")
            
            # Calculate overall credibility score combining multiple factors
            overall_credibility = (
                credibility_factors.domain_authority * 0.25 +
                credibility_factors.search_ranking * 0.15 +
                credibility_factors.content_quality * 0.20 +
                credibility_factors.recency * 0.15 +
                quality_metrics.source_authority * 0.25
            )
            
            # Create source and retrieved info
            source = Source(
                url=result.link,
                title=result.title,
                domain=urlparse(result.link).netloc,
                credibility_score=overall_credibility
            )
            
            info = RetrievedInfo(
                source=source,
                content=content,
                relevance_score=final_relevance,
                credibility_score=overall_credibility,
                gap_id=gap.id
            )
            
            retrieved_info.append(info)
        
        return retrieved_info
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all components"""
        return {
            "google_search": await self.google_client.health_check(),
            "kimi_k2": await self.content_processor.kimi_client.health_check()
        }

# Singleton instance for global use
dynamic_retrieval_engine = DynamicRetrievalEngine()