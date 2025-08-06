"""
Google Search API client for TTD-DR framework.
Provides comprehensive web search capabilities with rate limiting and error handling.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
import httpx
import json
from datetime import datetime
from pydantic import BaseModel
import logging
from urllib.parse import quote_plus

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

logger = logging.getLogger(__name__)

class GoogleSearchResult(BaseModel):
    """Represents a single Google Search result"""
    title: str
    link: str
    snippet: str
    display_link: str = ""
    formatted_url: str = ""
    html_snippet: str = ""
    cache_id: str = ""
    
    # Additional metadata
    page_map: Dict[str, Any] = {}
    mime: str = ""
    file_format: str = ""

class GoogleSearchResponse(BaseModel):
    """Complete Google Search API response"""
    items: List[GoogleSearchResult] = []
    search_information: Dict[str, Any] = {}
    queries: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    spelling: Dict[str, Any] = {}
    
    # Metadata
    total_results: int = 0
    search_time: float = 0.0
    kind: str = ""

class GoogleSearchError(Exception):
    """Custom exception for Google Search API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: str = "unknown"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

class GoogleSearchRateLimiter:
    """Rate limiter specifically for Google Search API"""
    def __init__(self, max_requests: int = 100, time_window: int = 86400):  # 100 requests per day
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            # Calculate wait time until oldest request expires
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request) + 1
            if wait_time > 0:
                logger.warning(f"Google Search API rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self.requests.append(now)

class GoogleSearchClient:
    """Client for Google Custom Search API"""
    
    def __init__(self):
        self.api_key = settings.google_search_api_key
        self.search_engine_id = settings.google_search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Rate limiter (Google allows 100 queries per day for free tier)
        self.rate_limiter = GoogleSearchRateLimiter(max_requests=100, time_window=86400)
        
        # HTTP client with timeout and retry configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google Search API credentials not configured")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _build_search_params(self, query: str, **kwargs) -> Dict[str, Any]:
        """Build search parameters for Google API"""
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": kwargs.get("num_results", 10),  # Number of results (max 10 per request)
            "start": kwargs.get("start", 1),  # Starting index
            "safe": kwargs.get("safe", "medium"),  # Safe search
            "lr": kwargs.get("language", "lang_en"),  # Language restriction
            "gl": kwargs.get("country", "us"),  # Country restriction
        }
        
        # Optional parameters
        if kwargs.get("site_search"):
            params["siteSearch"] = kwargs["site_search"]
        
        if kwargs.get("file_type"):
            params["fileType"] = kwargs["file_type"]
        
        if kwargs.get("date_restrict"):
            params["dateRestrict"] = kwargs["date_restrict"]
        
        if kwargs.get("sort"):
            params["sort"] = kwargs["sort"]
        
        return params
    
    async def _make_request(self, params: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        """Make HTTP request to Google Search API with retry logic"""
        await self.rate_limiter.acquire()
        
        for attempt in range(retries):
            try:
                response = await self.client.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Google API rate limited, waiting {wait_time} seconds (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    error_reason = error_data.get("error", {}).get("errors", [{}])[0].get("reason", "forbidden")
                    
                    if error_reason == "dailyLimitExceeded":
                        raise GoogleSearchError("Daily quota exceeded", response.status_code, "quota_exceeded")
                    elif error_reason == "keyInvalid":
                        raise GoogleSearchError("Invalid API key", response.status_code, "invalid_key")
                    else:
                        raise GoogleSearchError(f"Access forbidden: {error_reason}", response.status_code, "forbidden")
                elif response.status_code == 400:
                    error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    error_message = error_data.get("error", {}).get("message", "Bad request")
                    raise GoogleSearchError(f"Bad request: {error_message}", response.status_code, "bad_request")
                else:
                    error_detail = response.text
                    raise GoogleSearchError(f"API error: {error_detail}", response.status_code, "api_error")
                    
            except httpx.TimeoutException:
                if attempt < retries - 1:
                    logger.warning(f"Request timeout, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise GoogleSearchError("Request timeout", None, "timeout")
            except httpx.RequestError as e:
                if attempt < retries - 1:
                    logger.warning(f"Request error: {e}, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise GoogleSearchError(f"Request error: {e}", None, "network")
        
        raise GoogleSearchError("Max retries exceeded", None, "retry_exhausted")
    
    async def search(self, query: str, **kwargs) -> GoogleSearchResponse:
        """
        Perform a Google search
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
                - num_results: Number of results to return (1-10, default 10)
                - start: Starting index for results (default 1)
                - safe: Safe search setting ("off", "medium", "high")
                - language: Language restriction (e.g., "lang_en")
                - country: Country restriction (e.g., "us")
                - site_search: Restrict to specific site
                - file_type: Restrict to specific file type
                - date_restrict: Date restriction (e.g., "d1", "w1", "m1", "y1")
                - sort: Sort order
        
        Returns:
            GoogleSearchResponse with search results
        
        Raises:
            GoogleSearchError: If search fails
        """
        if not self.api_key or not self.search_engine_id:
            raise GoogleSearchError("Google Search API credentials not configured", None, "configuration")
        
        try:
            params = self._build_search_params(query, **kwargs)
            response_data = await self._make_request(params)
            
            # Parse response
            items = []
            if "items" in response_data:
                for item in response_data["items"]:
                    result = GoogleSearchResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        display_link=item.get("displayLink", ""),
                        formatted_url=item.get("formattedUrl", ""),
                        html_snippet=item.get("htmlSnippet", ""),
                        cache_id=item.get("cacheId", ""),
                        page_map=item.get("pagemap", {}),
                        mime=item.get("mime", ""),
                        file_format=item.get("fileFormat", "")
                    )
                    items.append(result)
            
            # Extract metadata
            search_info = response_data.get("searchInformation", {})
            total_results = int(search_info.get("totalResults", "0"))
            search_time = float(search_info.get("searchTime", "0"))
            
            return GoogleSearchResponse(
                items=items,
                search_information=search_info,
                queries=response_data.get("queries", {}),
                context=response_data.get("context", {}),
                spelling=response_data.get("spelling", {}),
                total_results=total_results,
                search_time=search_time,
                kind=response_data.get("kind", "")
            )
            
        except GoogleSearchError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise GoogleSearchError(f"Unexpected error: {e}", None, "unexpected")
    
    async def search_multiple_pages(self, query: str, max_results: int = 20, **kwargs) -> GoogleSearchResponse:
        """
        Search multiple pages to get more results
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
        
        Returns:
            Combined GoogleSearchResponse with results from multiple pages
        """
        all_items = []
        search_info = {}
        queries = {}
        context = {}
        spelling = {}
        
        results_per_page = min(10, max_results)  # Google API max is 10 per request
        pages_needed = (max_results + results_per_page - 1) // results_per_page
        
        for page in range(pages_needed):
            start_index = page * results_per_page + 1
            remaining_results = max_results - len(all_items)
            current_page_size = min(results_per_page, remaining_results)
            
            if current_page_size <= 0:
                break
            
            try:
                response = await self.search(
                    query,
                    num_results=current_page_size,
                    start=start_index,
                    **kwargs
                )
                
                all_items.extend(response.items)
                
                # Keep metadata from first page
                if page == 0:
                    search_info = response.search_information
                    queries = response.queries
                    context = response.context
                    spelling = response.spelling
                
                # If we got fewer results than requested, we've reached the end
                if len(response.items) < current_page_size:
                    break
                    
            except GoogleSearchError as e:
                if page == 0:
                    # If first page fails, re-raise the error
                    raise
                else:
                    # If subsequent pages fail, log and continue with what we have
                    logger.warning(f"Failed to fetch page {page + 1}: {e}")
                    break
        
        return GoogleSearchResponse(
            items=all_items,
            search_information=search_info,
            queries=queries,
            context=context,
            spelling=spelling,
            total_results=search_info.get("totalResults", len(all_items)),
            search_time=sum(getattr(response, 'search_time', 0) for response in [])
        )
    
    async def health_check(self) -> bool:
        """Check if the Google Search API is accessible"""
        try:
            response = await self.search("test", num_results=1)
            return len(response.items) >= 0  # Even 0 results is a successful response
        except Exception as e:
            logger.error(f"Google Search API health check failed: {e}")
            return False

# Singleton instance for global use
google_search_client = GoogleSearchClient()