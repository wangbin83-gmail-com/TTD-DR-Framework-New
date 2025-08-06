"""
Unit tests for Google Search API client.
Tests authentication, rate limiting, error handling, and search functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import httpx

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.google_search_client import (
    GoogleSearchClient, GoogleSearchResponse, GoogleSearchResult, 
    GoogleSearchError, GoogleSearchRateLimiter
)
from config.settings import settings

class TestGoogleSearchRateLimiter:
    """Test the rate limiter functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed immediately"""
        limiter = GoogleSearchRateLimiter(max_requests=5, time_window=60)
        
        # Should allow requests under the limit
        for _ in range(5):
            await limiter.acquire()  # Should not block
        
        assert len(limiter.requests) == 5
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Test that requests are blocked when limit is exceeded"""
        limiter = GoogleSearchRateLimiter(max_requests=2, time_window=1)
        
        # Fill up the limit
        await limiter.acquire()
        await limiter.acquire()
        
        # This should be very fast since we're not actually waiting
        start_time = datetime.now()
        
        # Mock time.time to simulate time passing
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 0, 0, 2]  # Simulate time passing
            await limiter.acquire()
        
        # Should have cleaned up old requests
        assert len(limiter.requests) <= 2

class TestGoogleSearchClient:
    """Test the Google Search client functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = GoogleSearchClient()
        
        # Mock API credentials
        self.client.api_key = "test_api_key"
        self.client.search_engine_id = "test_engine_id"
    
    def test_build_search_params_basic(self):
        """Test basic search parameter building"""
        params = self.client._build_search_params("test query")
        
        assert params["key"] == "test_api_key"
        assert params["cx"] == "test_engine_id"
        assert params["q"] == "test query"
        assert params["num"] == 10
        assert params["start"] == 1
    
    def test_build_search_params_with_options(self):
        """Test search parameter building with additional options"""
        params = self.client._build_search_params(
            "test query",
            num_results=5,
            start=11,
            site_search="example.com",
            file_type="pdf",
            date_restrict="m1"
        )
        
        assert params["num"] == 5
        assert params["start"] == 11
        assert params["siteSearch"] == "example.com"
        assert params["fileType"] == "pdf"
        assert params["dateRestrict"] == "m1"
    
    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search operation"""
        # Mock successful API response
        mock_response_data = {
            "items": [
                {
                    "title": "Test Result 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a test snippet",
                    "displayLink": "example.com",
                    "formattedUrl": "https://example.com/1"
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "Another test snippet",
                    "displayLink": "example.com",
                    "formattedUrl": "https://example.com/2"
                }
            ],
            "searchInformation": {
                "totalResults": "1000",
                "searchTime": "0.45"
            },
            "queries": {},
            "context": {},
            "spelling": {}
        }
        
        with patch.object(self.client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            response = await self.client.search("test query")
            
            assert isinstance(response, GoogleSearchResponse)
            assert len(response.items) == 2
            assert response.items[0].title == "Test Result 1"
            assert response.items[0].link == "https://example.com/1"
            assert response.total_results == 1000
            assert response.search_time == 0.45
    
    @pytest.mark.asyncio
    async def test_search_no_credentials(self):
        """Test search with missing credentials"""
        client = GoogleSearchClient()
        client.api_key = None
        client.search_engine_id = None
        
        with pytest.raises(GoogleSearchError) as exc_info:
            await client.search("test query")
        
        assert exc_info.value.error_type == "configuration"
        assert "not configured" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_search_api_error_handling(self):
        """Test handling of various API errors"""
        error_scenarios = [
            (403, {"error": {"errors": [{"reason": "dailyLimitExceeded"}]}}, "quota_exceeded"),
            (403, {"error": {"errors": [{"reason": "keyInvalid"}]}}, "invalid_key"),
            (400, {"error": {"message": "Invalid query"}}, "bad_request"),
            (500, {}, "server")
        ]
        
        for status_code, response_data, expected_error_type in error_scenarios:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.json.return_value = response_data
            mock_response.text = "Error response"
            mock_response.headers = {"content-type": "application/json"}
            
            with patch.object(self.client.client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_response
                
                with pytest.raises(GoogleSearchError) as exc_info:
                    await self.client._make_request({})
                
                assert exc_info.value.error_type == expected_error_type
    
    @pytest.mark.asyncio
    async def test_search_retry_logic(self):
        """Test retry logic for transient failures"""
        # Mock a scenario where first request fails, second succeeds
        mock_responses = [
            Mock(status_code=500, text="Server error"),
            Mock(status_code=200)
        ]
        mock_responses[1].json.return_value = {"items": [], "searchInformation": {}}
        
        with patch.object(self.client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = mock_responses
            
            # Should succeed after retry
            result = await self.client._make_request({})
            assert result == {"items": [], "searchInformation": {}}
            
            # Should have made 2 requests
            assert mock_get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_search_timeout_handling(self):
        """Test handling of request timeouts"""
        with patch.object(self.client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timeout")
            
            with pytest.raises(GoogleSearchError) as exc_info:
                await self.client._make_request({})
            
            assert exc_info.value.error_type == "timeout"
    
    @pytest.mark.asyncio
    async def test_search_multiple_pages(self):
        """Test searching multiple pages for more results"""
        # Mock responses for multiple pages
        page_responses = [
            {
                "items": [{"title": f"Result {i}", "link": f"https://example.com/{i}", 
                          "snippet": f"Snippet {i}"} for i in range(1, 11)],
                "searchInformation": {"totalResults": "100", "searchTime": "0.3"}
            },
            {
                "items": [{"title": f"Result {i}", "link": f"https://example.com/{i}", 
                          "snippet": f"Snippet {i}"} for i in range(11, 21)],
                "searchInformation": {"totalResults": "100", "searchTime": "0.4"}
            }
        ]
        
        with patch.object(self.client, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = [
                GoogleSearchResponse(
                    items=[GoogleSearchResult(**item) for item in page["items"]],
                    search_information=page["searchInformation"],
                    total_results=int(page["searchInformation"]["totalResults"]),
                    search_time=float(page["searchInformation"]["searchTime"])
                ) for page in page_responses
            ]
            
            response = await self.client.search_multiple_pages("test query", max_results=20)
            
            assert len(response.items) == 20
            assert response.items[0].title == "Result 1"
            assert response.items[19].title == "Result 20"
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        with patch.object(self.client, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = GoogleSearchResponse(items=[])
            
            health = await self.client.health_check()
            assert health is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure"""
        with patch.object(self.client, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = GoogleSearchError("API error")
            
            health = await self.client.health_check()
            assert health is False

class TestGoogleSearchResult:
    """Test GoogleSearchResult model"""
    
    def test_google_search_result_creation(self):
        """Test creating a GoogleSearchResult"""
        result = GoogleSearchResult(
            title="Test Title",
            link="https://example.com",
            snippet="Test snippet"
        )
        
        assert result.title == "Test Title"
        assert result.link == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.display_link == ""  # Default value
    
    def test_google_search_result_with_all_fields(self):
        """Test creating a GoogleSearchResult with all fields"""
        result = GoogleSearchResult(
            title="Test Title",
            link="https://example.com",
            snippet="Test snippet",
            display_link="example.com",
            formatted_url="https://example.com/page",
            html_snippet="<b>Test</b> snippet",
            cache_id="abc123",
            page_map={"metatags": [{"title": "Test"}]},
            mime="text/html",
            file_format="HTML"
        )
        
        assert result.display_link == "example.com"
        assert result.html_snippet == "<b>Test</b> snippet"
        assert result.page_map == {"metatags": [{"title": "Test"}]}

class TestGoogleSearchResponse:
    """Test GoogleSearchResponse model"""
    
    def test_google_search_response_creation(self):
        """Test creating a GoogleSearchResponse"""
        results = [
            GoogleSearchResult(title="Test 1", link="https://example.com/1", snippet="Snippet 1"),
            GoogleSearchResult(title="Test 2", link="https://example.com/2", snippet="Snippet 2")
        ]
        
        response = GoogleSearchResponse(
            items=results,
            total_results=100,
            search_time=0.45
        )
        
        assert len(response.items) == 2
        assert response.total_results == 100
        assert response.search_time == 0.45
    
    def test_google_search_response_empty(self):
        """Test creating an empty GoogleSearchResponse"""
        response = GoogleSearchResponse()
        
        assert len(response.items) == 0
        assert response.total_results == 0
        assert response.search_time == 0.0

@pytest.mark.integration
class TestGoogleSearchIntegration:
    """Integration tests for Google Search (requires actual API credentials)"""
    
    @pytest.mark.skipif(
        not settings.google_search_api_key or not settings.google_search_engine_id,
        reason="Google Search API credentials not configured"
    )
    @pytest.mark.asyncio
    async def test_real_search_query(self):
        """Test a real search query (only runs if credentials are configured)"""
        client = GoogleSearchClient()
        
        try:
            response = await client.search("Python programming", num_results=3)
            
            assert isinstance(response, GoogleSearchResponse)
            assert len(response.items) <= 3
            
            if response.items:
                result = response.items[0]
                assert result.title
                assert result.link.startswith("http")
                assert result.snippet
                
        except GoogleSearchError as e:
            # Skip test if API is not available
            pytest.skip(f"Google Search API not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__])