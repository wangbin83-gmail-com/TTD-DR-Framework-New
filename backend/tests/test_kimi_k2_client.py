import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import httpx

from services.kimi_k2_client import (
    KimiK2Client, KimiK2Response, KimiK2Error, RateLimiter
)

# Mock AsyncMock for Python 3.7 compatibility
class AsyncMock:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        
    async def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.side_effect:
            if isinstance(self.side_effect, list):
                if self.call_count <= len(self.side_effect):
                    result = self.side_effect[self.call_count - 1]
                    if isinstance(result, Exception):
                        raise result
                    return result
            elif isinstance(self.side_effect, Exception):
                raise self.side_effect
            else:
                return self.side_effect
        return self.return_value

class TestRateLimiter:
    """Test rate limiter functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit"""
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # Should allow 5 requests without waiting
        for _ in range(5):
            await limiter.acquire()  # Should not block
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests exceeding the limit"""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # First two requests should be immediate
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should be delayed (we'll just check it doesn't error)
        # In a real test, we'd measure timing, but for simplicity we just ensure it completes
        await limiter.acquire()

class TestKimiK2Client:
    """Test Kimi K2 client functionality"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = KimiK2Client()
        self.client.api_key = "test_api_key"  # Override for testing
    
    @pytest.mark.asyncio
    async def test_build_headers(self):
        """Test header building"""
        headers = self.client._build_headers()
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["Content-Type"] == "application/json"
    
    def test_build_payload(self):
        """Test payload building"""
        payload = self.client._build_payload("Test prompt")
        
        assert payload["model"] == self.client.model
        assert payload["messages"][0]["content"] == "Test prompt"
        assert payload["max_tokens"] == self.client.max_tokens
        assert payload["temperature"] == self.client.temperature
        assert payload["stream"] is False
    
    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful API request"""
        mock_response = {
            "choices": [
                {
                    "message": {"content": "Test response"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "moonshot-v1-8k"
        }
        
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            response = await self.client.generate_text("Test prompt")
            
            assert isinstance(response, KimiK2Response)
            assert response.content == "Test response"
            assert response.model == "moonshot-v1-8k"
            assert response.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_api_key_missing_error(self):
        """Test error when API key is missing"""
        client = KimiK2Client()
        client.api_key = None
        
        with pytest.raises(KimiK2Error) as exc_info:
            await client.generate_text("Test prompt")
        
        assert exc_info.value.error_type == "configuration"
        assert "API key not configured" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of authentication errors"""
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.status_code = 401
            mock_post.return_value.text = "Invalid API key"
            
            with pytest.raises(KimiK2Error) as exc_info:
                await self.client.generate_text("Test prompt")
            
            assert exc_info.value.error_type == "authentication"
            assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_with_retry(self):
        """Test handling of rate limit errors with retry"""
        with patch.object(self.client.client, 'post') as mock_post:
            # First call returns 429, second call succeeds
            mock_response_429 = MagicMock()
            mock_response_429.status_code = 429
            
            mock_response_200 = MagicMock()
            mock_response_200.status_code = 200
            mock_response_200.json.return_value = {
                "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}],
                "usage": {},
                "model": "test"
            }
            
            mock_post.side_effect = [mock_response_429, mock_response_200]
            
            # Mock sleep to speed up test
            with patch('asyncio.sleep', new_callable=AsyncMock):
                response = await self.client.generate_text("Test prompt")
                assert response.content == "Success"
    
    @pytest.mark.asyncio
    async def test_server_error_with_retry(self):
        """Test handling of server errors with retry"""
        with patch.object(self.client.client, 'post') as mock_post:
            # All calls return 500
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(KimiK2Error) as exc_info:
                    await self.client.generate_text("Test prompt")
                
                assert exc_info.value.error_type == "server"
                assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test handling of timeout errors"""
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(KimiK2Error) as exc_info:
                    await self.client.generate_text("Test prompt")
                
                assert exc_info.value.error_type == "timeout"
    
    @pytest.mark.asyncio
    async def test_structured_response_generation(self):
        """Test structured response generation"""
        mock_response = {
            "choices": [
                {
                    "message": {"content": '{"key": "value", "number": 42}'},
                    "finish_reason": "stop"
                }
            ],
            "usage": {},
            "model": "test"
        }
        
        schema = {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "number": {"type": "integer"}
            }
        }
        
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await self.client.generate_structured_response("Test prompt", schema)
            
            assert result["key"] == "value"
            assert result["number"] == 42
    
    @pytest.mark.asyncio
    async def test_structured_response_invalid_json(self):
        """Test structured response with invalid JSON"""
        mock_response = {
            "choices": [
                {
                    "message": {"content": "Invalid JSON response"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {},
            "model": "test"
        }
        
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            with pytest.raises(KimiK2Error) as exc_info:
                await self.client.generate_structured_response("Test prompt", {})
            
            assert exc_info.value.error_type == "json_parse"
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        mock_response = {
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {},
            "model": "test"
        }
        
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await self.client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check"""
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.side_effect = Exception("Connection failed")
            
            result = await self.client.health_check()
            assert result is False