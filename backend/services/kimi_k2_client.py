import asyncio
import time
from typing import Dict, List, Optional, Any
import httpx
import json
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

logger = logging.getLogger(__name__)

class KimiK2Response(BaseModel):
    """Standardized response from Kimi K2 API"""
    content: str
    usage: Dict[str, int] = {}
    model: str = ""
    finish_reason: str = ""
    
class KimiK2Error(Exception):
    """Custom exception for Kimi K2 API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: str = "unknown"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self.requests.append(now)

class KimiK2Client:
    """Client for interacting with Kimi K2 API"""
    
    def __init__(self):
        self.api_key = settings.kimi_k2_api_key
        self.base_url = settings.kimi_k2_base_url
        self.model = settings.kimi_k2_model
        self.max_tokens = settings.kimi_k2_max_tokens
        self.temperature = settings.kimi_k2_temperature
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            settings.kimi_k2_rate_limit_requests,
            settings.kimi_k2_rate_limit_period
        )
        
        # HTTP client with optimized timeout and connection settings for Kimi K2
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,    # Connection timeout
                read=120.0,      # Read timeout - increased for long text generation
                write=30.0,      # Write timeout
                pool=10.0        # Pool timeout
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0
            ),
            follow_redirects=True,
            verify=True
        )
        
        if not self.api_key:
            logger.warning("Kimi K2 API key not configured")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build request payload"""
        return {
            "model": kwargs.get("model", self.model),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }
    
    async def _make_request(self, payload: Dict[str, Any], retries: int = 5) -> Dict[str, Any]:
        """Make HTTP request with intelligent retry logic for Kimi K2"""
        await self.rate_limiter.acquire()
        
        for attempt in range(retries):
            try:
                # Log attempt for debugging
                if attempt > 0:
                    logger.info(f"Kimi K2 API request attempt {attempt + 1}/{retries}")
                
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload
                )
                
                if response.status_code == 200:
                    logger.debug(f"Kimi K2 API request successful on attempt {attempt + 1}")
                    return response.json()
                    
                elif response.status_code == 429:  # Rate limited
                    # For rate limiting, use longer backoff
                    wait_time = min(60, (2 ** attempt) * 3)  # Cap at 60 seconds
                    logger.warning(f"Kimi K2 rate limited, waiting {wait_time} seconds (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                    
                elif response.status_code == 401:
                    # Authentication errors should not be retried
                    raise KimiK2Error("Invalid API key", response.status_code, "authentication")
                    
                elif response.status_code == 400:
                    # Bad request errors should not be retried
                    error_detail = response.text
                    raise KimiK2Error(f"Bad request: {error_detail}", response.status_code, "bad_request")
                    
                elif response.status_code >= 500:
                    # Server errors can be retried
                    if attempt < retries - 1:
                        wait_time = min(30, 2 ** attempt)  # Cap at 30 seconds
                        logger.warning(f"Kimi K2 server error {response.status_code}, retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise KimiK2Error(f"Server error: {response.status_code}", response.status_code, "server")
                        
                elif response.status_code == 502 or response.status_code == 503:
                    # Gateway errors, retry with longer delay
                    if attempt < retries - 1:
                        wait_time = min(45, (2 ** attempt) * 5)  # Longer delay for gateway errors
                        logger.warning(f"Kimi K2 gateway error {response.status_code}, retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise KimiK2Error(f"Gateway error: {response.status_code}", response.status_code, "gateway")
                        
                else:
                    # Other HTTP errors
                    error_detail = response.text
                    logger.error(f"Kimi K2 API error {response.status_code}: {error_detail}")
                    raise KimiK2Error(f"API error: {error_detail}", response.status_code, "api")
                    
            except httpx.TimeoutException as e:
                # Timeout errors - retry with exponential backoff
                if attempt < retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2)  # Longer delay for timeouts
                    logger.warning(f"Kimi K2 request timeout ({e}), retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Kimi K2 request timeout after {retries} attempts")
                    raise KimiK2Error("Request timeout after multiple attempts", None, "timeout")
                    
            except httpx.ConnectTimeout:
                # Connection timeout - retry with shorter delay
                if attempt < retries - 1:
                    wait_time = min(20, 2 ** attempt)
                    logger.warning(f"Kimi K2 connection timeout, retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise KimiK2Error("Connection timeout", None, "connection_timeout")
                    
            except httpx.ReadTimeout:
                # Read timeout - this is common for long text generation
                if attempt < retries - 1:
                    wait_time = min(30, (2 ** attempt) * 3)  # Longer delay for read timeouts
                    logger.warning(f"Kimi K2 read timeout (long generation), retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise KimiK2Error("Read timeout - response too long", None, "read_timeout")
                    
            except httpx.RequestError as e:
                # Network errors
                if attempt < retries - 1:
                    wait_time = min(25, 2 ** attempt)
                    logger.warning(f"Kimi K2 network error ({e}), retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Kimi K2 network error after {retries} attempts: {e}")
                    raise KimiK2Error(f"Network error: {e}", None, "network")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error in Kimi K2 request: {e}")
                if attempt < retries - 1:
                    wait_time = 5
                    logger.warning(f"Unexpected error, retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise KimiK2Error(f"Unexpected error: {e}", None, "unexpected")
        
        raise KimiK2Error("Max retries exceeded", None, "retry_exhausted")
    
    async def generate_text(self, prompt: str, **kwargs) -> KimiK2Response:
        """Generate text using Kimi K2 model"""
        if not self.api_key:
            raise KimiK2Error("Kimi K2 API key not configured", None, "configuration")
        
        try:
            payload = self._build_payload(prompt, **kwargs)
            response_data = await self._make_request(payload)
            
            # Parse response
            if "choices" not in response_data or not response_data["choices"]:
                raise KimiK2Error("Invalid response format", None, "response_format")
            
            choice = response_data["choices"][0]
            content = choice.get("message", {}).get("content", "")
            
            return KimiK2Response(
                content=content,
                usage=response_data.get("usage", {}),
                model=response_data.get("model", self.model),
                finish_reason=choice.get("finish_reason", "")
            )
            
        except KimiK2Error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_text: {e}")
            raise KimiK2Error(f"Unexpected error: {e}", None, "unexpected")
    
    async def generate_structured_response(self, prompt: str, response_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured response that conforms to a schema"""
        # Add schema instructions to prompt
        schema_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(response_schema, indent=2)}

Ensure your response is valid JSON and follows the schema exactly. Do not wrap the JSON in markdown code blocks.
"""
        
        response = await self.generate_text(schema_prompt, **kwargs)
        
        try:
            # Clean the response content to handle markdown code blocks
            content = response.content.strip()
            
            # Remove markdown code block markers if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            elif content.startswith('```'):
                content = content[3:]   # Remove ```
            
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            
            content = content.strip()
            
            # Try to parse as JSON
            structured_data = json.loads(content)
            return structured_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured response as JSON: {e}")
            logger.error(f"Response content: {response.content}")
            logger.error(f"Cleaned content: {content}")
            raise KimiK2Error(f"Invalid JSON response: {e}", None, "json_parse")
    
    async def health_check(self) -> bool:
        """Check if the API is accessible"""
        try:
            response = await self.generate_text("Hello", max_tokens=10)
            return bool(response.content)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Singleton instance for global use
kimi_k2_client = KimiK2Client()