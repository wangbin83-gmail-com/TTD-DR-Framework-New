from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Kimi K2 API Configuration
    kimi_k2_api_key: Optional[str] = ""
    kimi_k2_base_url: str = "https://api.moonshot.cn/v1"
    kimi_k2_model: str = "k1-6-20241203"
    kimi_k2_max_tokens: int = 4000
    kimi_k2_temperature: float = 0.7
    
    # Google Search API Configuration
    google_search_api_key: Optional[str] = None
    google_search_engine_id: Optional[str] = None
    
    # Rate Limiting Configuration
    kimi_k2_rate_limit_requests: int = 60
    kimi_k2_rate_limit_period: int = 60  # seconds
    
    # Application Configuration
    debug: bool = True
    log_level: str = "INFO"
    
    # Development Configuration
    development_mode: bool = False
    jwt_secret_key: str = "ttdr-framework-secret-key-change-in-production"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Warning: Failed to load settings from .env file: {e}")
    settings = Settings(_env_file=None)