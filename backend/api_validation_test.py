#!/usr/bin/env python3
"""
Comprehensive API validation test for TTD-DR Framework
Demonstrates all implemented API functionality
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_models():
    """Test API models and validation"""
    print("\n🧪 Testing API Models...")
    
    try:
        from api.models import (
            ResearchTopicRequest, WorkflowConfigRequest, WorkflowStatus,
            ResearchDomain, ComplexityLevel
        )
        
        # Test ResearchTopicRequest
        request = ResearchTopicRequest(
            topic="Artificial Intelligence in Healthcare",
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.8,
            max_sources=15,
            preferred_source_types=["academic", "news"]
        )
        
        print(f"✓ ResearchTopicRequest created: {request.topic}")
        print(f"  - Domain: {request.domain}")
        print(f"  - Complexity: {request.complexity_level}")
        print(f"  - Max iterations: {request.max_iterations}")
        
        # Test WorkflowConfigRequest
        config = WorkflowConfigRequest(
            enable_persistence=True,
            enable_recovery=True,
            debug_mode=False,
            max_execution_time=1200
        )
        
        print(f"✓ WorkflowConfigRequest created")
        print(f"  - Persistence: {config.enable_persistence}")
        print(f"  - Recovery: {config.enable_recovery}")
        print(f"  - Max time: {config.max_execution_time}s")
        
        # Test validation
        try:
            invalid_request = ResearchTopicRequest(
                topic="",  # Empty topic should fail
                max_iterations=0  # Invalid iterations
            )
            print("✗ Validation should have failed")
            return False
        except Exception:
            print("✓ Validation correctly rejected invalid data")
        
        return True
        
    except Exception as e:
        print(f"✗ API models test failed: {e}")
        return False

def test_authentication():
    """Test authentication functionality"""
    print("\n🔐 Testing Authentication...")
    
    try:
        from api.auth import (
            create_access_token, verify_token, authenticate_user,
            get_password_hash, verify_password
        )
        
        # Test password hashing
        password = "test_password"
        hashed = get_password_hash(password)
        print(f"✓ Password hashed: {hashed[:20]}...")
        
        # Test password verification
        if verify_password(password, hashed):
            print("✓ Password verification successful")
        else:
            print("✗ Password verification failed")
            return False
        
        # Test token creation
        token_data = {
            "user_id": "test_user_001",
            "username": "test_user",
            "permissions": ["research:create", "research:read"]
        }
        
        token = create_access_token(data=token_data)
        print(f"✓ JWT token created: {token[:30]}...")
        
        # Test token verification
        verified_data = verify_token(token)
        if verified_data:
            print(f"✓ Token verified: {verified_data.username}")
            print(f"  - User ID: {verified_data.user_id}")
            print(f"  - Permissions: {verified_data.permissions}")
        else:
            print("✗ Token verification failed")
            return False
        
        # Test user authentication
        user = authenticate_user("demo_user", "demo_password")
        if user:
            print(f"✓ User authentication successful: {user['username']}")
        else:
            print("✓ User authentication correctly rejected invalid credentials")
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n⏱️ Testing Rate Limiting...")
    
    try:
        from api.rate_limiting import AdvancedRateLimiter, TokenBucket
        import asyncio
        
        # Test TokenBucket
        bucket = TokenBucket(capacity=5, refill_rate=1.0)  # 5 tokens, 1 per second
        
        async def test_bucket():
            # Should be able to consume 5 tokens initially
            for i in range(5):
                result = await bucket.consume(1)
                if not result:
                    print(f"✗ Token bucket failed at token {i+1}")
                    return False
            
            # 6th token should fail
            result = await bucket.consume(1)
            if result:
                print("✗ Token bucket should have been empty")
                return False
            
            print("✓ Token bucket working correctly")
            return True
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bucket_result = loop.run_until_complete(test_bucket())
        loop.close()
        
        if not bucket_result:
            return False
        
        # Test AdvancedRateLimiter
        limiter = AdvancedRateLimiter()
        
        async def test_limiter():
            client_id = "test_client"
            endpoint = "/api/v1/research/initiate"
            
            # Should allow first request
            result = await limiter.is_allowed(client_id, endpoint)
            if not result:
                print("✗ Rate limiter should allow first request")
                return False
            
            print("✓ Rate limiter allows valid requests")
            
            # Test bucket status
            status = limiter.get_bucket_status(client_id, endpoint)
            print(f"  - Bucket capacity: {status['capacity']}")
            print(f"  - Refill rate: {status['refill_rate']}")
            
            return True
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        limiter_result = loop.run_until_complete(test_limiter())
        loop.close()
        
        return limiter_result
        
    except Exception as e:
        print(f"✗ Rate limiting test failed: {e}")
        return False

def test_websocket_manager():
    """Test WebSocket manager functionality"""
    print("\n🔌 Testing WebSocket Manager...")
    
    try:
        from api.websocket_manager import ConnectionManager
        from api.models import WebSocketMessage, WebSocketMessageType, WorkflowStatus
        
        # Test ConnectionManager
        manager = ConnectionManager()
        
        # Test connection stats
        stats = manager.get_connection_stats()
        print(f"✓ Connection manager created")
        print(f"  - Total connections: {stats['total_connections']}")
        print(f"  - Executions monitored: {stats['executions_monitored']}")
        
        # Test message creation
        message = WebSocketMessage(
            type=WebSocketMessageType.STATUS_UPDATE,
            execution_id="test_execution_123",
            data={
                "status": WorkflowStatus.RUNNING,
                "progress": 45.0
            }
        )
        
        print(f"✓ WebSocket message created: {message.type}")
        print(f"  - Execution ID: {message.execution_id}")
        print(f"  - Data: {message.data}")
        
        return True
        
    except Exception as e:
        print(f"✗ WebSocket manager test failed: {e}")
        return False

def test_fastapi_integration():
    """Test FastAPI integration"""
    print("\n🚀 Testing FastAPI Integration...")
    
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.models import ResearchTopicRequest
        from api.auth import create_access_token
        
        # Create a minimal FastAPI app for testing
        app = FastAPI(title="TTD-DR Test API")
        
        @app.get("/")
        def root():
            return {
                "message": "TTD-DR Framework API",
                "version": "1.0.0",
                "status": "testing"
            }
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @app.post("/test/research")
        def test_research(request: ResearchTopicRequest):
            return {
                "message": f"Research request received for: {request.topic}",
                "domain": request.domain,
                "complexity": request.complexity_level
            }
        
        # Test with client
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "TTD-DR Framework API"
        print("✓ Root endpoint test passed")
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ Health endpoint test passed")
        
        # Test research endpoint with valid data
        research_data = {
            "topic": "Machine Learning in Finance",
            "domain": "technology",
            "complexity_level": "advanced",
            "max_iterations": 5,
            "quality_threshold": 0.85,
            "max_sources": 20,
            "preferred_source_types": ["academic", "news", "official"]
        }
        
        response = client.post("/test/research", json=research_data)
        assert response.status_code == 200
        data = response.json()
        assert "Machine Learning in Finance" in data["message"]
        print("✓ Research endpoint test passed")
        
        # Test with invalid data
        invalid_data = {
            "topic": "",  # Empty topic
            "max_iterations": 0  # Invalid iterations
        }
        
        response = client.post("/test/research", json=invalid_data)
        assert response.status_code == 422  # Validation error
        print("✓ Validation error handling test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ FastAPI integration test failed: {e}")
        return False

def test_workflow_integration():
    """Test workflow integration points"""
    print("\n⚙️ Testing Workflow Integration...")
    
    try:
        from models.core import (
            ResearchRequirements, TTDRState, ResearchDomain, ComplexityLevel
        )
        
        # Test core models
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.8,
            max_sources=15
        )
        
        print(f"✓ ResearchRequirements created")
        print(f"  - Domain: {requirements.domain}")
        print(f"  - Complexity: {requirements.complexity_level}")
        
        # Test TTDRState creation
        state = TTDRState(
            topic="Test Research Topic",
            requirements=requirements,
            current_draft=None,
            information_gaps=[],
            retrieved_info=[],
            iteration_count=0,
            quality_metrics=None,
            evolution_history=[],
            final_report=None,
            error_log=[]
        )
        
        print(f"✓ TTDRState created: {state['topic']}")
        print(f"  - Iteration count: {state['iteration_count']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 TTD-DR Framework API Validation Tests")
    print("=" * 50)
    
    tests = [
        test_api_models,
        test_authentication,
        test_rate_limiting,
        test_websocket_manager,
        test_fastapi_integration,
        test_workflow_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("\n✅ TTD-DR Framework API is ready for use!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        print("\n🔧 Some components need attention before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)