#!/usr/bin/env python3
"""
Simple API test to verify basic functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that we can import the basic modules"""
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
        
        from fastapi.testclient import TestClient
        print("✓ TestClient imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_api_models():
    """Test API models"""
    try:
        # Test basic model imports
        from api.models import ResearchTopicRequest, WorkflowStatus
        print("✓ API models imported successfully")
        
        # Test model creation
        request = ResearchTopicRequest(
            topic="Test topic",
            domain="technology",
            complexity_level="intermediate"
        )
        print(f"✓ ResearchTopicRequest created: {request.topic}")
        
        return True
    except Exception as e:
        print(f"✗ API models error: {e}")
        return False

def test_auth_module():
    """Test authentication module"""
    try:
        from api.auth import create_access_token, verify_token
        print("✓ Auth module imported successfully")
        
        # Test token creation
        token = create_access_token(data={"user_id": "test", "username": "test"})
        print(f"✓ Token created: {token[:20]}...")
        
        # Test token verification
        token_data = verify_token(token)
        if token_data:
            print(f"✓ Token verified: {token_data.username}")
        else:
            print("✗ Token verification failed")
            
        return True
    except Exception as e:
        print(f"✗ Auth module error: {e}")
        return False

def test_basic_app():
    """Test basic FastAPI app creation"""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Create a simple app
        app = FastAPI(title="Test API")
        
        @app.get("/")
        def root():
            return {"message": "Test API"}
        
        @app.get("/health")
        def health():
            return {"status": "healthy"}
        
        # Test with client
        client = TestClient(app)
        
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Test API"
        print("✓ Basic API test passed")
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✓ Health endpoint test passed")
        
        return True
    except Exception as e:
        print(f"✗ Basic app test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Running TTD-DR Framework API Tests")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_api_models,
        test_auth_module,
        test_basic_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)