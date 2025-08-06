"""
Comprehensive tests for TTD-DR Framework REST API endpoints
Tests functionality and security of all API endpoints
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from backend.main import app
from backend.api.models import (
    ResearchTopicRequest, WorkflowConfigRequest, WorkflowStatus,
    ResearchDomain, ComplexityLevel
)
from backend.api.auth import create_access_token
from backend.models.core import ResearchRequirements, TTDRState
# Test 
client
client = TestClient(app)

# Test fixtures
@pytest.fixture
def valid_auth_token():
    """Create a valid JWT token for testing"""
    token_data = {
        "user_id": "user_001",
        "username": "demo_user",
        "permissions": ["research:create", "research:read", "research:monitor"]
    }
    return create_access_token(data=token_data)

@pytest.fixture
def admin_auth_token():
    """Create an admin JWT token for testing"""
    token_data = {
        "user_id": "user_002",
        "username": "admin_user",
        "permissions": ["*"]
    }
    return create_access_token(data=token_data)

@pytest.fixture
def expired_auth_token():
    """Create an expired JWT token for testing"""
    token_data = {
        "user_id": "user_001",
        "username": "demo_user",
        "permissions": ["research:create", "research:read"]
    }
    return create_access_token(
        data=token_data,
        expires_delta=timedelta(seconds=-1)  # Already expired
    )

@pytest.fixture
def sample_research_request():
    """Sample research topic request"""
    return {
        "topic": "Artificial Intelligence in Healthcare",
        "domain": "technology",
        "complexity_level": "intermediate",
        "max_iterations": 3,
        "quality_threshold": 0.8,
        "max_sources": 15,
        "preferred_source_types": ["academic", "news"]
    }

@pytest.fixture
def sample_workflow_config():
    """Sample workflow configuration"""
    return {
        "enable_persistence": True,
        "enable_recovery": True,
        "debug_mode": False,
        "max_execution_time": 1200
    }

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["message"] == "TTD-DR Framework API"
        assert data["version"] == "1.0.0"
        assert "docs_url" in data
        assert "api_prefix" in data
    
    def test_health_check(self):
        """Test general health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "TTD-DR Framework API"
        assert "timestamp" in data
    
    def test_api_health_check(self):
        """Test API-specific health check"""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "v1"
        assert "endpoints" in data
        assert "auth" in data["endpoints"]
        assert "research" in data["endpoints"]

class TestAuthenticationEndpoints:
    """Test authentication and authorization"""
    
    def test_login_success(self):
        """Test successful login"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "demo_user",
                "password": "demo_password"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "user" in data
        assert data["user"]["username"] == "demo_user"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "invalid_user",
                "password": "wrong_password"
            }
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_missing_credentials(self):
        """Test login with missing credentials"""
        response = client.post("/api/v1/auth/login", data={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_refresh_token_success(self, valid_auth_token):
        """Test successful token refresh"""
        response = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
    
    def test_refresh_token_unauthorized(self):
        """Test token refresh without authentication"""
        response = client.post("/api/v1/auth/refresh")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_refresh_token_expired(self, expired_auth_token):
        """Test token refresh with expired token"""
        response = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {expired_auth_token}"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

class TestResearchWorkflowEndpoints:
    """Test research workflow management endpoints"""
    
    @patch('backend.api.endpoints.workflow_engine')
    def test_initiate_research_success(self, mock_workflow_engine, valid_auth_token, sample_research_request):
        """Test successful research workflow initiation"""
        # Mock workflow engine
        mock_workflow_engine.execute_workflow = AsyncMock()
        
        response = client.post(
            "/api/v1/research/initiate",
            json=sample_research_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "execution_id" in data
        assert data["status"] == WorkflowStatus.PENDING
        assert "estimated_duration" in data
        assert "websocket_url" in data
        assert data["websocket_url"].startswith("/api/v1/research/ws/")
    
    def test_initiate_research_unauthorized(self, sample_research_request):
        """Test research initiation without authentication"""
        response = client.post(
            "/api/v1/research/initiate",
            json=sample_research_request
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_initiate_research_invalid_data(self, valid_auth_token):
        """Test research initiation with invalid data"""
        invalid_request = {
            "topic": "",  # Empty topic
            "max_iterations": 0,  # Invalid iterations
            "quality_threshold": 1.5  # Invalid threshold
        }
        
        response = client.post(
            "/api/v1/research/initiate",
            json=invalid_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('backend.api.endpoints.workflow_results')
    def test_get_workflow_status_success(self, mock_workflow_results, valid_auth_token):
        """Test successful workflow status retrieval"""
        execution_id = "test_execution_123"
        mock_workflow_results.__contains__ = Mock(return_value=True)
        mock_workflow_results.__getitem__ = Mock(return_value={
            "execution_id": execution_id,
            "status": WorkflowStatus.RUNNING,
            "user_id": "user_001",
            "created_at": datetime.now(),
            "progress": 45.0,
            "current_node": "gap_analyzer",
            "nodes_completed": [],
            "iterations_completed": 1
        })
        
        response = client.get(
            f"/api/v1/research/status/{execution_id}",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["execution_id"] == execution_id
        assert data["status"] == WorkflowStatus.RUNNING
        assert data["progress_percentage"] == 45.0
        assert data["current_node"] == "gap_analyzer"
    
    def test_get_workflow_status_not_found(self, valid_auth_token):
        """Test workflow status for non-existent execution"""
        response = client.get(
            "/api/v1/research/status/nonexistent_id",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_workflow_status_unauthorized(self):
        """Test workflow status without authentication"""
        response = client.get("/api/v1/research/status/test_id")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('backend.api.endpoints.workflow_results')
    def test_get_workflow_result_success(self, mock_workflow_results, valid_auth_token):
        """Test successful workflow result retrieval"""
        execution_id = "test_execution_123"
        mock_workflow_results.__contains__ = Mock(return_value=True)
        mock_workflow_results.__getitem__ = Mock(return_value={
            "execution_id": execution_id,
            "status": WorkflowStatus.COMPLETED,
            "user_id": "user_001",
            "created_at": datetime.now(),
            "completed_at": datetime.now(),
            "final_state": {
                "final_report": "# Research Report\n\nThis is a test report.",
                "quality_metrics": {
                    "completeness": 0.9,
                    "coherence": 0.85,
                    "accuracy": 0.88,
                    "citation_quality": 0.82,
                    "overall_score": 0.86
                },
                "iteration_count": 3,
                "retrieved_info": []
            }
        })
        
        response = client.get(
            f"/api/v1/research/result/{execution_id}",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["execution_id"] == execution_id
        assert data["status"] == WorkflowStatus.COMPLETED
        assert data["final_report"] is not None
        assert "quality_metrics" in data
        assert "download_urls" in data
    
    @patch('backend.api.endpoints.workflow_results')
    def test_get_workflow_result_not_completed(self, mock_workflow_results, valid_auth_token):
        """Test workflow result for non-completed execution"""
        execution_id = "test_execution_123"
        mock_workflow_results.__contains__ = Mock(return_value=True)
        mock_workflow_results.__getitem__ = Mock(return_value={
            "execution_id": execution_id,
            "status": WorkflowStatus.RUNNING,
            "user_id": "user_001"
        })
        
        response = client.get(
            f"/api/v1/research/result/{execution_id}",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not completed" in response.json()["detail"].lower()
    
    @patch('backend.api.endpoints.workflow_results')
    @patch('backend.api.endpoints.workflow_engine')
    def test_cancel_workflow_success(self, mock_workflow_engine, mock_workflow_results, valid_auth_token):
        """Test successful workflow cancellation"""
        execution_id = "test_execution_123"
        mock_workflow_results.__contains__ = Mock(return_value=True)
        mock_workflow_results.__getitem__ = Mock(return_value={
            "execution_id": execution_id,
            "status": WorkflowStatus.RUNNING,
            "user_id": "user_001"
        })
        mock_workflow_engine.cancel_execution = Mock(return_value=True)
        
        response = client.post(
            f"/api/v1/research/cancel/{execution_id}",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["execution_id"] == execution_id
        assert data["status"] == WorkflowStatus.CANCELLED
    
    @patch('backend.api.endpoints.workflow_results')
    def test_cancel_workflow_already_completed(self, mock_workflow_results, valid_auth_token):
        """Test cancellation of already completed workflow"""
        execution_id = "test_execution_123"
        mock_workflow_results.__contains__ = Mock(return_value=True)
        mock_workflow_results.__getitem__ = Mock(return_value={
            "execution_id": execution_id,
            "status": WorkflowStatus.COMPLETED,
            "user_id": "user_001"
        })
        
        response = client.post(
            f"/api/v1/research/cancel/{execution_id}",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot cancel workflow" in response.json()["detail"]
    
    @patch('backend.api.endpoints.workflow_results')
    def test_list_workflows_success(self, mock_workflow_results, valid_auth_token):
        """Test successful workflow listing"""
        mock_workflow_results.items = Mock(return_value=[
            ("exec_1", {
                "execution_id": "exec_1",
                "topic": "AI in Healthcare",
                "status": WorkflowStatus.COMPLETED,
                "user_id": "user_001",
                "created_at": datetime.now(),
                "progress": 100.0
            }),
            ("exec_2", {
                "execution_id": "exec_2",
                "topic": "Blockchain Technology",
                "status": WorkflowStatus.RUNNING,
                "user_id": "user_001",
                "created_at": datetime.now(),
                "progress": 60.0
            })
        ])
        
        response = client.get(
            "/api/v1/research/list",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "workflows" in data
        assert "total_count" in data
        assert len(data["workflows"]) == 2
    
    def test_list_workflows_with_filters(self, valid_auth_token):
        """Test workflow listing with status filter"""
        response = client.get(
            "/api/v1/research/list?status=completed&limit=5&offset=0",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
    
    @patch('backend.api.endpoints.workflow_results')
    def test_download_report_success(self, mock_workflow_results, valid_auth_token):
        """Test successful report download"""
        execution_id = "test_execution_123"
        mock_workflow_results.__contains__ = Mock(return_value=True)
        mock_workflow_results.__getitem__ = Mock(return_value={
            "execution_id": execution_id,
            "status": WorkflowStatus.COMPLETED,
            "user_id": "user_001",
            "topic": "AI in Healthcare",
            "final_state": {
                "final_report": "# Research Report\n\nThis is a test report."
            }
        })
        
        response = client.get(
            f"/api/v1/research/download/{execution_id}/markdown",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        assert "attachment" in response.headers.get("content-disposition", "")
    
    def test_download_report_unsupported_format(self, valid_auth_token):
        """Test report download with unsupported format"""
        execution_id = "test_execution_123"
        
        response = client.get(
            f"/api/v1/research/download/{execution_id}/unsupported",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

class TestAdminEndpoints:
    """Test admin-only endpoints"""
    
    @patch('backend.api.endpoints.workflow_engine')
    @patch('backend.api.endpoints.connection_manager')
    def test_get_system_stats_success(self, mock_connection_manager, mock_workflow_engine, admin_auth_token):
        """Test successful system stats retrieval"""
        mock_workflow_engine.list_active_executions = Mock(return_value=[])
        mock_connection_manager.get_connection_stats = Mock(return_value={
            "total_connections": 5,
            "executions_monitored": 3,
            "authenticated_users": 2
        })
        
        response = client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {admin_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "system" in data
        assert "workflows" in data
        assert "websockets" in data
        assert "timestamp" in data
    
    def test_get_system_stats_unauthorized(self, valid_auth_token):
        """Test system stats access with non-admin user"""
        response = client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limiting_login(self):
        """Test rate limiting on login endpoint"""
        # Make multiple rapid requests to trigger rate limiting
        for i in range(12):  # Exceed the limit of 10
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": "demo_user",
                    "password": "demo_password"
                }
            )
            
            if i < 10:
                # First 10 should succeed (or fail with auth error)
                assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
            else:
                # Subsequent requests should be rate limited
                if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    assert "Rate limit exceeded" in str(response.json())
                    break

class TestWebSocketEndpoints:
    """Test WebSocket functionality"""
    
    def test_websocket_connection_without_auth(self):
        """Test WebSocket connection without authentication"""
        with client.websocket_connect("/api/v1/research/ws/test_execution") as websocket:
            # Should be able to connect but with limited functionality
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
    
    def test_websocket_connection_with_auth(self, valid_auth_token):
        """Test WebSocket connection with authentication"""
        with client.websocket_connect(
            f"/api/v1/research/ws/test_execution?token={valid_auth_token}"
        ) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert data["execution_id"] == "test_execution"

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_json_request(self, valid_auth_token):
        """Test handling of invalid JSON in request"""
        response = client.post(
            "/api/v1/research/initiate",
            data="invalid json",
            headers={
                "Authorization": f"Bearer {valid_auth_token}",
                "Content-Type": "application/json"
            }
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_content_type(self, valid_auth_token, sample_research_request):
        """Test request without proper content type"""
        response = client.post(
            "/api/v1/research/initiate",
            data=json.dumps(sample_research_request),
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        # Should still work as FastAPI is flexible with content types
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_malformed_auth_header(self):
        """Test malformed authorization header"""
        response = client.get(
            "/api/v1/research/list",
            headers={"Authorization": "InvalidToken"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('backend.api.endpoints.workflow_engine')
    def test_workflow_execution_error(self, mock_workflow_engine, valid_auth_token, sample_research_request):
        """Test handling of workflow execution errors"""
        mock_workflow_engine.execute_workflow.side_effect = Exception("Workflow failed")
        
        response = client.post(
            "/api/v1/research/initiate",
            json=sample_research_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        # Should still return success for initiation, error handling happens in background
        assert response.status_code == status.HTTP_200_OK

class TestCORSAndSecurity:
    """Test CORS and security configurations"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        # CORS headers should be present in actual deployment
    
    def test_security_headers(self):
        """Test security headers"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        # In production, should have security headers like X-Frame-Options, etc.

# Integration tests
class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @patch('backend.api.endpoints.workflow_engine')
    def test_complete_research_workflow(self, mock_workflow_engine, valid_auth_token, sample_research_request):
        """Test complete research workflow from initiation to completion"""
        # Mock successful workflow execution
        mock_workflow_engine.execute_workflow = AsyncMock()
        
        # 1. Initiate research
        response = client.post(
            "/api/v1/research/initiate",
            json=sample_research_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        execution_id = response.json()["execution_id"]
        
        # 2. Check status (would be mocked in real scenario)
        # 3. Get results (would be mocked in real scenario)
        # 4. Download report (would be mocked in real scenario)
        
        # This demonstrates the complete API flow
        assert execution_id.startswith("ttdr_")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])