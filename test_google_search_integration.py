#!/usr/bin/env python3
"""
Simple integration test for Google Search API implementation.
Tests the basic functionality without requiring actual API credentials.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient, GoogleSearchResponse, GoogleSearchResult
from backend.services.dynamic_retrieval_engine import DynamicRetrievalEngine
from backend.workflow.retrieval_engine_node import retrieval_engine_node
from backend.models.core import InformationGap, GapType, Priority, SearchQuery, TTDRState, ResearchRequirements

def test_google_search_client_creation():
    """Test that GoogleSearchClient can be created"""
    client = GoogleSearchClient()
    assert client is not None
    print("✓ GoogleSearchClient created successfully")

def test_dynamic_retrieval_engine_creation():
    """Test that DynamicRetrievalEngine can be created"""
    engine = DynamicRetrievalEngine()
    assert engine is not None
    print("✓ DynamicRetrievalEngine created successfully")

async def test_mock_search_functionality():
    """Test search functionality with mocked responses"""
    client = GoogleSearchClient()
    client.api_key = "test_key"
    client.search_engine_id = "test_engine"
    
    # Mock the HTTP request
    mock_response_data = {
        "items": [
            {
                "title": "Test Result",
                "link": "https://example.com/test",
                "snippet": "This is a test search result snippet"
            }
        ],
        "searchInformation": {
            "totalResults": "1",
            "searchTime": "0.3"
        }
    }
    
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response_data
        
        response = await client.search("test query")
        
        assert isinstance(response, GoogleSearchResponse)
        assert len(response.items) == 1
        assert response.items[0].title == "Test Result"
        print("✓ Mock search functionality works")

def test_retrieval_engine_node():
    """Test the retrieval engine node with mock data"""
    # Create test state
    state = {
        "topic": "Test Topic",
        "requirements": ResearchRequirements(),
        "information_gaps": [
            InformationGap(
                id="gap1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="Need introduction content",
                priority=Priority.HIGH,
                search_queries=[
                    SearchQuery(query="test query", priority=Priority.HIGH)
                ]
            )
        ],
        "retrieved_info": [],
        "iteration_count": 1,
        "error_log": []
    }
    
    # Mock the async retrieval to avoid actual API calls
    with patch('backend.workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
        mock_retrieval.return_value = []
        
        result_state = retrieval_engine_node(state)
        
        assert "retrieved_info" in result_state
        print("✓ Retrieval engine node executes successfully")

def test_error_handling():
    """Test error handling in retrieval engine node"""
    state = {
        "topic": "Test Topic",
        "requirements": None,
        "information_gaps": [],
        "retrieved_info": [],
        "iteration_count": 1,
        "error_log": []
    }
    
    result_state = retrieval_engine_node(state)
    
    assert len(result_state["error_log"]) > 0
    assert "No information gaps" in result_state["error_log"][0]
    print("✓ Error handling works correctly")

async def main():
    """Run all tests"""
    print("Running Google Search API Integration Tests...")
    print("=" * 50)
    
    try:
        # Test basic creation
        test_google_search_client_creation()
        test_dynamic_retrieval_engine_creation()
        
        # Test async functionality
        await test_mock_search_functionality()
        
        # Test node functionality
        test_retrieval_engine_node()
        test_error_handling()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("\nGoogle Search API integration is ready for use.")
        print("To use with real API calls, configure:")
        print("- GOOGLE_SEARCH_API_KEY in your .env file")
        print("- GOOGLE_SEARCH_ENGINE_ID in your .env file")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)