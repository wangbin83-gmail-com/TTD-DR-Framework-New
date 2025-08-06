"""
Unit tests for Retrieval Engine Node.
Tests LangGraph integration, error handling, and workflow functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.retrieval_engine_node import (
    retrieval_engine_node, test_retrieval_engine_health,
    _run_async_retrieval, _filter_retrieved_info, _create_fallback_retrieved_info
)
from models.core import (
    TTDRState, InformationGap, GapType, Priority, SearchQuery,
    RetrievedInfo, Source, ResearchRequirements, ComplexityLevel, ResearchDomain
)
from services.google_search_client import GoogleSearchError

class TestRetrievalEngineNode:
    """Test the main retrieval engine node function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.base_state = {
            "topic": "Machine Learning Applications",
            "requirements": ResearchRequirements(
                domain=ResearchDomain.TECHNOLOGY,
                complexity_level=ComplexityLevel.INTERMEDIATE,
                max_sources=20,
                quality_threshold=0.8
            ),
            "current_draft": None,
            "information_gaps": [
                InformationGap(
                    id="gap1",
                    section_id="introduction",
                    gap_type=GapType.CONTENT,
                    description="Need overview of machine learning applications",
                    priority=Priority.HIGH,
                    search_queries=[
                        SearchQuery(query="machine learning applications overview", priority=Priority.HIGH)
                    ]
                ),
                InformationGap(
                    id="gap2",
                    section_id="methodology",
                    gap_type=GapType.EVIDENCE,
                    description="Need research methodology examples",
                    priority=Priority.MEDIUM,
                    search_queries=[
                        SearchQuery(query="machine learning research methodology", priority=Priority.MEDIUM)
                    ]
                )
            ],
            "retrieved_info": [],
            "iteration_count": 1,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
    
    def test_retrieval_engine_node_success(self):
        """Test successful retrieval engine node execution"""
        # Mock retrieved information
        mock_retrieved_info = [
            RetrievedInfo(
                source=Source(
                    url="https://example.com/ml-apps",
                    title="Machine Learning Applications",
                    domain="example.com",
                    credibility_score=0.8
                ),
                content="Machine learning has various applications in healthcare, finance, and technology.",
                relevance_score=0.9,
                credibility_score=0.8,
                gap_id="gap1"
            ),
            RetrievedInfo(
                source=Source(
                    url="https://research.com/methodology",
                    title="ML Research Methods",
                    domain="research.com",
                    credibility_score=0.7
                ),
                content="Research methodology in machine learning involves data collection, model training, and evaluation.",
                relevance_score=0.8,
                credibility_score=0.7,
                gap_id="gap2"
            )
        ]
        
        with patch('workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
            mock_retrieval.return_value = mock_retrieved_info
            
            result_state = retrieval_engine_node(self.base_state)
            
            assert "retrieved_info" in result_state
            assert len(result_state["retrieved_info"]) == 2
            assert result_state["retrieved_info"][0].gap_id == "gap1"
            assert result_state["retrieved_info"][1].gap_id == "gap2"
    
    def test_retrieval_engine_node_no_gaps(self):
        """Test node execution with no information gaps"""
        state_no_gaps = {**self.base_state, "information_gaps": []}
        
        result_state = retrieval_engine_node(state_no_gaps)
        
        assert result_state["retrieved_info"] == []
        assert len(result_state["error_log"]) == 1
        assert "No information gaps" in result_state["error_log"][0]
    
    def test_retrieval_engine_node_google_search_error(self):
        """Test handling of Google Search API errors"""
        with patch('workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
            mock_retrieval.side_effect = GoogleSearchError("Daily quota exceeded", 403, "quota_exceeded")
            
            result_state = retrieval_engine_node(self.base_state)
            
            # Should have fallback retrieved info
            assert len(result_state["retrieved_info"]) > 0
            assert len(result_state["error_log"]) == 1
            assert "quota exceeded" in result_state["error_log"][0]
    
    def test_retrieval_engine_node_configuration_error(self):
        """Test handling of configuration errors"""
        with patch('workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
            mock_retrieval.side_effect = GoogleSearchError("API not configured", None, "configuration")
            
            result_state = retrieval_engine_node(self.base_state)
            
            # Should have fallback retrieved info
            assert len(result_state["retrieved_info"]) > 0
            assert len(result_state["error_log"]) == 1
            assert "not configured" in result_state["error_log"][0]
    
    def test_retrieval_engine_node_unexpected_error(self):
        """Test handling of unexpected errors"""
        with patch('workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
            mock_retrieval.side_effect = Exception("Unexpected error occurred")
            
            result_state = retrieval_engine_node(self.base_state)
            
            # Should have fallback retrieved info
            assert len(result_state["retrieved_info"]) > 0
            assert len(result_state["error_log"]) == 1
            assert "Unexpected error" in result_state["error_log"][0]
    
    def test_retrieval_engine_node_max_sources_calculation(self):
        """Test calculation of max results per gap based on requirements"""
        # Test with specific max_sources
        state_with_limits = {
            **self.base_state,
            "requirements": ResearchRequirements(max_sources=10)
        }
        
        mock_retrieved_info = []
        
        with patch('workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
            mock_retrieval.return_value = mock_retrieved_info
            
            retrieval_engine_node(state_with_limits)
            
            # Should calculate max_results_per_gap = 10 / 2 gaps = 5
            mock_retrieval.assert_called_once()
            args = mock_retrieval.call_args[0]
            assert args[2] == 5  # max_results_per_gap
    
    def test_retrieval_engine_node_no_requirements(self):
        """Test node execution without requirements"""
        state_no_requirements = {**self.base_state, "requirements": None}
        
        mock_retrieved_info = []
        
        with patch('workflow.retrieval_engine_node._run_async_retrieval') as mock_retrieval:
            mock_retrieval.return_value = mock_retrieved_info
            
            retrieval_engine_node(state_no_requirements)
            
            # Should use default max_results_per_gap = 5
            mock_retrieval.assert_called_once()
            args = mock_retrieval.call_args[0]
            assert args[2] == 5  # max_results_per_gap

class TestAsyncRetrievalRunner:
    """Test the async retrieval runner utility"""
    
    @pytest.mark.asyncio
    async def test_run_async_retrieval_success(self):
        """Test successful async retrieval execution"""
        from services.dynamic_retrieval_engine import DynamicRetrievalEngine
        
        mock_engine = Mock(spec=DynamicRetrievalEngine)
        mock_engine.retrieve_information = AsyncMock()
        mock_engine.retrieve_information.return_value = []
        
        gaps = [Mock()]
        max_results = 5
        
        result = _run_async_retrieval(mock_engine, gaps, max_results)
        
        assert result == []
        mock_engine.retrieve_information.assert_called_once_with(gaps, max_results)
    
    def test_run_async_retrieval_with_running_loop(self):
        """Test async retrieval when event loop is already running"""
        from services.dynamic_retrieval_engine import DynamicRetrievalEngine
        
        mock_engine = Mock(spec=DynamicRetrievalEngine)
        
        # Mock asyncio.run to raise RuntimeError (simulating running event loop)
        with patch('asyncio.run') as mock_run:
            mock_run.side_effect = RuntimeError("asyncio.run() cannot be called from a running event loop")
            
            with patch('asyncio.new_event_loop') as mock_new_loop:
                mock_loop = Mock()
                mock_new_loop.return_value = mock_loop
                mock_loop.run_until_complete.return_value = []
                
                with patch('asyncio.set_event_loop') as mock_set_loop:
                    result = _run_async_retrieval(mock_engine, [], 5)
                    
                    assert result == []
                    mock_new_loop.assert_called_once()
                    mock_set_loop.assert_called_once_with(mock_loop)
                    mock_loop.close.assert_called_once()

class TestRetrievedInfoFiltering:
    """Test retrieved information filtering functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.high_quality_info = RetrievedInfo(
            source=Source(url="https://example.com", title="Test", domain="example.com", credibility_score=0.8),
            content="High quality content",
            relevance_score=0.9,
            credibility_score=0.8
        )
        
        self.medium_quality_info = RetrievedInfo(
            source=Source(url="https://example.com", title="Test", domain="example.com", credibility_score=0.6),
            content="Medium quality content",
            relevance_score=0.6,
            credibility_score=0.6
        )
        
        self.low_quality_info = RetrievedInfo(
            source=Source(url="https://example.com", title="Test", domain="example.com", credibility_score=0.3),
            content="Low quality content",
            relevance_score=0.3,
            credibility_score=0.3
        )
    
    def test_filter_retrieved_info_with_requirements(self):
        """Test filtering with specific quality requirements"""
        state = {
            "requirements": ResearchRequirements(quality_threshold=0.8)
        }
        
        retrieved_info = [self.high_quality_info, self.medium_quality_info, self.low_quality_info]
        
        filtered_info = _filter_retrieved_info(retrieved_info, state)
        
        # Should keep only high quality info
        assert len(filtered_info) == 1
        assert filtered_info[0] == self.high_quality_info
    
    def test_filter_retrieved_info_relaxed_thresholds(self):
        """Test filtering with relaxed thresholds when too much is filtered"""
        state = {
            "requirements": ResearchRequirements(quality_threshold=0.9)  # Very high threshold
        }
        
        # All info is below the strict threshold
        retrieved_info = [self.medium_quality_info, self.medium_quality_info, self.medium_quality_info]
        
        filtered_info = _filter_retrieved_info(retrieved_info, state)
        
        # Should relax thresholds and keep medium quality info
        assert len(filtered_info) == 3
    
    def test_filter_retrieved_info_no_requirements(self):
        """Test filtering without requirements"""
        state = {"requirements": None}
        
        retrieved_info = [self.high_quality_info, self.medium_quality_info, self.low_quality_info]
        
        filtered_info = _filter_retrieved_info(retrieved_info, state)
        
        # Should use default thresholds and filter out low quality
        assert len(filtered_info) == 2
        assert self.low_quality_info not in filtered_info
    
    def test_filter_retrieved_info_sorting(self):
        """Test that filtered info is sorted by quality"""
        state = {"requirements": ResearchRequirements(quality_threshold=0.5)}
        
        retrieved_info = [self.medium_quality_info, self.high_quality_info, self.low_quality_info]
        
        filtered_info = _filter_retrieved_info(retrieved_info, state)
        
        # Should be sorted by combined quality score (descending)
        assert filtered_info[0] == self.high_quality_info
        assert filtered_info[1] == self.medium_quality_info
    
    def test_filter_retrieved_info_empty_list(self):
        """Test filtering empty list"""
        state = {"requirements": ResearchRequirements()}
        
        filtered_info = _filter_retrieved_info([], state)
        
        assert filtered_info == []

class TestFallbackRetrievedInfo:
    """Test fallback retrieved information creation"""
    
    def test_create_fallback_retrieved_info(self):
        """Test creation of fallback retrieved information"""
        gaps = [
            InformationGap(
                id="gap1",
                section_id="intro",
                gap_type=GapType.CONTENT,
                description="Need introduction content",
                priority=Priority.HIGH
            ),
            InformationGap(
                id="gap2",
                section_id="methods",
                gap_type=GapType.EVIDENCE,
                description="Need methodology evidence",
                priority=Priority.MEDIUM
            )
        ]
        
        fallback_info = _create_fallback_retrieved_info(gaps)
        
        assert len(fallback_info) == 2
        
        # Check first fallback item
        info1 = fallback_info[0]
        assert info1.gap_id == "gap1"
        assert info1.source.url == "https://example.com/research/intro"
        assert "introduction content" in info1.content
        assert info1.relevance_score == 0.6
        assert info1.credibility_score == 0.6
        
        # Check second fallback item
        info2 = fallback_info[1]
        assert info2.gap_id == "gap2"
        assert info2.source.url == "https://example.com/research/methods"
        assert "methodology evidence" in info2.content
    
    def test_create_fallback_retrieved_info_empty_gaps(self):
        """Test fallback creation with empty gaps list"""
        fallback_info = _create_fallback_retrieved_info([])
        assert fallback_info == []

class TestRetrievalEngineHealthCheck:
    """Test retrieval engine health check functionality"""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        with patch('workflow.retrieval_engine_node.DynamicRetrievalEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.health_check = AsyncMock()
            mock_engine.health_check.return_value = {"google_search": True, "kimi_k2": True}
            mock_engine_class.return_value = mock_engine
            
            health_status = await test_retrieval_engine_health()
            
            assert health_status["status"] == "healthy"
            assert health_status["components"]["google_search"] is True
            assert health_status["components"]["kimi_k2"] is True
            assert "timestamp" in health_status
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check with some components failing"""
        with patch('workflow.retrieval_engine_node.DynamicRetrievalEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.health_check = AsyncMock()
            mock_engine.health_check.return_value = {"google_search": False, "kimi_k2": True}
            mock_engine_class.return_value = mock_engine
            
            health_status = await test_retrieval_engine_health()
            
            assert health_status["status"] == "degraded"
            assert health_status["components"]["google_search"] is False
            assert health_status["components"]["kimi_k2"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure"""
        with patch('workflow.retrieval_engine_node.DynamicRetrievalEngine') as mock_engine_class:
            mock_engine_class.side_effect = Exception("Health check failed")
            
            health_status = await test_retrieval_engine_health()
            
            assert health_status["status"] == "unhealthy"
            assert "Health check failed" in health_status["error"]
            assert "timestamp" in health_status

if __name__ == "__main__":
    pytest.main([__file__])