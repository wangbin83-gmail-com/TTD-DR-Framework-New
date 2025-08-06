"""
Comprehensive tests for the workflow recovery system.
Tests workflow recovery and continuation strategies for task 13.1.
"""

import pytest
import tempfile
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from services.workflow_recovery import (
    WorkflowRecoveryManager, WorkflowCheckpoint, RecoveryPlan, WorkflowState, 
    CheckpointType, WorkflowRecoveryContext
)
from services.error_handling import (
    ErrorHandlingFramework, ErrorCategory, RecoveryStrategy, ErrorContext, TTDRError
)
from models.core import TTDRState, Draft, ResearchStructure, Section, DraftMetadata

class TestWorkflowCheckpoint:
    """Test workflow checkpoint functionality"""
    
    def test_checkpoint_creation(self):
        """Test creating a workflow checkpoint"""
        state = {
            "topic": "Test Topic",
            "iteration_count": 1,
            "current_draft": None
        }
        
        checkpoint = WorkflowCheckpoint(
            state=state,
            workflow_state=WorkflowState.DRAFT_GENERATION,
            node_name="draft_generator",
            checkpoint_type=CheckpointType.AUTOMATIC
        )
        
        assert checkpoint.id is not None
        assert checkpoint.state == state
        assert checkpoint.workflow_state == WorkflowState.DRAFT_GENERATION
        assert checkpoint.node_name == "draft_generator"
        assert checkpoint.checkpoint_type == CheckpointType.AUTOMATIC
        assert isinstance(checkpoint.timestamp, datetime)

class TestRecoveryPlan:
    """Test recovery plan functionality"""
    
    def test_recovery_plan_creation(self):
        """Test creating a recovery plan"""
        checkpoint = WorkflowCheckpoint(
            state={"topic": "Test"},
            workflow_state=WorkflowState.GAP_ANALYSIS
        )
        
        plan = RecoveryPlan(
            recovery_strategy=RecoveryStrategy.RETRY,
            target_checkpoint=checkpoint,
            max_retry_attempts=3,
            backoff_factor=2.0
        )
        
        assert plan.recovery_strategy == RecoveryStrategy.RETRY
        assert plan.target_checkpoint == checkpoint
        assert plan.max_retry_attempts == 3
        assert plan.backoff_factor == 2.0

class TestWorkflowRecoveryManager:
    """Test workflow recovery manager functionality"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_manager = WorkflowRecoveryManager(
            checkpoint_dir=self.temp_dir,
            error_framework=ErrorHandlingFramework()
        )
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_checkpoint(self):
        """Test creating and storing a checkpoint"""
        state = {
            "topic": "AI Research",
            "iteration_count": 2,
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": []
        }
        
        checkpoint = self.recovery_manager.create_checkpoint(
            state=state,
            workflow_state=WorkflowState.INFORMATION_RETRIEVAL,
            node_name="retrieval_engine",
            metadata={"workflow_id": "test_workflow"}
        )
        
        # Check checkpoint was created
        assert checkpoint.id in self.recovery_manager.checkpoints
        assert checkpoint.state == state
        assert checkpoint.workflow_state == WorkflowState.INFORMATION_RETRIEVAL
        
        # Check checkpoint was persisted
        checkpoint_file = Path(self.temp_dir) / f"checkpoint_{checkpoint.id}.json"
        assert checkpoint_file.exists()
        
        # Verify file content
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            assert data["id"] == checkpoint.id
            assert data["workflow_state"] == "information_retrieval"
            assert data["node_name"] == "retrieval_engine"
    
    def test_get_latest_checkpoint(self):
        """Test retrieving the latest checkpoint"""
        # Create multiple checkpoints
        state1 = {"topic": "Topic 1", "iteration_count": 1}
        state2 = {"topic": "Topic 2", "iteration_count": 2}
        
        checkpoint1 = self.recovery_manager.create_checkpoint(
            state1, WorkflowState.DRAFT_GENERATION
        )
        
        # Wait a bit to ensure different timestamps
        import time
        time.sleep(0.01)
        
        checkpoint2 = self.recovery_manager.create_checkpoint(
            state2, WorkflowState.GAP_ANALYSIS
        )
        
        latest = self.recovery_manager.get_latest_checkpoint()
        
        assert latest.id == checkpoint2.id
        assert latest.state == state2
    
    def test_get_latest_checkpoint_by_workflow_id(self):
        """Test retrieving latest checkpoint filtered by workflow ID"""
        state1 = {"topic": "Topic 1"}
        state2 = {"topic": "Topic 2"}
        
        # Create checkpoints with different workflow IDs
        checkpoint1 = self.recovery_manager.create_checkpoint(
            state1, WorkflowState.DRAFT_GENERATION,
            metadata={"workflow_id": "workflow_1"}
        )
        
        checkpoint2 = self.recovery_manager.create_checkpoint(
            state2, WorkflowState.GAP_ANALYSIS,
            metadata={"workflow_id": "workflow_2"}
        )
        
        latest_workflow_1 = self.recovery_manager.get_latest_checkpoint("workflow_1")
        
        assert latest_workflow_1.id == checkpoint1.id
        assert latest_workflow_1.metadata["workflow_id"] == "workflow_1"
    
    def test_create_recovery_plan_network_error(self):
        """Test creating recovery plan for network errors"""
        from services.error_handling import NetworkError
        
        error = NetworkError("Connection failed")
        context = ErrorContext(component="retrieval_engine", operation="search")
        
        plan = self.recovery_manager.create_recovery_plan(error, context)
        
        assert plan.recovery_strategy == RecoveryStrategy.RETRY
        assert plan.max_retry_attempts == 3
        assert plan.backoff_factor == 2.0
        assert plan.timeout == 60.0
    
    def test_create_recovery_plan_rate_limit_error(self):
        """Test creating recovery plan for rate limit errors"""
        from services.error_handling import RateLimitError
        
        error = RateLimitError("Rate limit exceeded")
        context = ErrorContext(component="kimi_client", operation="generate")
        
        plan = self.recovery_manager.create_recovery_plan(error, context)
        
        assert plan.recovery_strategy == RecoveryStrategy.RETRY
        assert plan.max_retry_attempts == 5
        assert plan.backoff_factor == 3.0
        assert plan.timeout == 300.0
    
    def test_create_recovery_plan_processing_error(self):
        """Test creating recovery plan for processing errors"""
        from services.error_handling import ProcessingError
        
        error = ProcessingError("Data processing failed")
        context = ErrorContext(component="information_integrator", operation="integrate")
        
        plan = self.recovery_manager.create_recovery_plan(error, context)
        
        assert plan.recovery_strategy == RecoveryStrategy.FALLBACK
        assert len(plan.fallback_actions) > 0
    
    def test_create_recovery_plan_workflow_error(self):
        """Test creating recovery plan for workflow errors"""
        from services.error_handling import WorkflowError
        
        error = WorkflowError("Node execution failed", node="quality_assessor")
        context = ErrorContext(component="quality_assessor", operation="assess")
        
        plan = self.recovery_manager.create_recovery_plan(error, context)
        
        assert plan.recovery_strategy == RecoveryStrategy.FALLBACK
        assert "assess" in plan.skip_nodes
    
    def test_execute_recovery_plan_retry_success(self):
        """Test executing retry recovery plan successfully"""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        plan = RecoveryPlan(
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retry_attempts=3,
            backoff_factor=1.0  # No delay for testing
        )
        
        result = self.recovery_manager.execute_recovery_plan(plan, flaky_function)
        
        assert result == "success"
        assert call_count == 3
    
    def test_execute_recovery_plan_fallback(self):
        """Test executing fallback recovery plan"""
        def fallback_action(state):
            return {"recovered": True}
        
        plan = RecoveryPlan(
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_actions=[fallback_action]
        )
        
        state = {"topic": "Test Topic"}
        result = self.recovery_manager.execute_recovery_plan(plan, None, state)
        
        assert result == {"recovered": True}
    
    def test_execute_recovery_plan_skip(self):
        """Test executing skip recovery plan"""
        plan = RecoveryPlan(recovery_strategy=RecoveryStrategy.SKIP)
        
        result = self.recovery_manager.execute_recovery_plan(plan, None)
        
        assert result is None
    
    def test_execute_recovery_plan_abort(self):
        """Test executing abort recovery plan"""
        from services.error_handling import WorkflowError
        
        plan = RecoveryPlan(recovery_strategy=RecoveryStrategy.ABORT)
        
        with pytest.raises(WorkflowError, match="Workflow aborted"):
            self.recovery_manager.execute_recovery_plan(plan, None)
    
    def test_restore_from_checkpoint(self):
        """Test restoring workflow state from checkpoint"""
        original_state = {
            "topic": "Machine Learning",
            "iteration_count": 3,
            "current_draft": None,
            "quality_metrics": None
        }
        
        checkpoint = self.recovery_manager.create_checkpoint(
            original_state, WorkflowState.QUALITY_ASSESSMENT
        )
        
        restored_state = self.recovery_manager.restore_from_checkpoint(checkpoint.id)
        
        assert restored_state is not None
        assert restored_state["topic"] == "Machine Learning"
        assert restored_state["iteration_count"] == 3
    
    def test_restore_from_nonexistent_checkpoint(self):
        """Test restoring from non-existent checkpoint"""
        result = self.recovery_manager.restore_from_checkpoint("nonexistent_id")
        
        assert result is None
    
    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints"""
        # Create some old checkpoints
        old_time = datetime.now() - timedelta(hours=25)
        
        for i in range(3):
            checkpoint = WorkflowCheckpoint(
                state={"topic": f"Old Topic {i}"},
                workflow_state=WorkflowState.DRAFT_GENERATION
            )
            checkpoint.timestamp = old_time
            self.recovery_manager.checkpoints[checkpoint.id] = checkpoint
            self.recovery_manager._persist_checkpoint(checkpoint)
        
        # Create a recent checkpoint
        recent_checkpoint = self.recovery_manager.create_checkpoint(
            {"topic": "Recent Topic"}, WorkflowState.GAP_ANALYSIS
        )
        
        # Cleanup with 24 hour max age
        self.recovery_manager.cleanup_old_checkpoints(max_age_hours=24)
        
        # Only recent checkpoint should remain
        assert len(self.recovery_manager.checkpoints) == 1
        assert recent_checkpoint.id in self.recovery_manager.checkpoints

class TestFallbackActions:
    """Test fallback action implementations"""
    
    def setup_method(self):
        self.recovery_manager = WorkflowRecoveryManager()
    
    def test_fallback_draft_generation(self):
        """Test fallback draft generation"""
        state = {
            "topic": "Artificial Intelligence",
            "requirements": None,
            "current_draft": None
        }
        
        result_state = self.recovery_manager._fallback_draft_generation(state)
        
        assert result_state["current_draft"] is not None
        assert result_state["current_draft"]["topic"] == "Artificial Intelligence"
        assert "introduction" in result_state["current_draft"]["content"]
        assert "main_content" in result_state["current_draft"]["content"]
        assert "conclusion" in result_state["current_draft"]["content"]
    
    def test_fallback_gap_analysis(self):
        """Test fallback gap analysis"""
        # Create a mock draft
        sections = [
            {"id": "intro", "title": "Introduction"},
            {"id": "body", "title": "Main Content"}
        ]
        
        draft = {
            "structure": {"sections": sections},
            "content": {}
        }
        
        state = {
            "topic": "Test Topic",
            "current_draft": draft,
            "information_gaps": []
        }
        
        result_state = self.recovery_manager._fallback_gap_analysis(state)
        
        assert len(result_state["information_gaps"]) == 2
        assert result_state["information_gaps"][0]["section_id"] == "intro"
        assert result_state["information_gaps"][1]["section_id"] == "body"
    
    def test_fallback_information_retrieval(self):
        """Test fallback information retrieval"""
        gaps = [
            {"id": "gap1", "description": "Need more info about AI"},
            {"id": "gap2", "description": "Need examples"}
        ]
        
        state = {
            "information_gaps": gaps,
            "retrieved_info": []
        }
        
        result_state = self.recovery_manager._fallback_information_retrieval(state)
        
        assert len(result_state["retrieved_info"]) == 2
        assert result_state["retrieved_info"][0]["gap_id"] == "gap1"
        assert result_state["retrieved_info"][1]["gap_id"] == "gap2"
    
    def test_fallback_information_integration(self):
        """Test fallback information integration"""
        draft = {
            "content": {"section1": "Original content"},
            "iteration": 0
        }
        
        gaps = [{"id": "gap1", "section_id": "section1"}]
        
        retrieved_info = [{
            "gap_id": "gap1",
            "content": "Additional information"
        }]
        
        state = {
            "current_draft": draft,
            "information_gaps": gaps,
            "retrieved_info": retrieved_info
        }
        
        result_state = self.recovery_manager._fallback_information_integration(state)
        
        assert "Additional information" in result_state["current_draft"]["content"]["section1"]
        assert result_state["current_draft"]["iteration"] == 1
    
    def test_fallback_quality_assessment(self):
        """Test fallback quality assessment"""
        draft = {
            "content": {
                "section1": "This is some content for testing quality assessment.",
                "section2": "More content here to reach a reasonable length."
            }
        }
        
        state = {
            "current_draft": draft,
            "quality_metrics": None
        }
        
        result_state = self.recovery_manager._fallback_quality_assessment(state)
        
        assert result_state["quality_metrics"] is not None
        assert "completeness" in result_state["quality_metrics"]
        assert "coherence" in result_state["quality_metrics"]
        assert "overall_score" in result_state["quality_metrics"]
    
    def test_fallback_self_evolution(self):
        """Test fallback self-evolution"""
        state = {
            "evolution_history": []
        }
        
        result_state = self.recovery_manager._fallback_self_evolution(state)
        
        assert len(result_state["evolution_history"]) == 1
        assert result_state["evolution_history"][0]["component"] == "fallback_system"
        assert result_state["evolution_history"][0]["improvement_type"] == "basic_enhancement"
    
    def test_fallback_report_synthesis(self):
        """Test fallback report synthesis"""
        sections = [
            {"id": "intro", "title": "Introduction"},
            {"id": "body", "title": "Main Content"}
        ]
        
        draft = {
            "topic": "Test Research Report",
            "structure": {"sections": sections},
            "content": {
                "intro": "This is the introduction.",
                "body": "This is the main content."
            }
        }
        
        state = {
            "current_draft": draft,
            "final_report": None
        }
        
        result_state = self.recovery_manager._fallback_report_synthesis(state)
        
        assert result_state["final_report"] is not None
        assert "# Test Research Report" in result_state["final_report"]
        assert "## Introduction" in result_state["final_report"]
        assert "## Main Content" in result_state["final_report"]

class TestWorkflowRecoveryContext:
    """Test workflow recovery context manager"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_manager = WorkflowRecoveryManager(checkpoint_dir=self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recovery_context_success(self):
        """Test recovery context with successful operation"""
        state = {"topic": "Test Topic", "iteration_count": 1}
        
        with WorkflowRecoveryContext(
            self.recovery_manager,
            WorkflowState.DRAFT_GENERATION,
            "draft_generator",
            state
        ) as ctx:
            assert ctx.checkpoint is not None
            result = "operation_success"
        
        assert result == "operation_success"
        # Checkpoint should be created
        assert len(self.recovery_manager.checkpoints) == 1
    
    def test_recovery_context_with_error(self):
        """Test recovery context with error"""
        state = {"topic": "Test Topic", "iteration_count": 1}
        
        with pytest.raises(ValueError):
            with WorkflowRecoveryContext(
                self.recovery_manager,
                WorkflowState.GAP_ANALYSIS,
                "gap_analyzer",
                state
            ) as ctx:
                raise ValueError("Test error in context")
        
        # Checkpoint and recovery plan should be created
        assert len(self.recovery_manager.checkpoints) == 1
        checkpoint_id = list(self.recovery_manager.checkpoints.keys())[0]
        assert checkpoint_id in self.recovery_manager.recovery_plans

class TestSerializationAndPersistence:
    """Test serialization and persistence functionality"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_manager = WorkflowRecoveryManager(checkpoint_dir=self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_serialize_complex_state(self):
        """Test serializing complex TTDRState"""
        # Create a complex state with various data types
        state = {
            "topic": "Complex Research Topic",
            "iteration_count": 5,
            "current_draft": {
                "id": "draft_123",
                "topic": "Test Topic",
                "content": {"section1": "Content 1"}
            },
            "information_gaps": [
                {"id": "gap1", "description": "Gap 1"},
                {"id": "gap2", "description": "Gap 2"}
            ],
            "retrieved_info": [
                {"source": {"url": "http://example.com"}, "content": "Retrieved content"}
            ],
            "quality_metrics": {
                "completeness": 0.8,
                "coherence": 0.7,
                "overall_score": 0.75
            },
            "evolution_history": [
                {"component": "test", "improvement_type": "optimization"}
            ],
            "final_report": None,
            "error_log": ["Error 1", "Error 2"]
        }
        
        serialized = self.recovery_manager._serialize_state(state)
        
        assert serialized["topic"] == "Complex Research Topic"
        assert serialized["iteration_count"] == 5
        assert isinstance(serialized["current_draft"], (dict, str))
        assert isinstance(serialized["information_gaps"], list)
        assert len(serialized["information_gaps"]) == 2
    
    def test_checkpoint_persistence_and_loading(self):
        """Test checkpoint persistence and loading from disk"""
        # Create initial recovery manager
        state = {"topic": "Persistence Test", "iteration_count": 1}
        checkpoint = self.recovery_manager.create_checkpoint(
            state, WorkflowState.INFORMATION_INTEGRATION
        )
        
        # Create new recovery manager (simulating restart)
        new_recovery_manager = WorkflowRecoveryManager(checkpoint_dir=self.temp_dir)
        
        # Check that checkpoint was loaded
        assert checkpoint.id in new_recovery_manager.checkpoints
        loaded_checkpoint = new_recovery_manager.checkpoints[checkpoint.id]
        assert loaded_checkpoint.workflow_state == WorkflowState.INFORMATION_INTEGRATION

class TestErrorIntegration:
    """Test integration with error handling framework"""
    
    def setup_method(self):
        self.error_framework = ErrorHandlingFramework()
        self.recovery_manager = WorkflowRecoveryManager(error_framework=self.error_framework)
    
    def test_error_classification_integration(self):
        """Test integration with error classification"""
        from services.error_handling import NetworkError
        
        error = NetworkError("Network failure")
        context = ErrorContext(component="retrieval_engine", operation="search")
        
        plan = self.recovery_manager.create_recovery_plan(error, context)
        
        # Should use error framework's classification
        assert plan.recovery_strategy == RecoveryStrategy.RETRY
        assert plan.max_retry_attempts == 3
    
    def test_recovery_with_error_framework(self):
        """Test recovery execution with error framework"""
        # Register a fallback handler in the error framework
        def test_fallback(error_record):
            return "framework_fallback_result"
        
        self.error_framework.recovery_manager.register_fallback_handler(
            ErrorCategory.PROCESSING, test_fallback
        )
        
        from services.error_handling import ProcessingError
        
        error = ProcessingError("Processing failed")
        context = ErrorContext(component="integrator", operation="integrate")
        
        plan = self.recovery_manager.create_recovery_plan(error, context)
        
        # Should have fallback actions
        assert plan.recovery_strategy == RecoveryStrategy.FALLBACK
        assert len(plan.fallback_actions) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])