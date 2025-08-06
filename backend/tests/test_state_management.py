"""
Tests for TTD-DR state management utilities
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from backend.models.core import (
    TTDRState, ResearchRequirements, Draft, InformationGap, 
    RetrievedInfo, QualityMetrics, EvolutionRecord, Source,
    ResearchStructure, Section, DraftMetadata, SearchQuery,
    GapType, Priority, ComplexityLevel, ResearchDomain
)
from backend.models.state_management import (
    TTDRStateManager, StateValidationError, StatePersistenceError,
    create_workflow_state, validate_workflow_state
)

class TestTTDRStateManager:
    """Test cases for TTDRStateManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing persistence"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def state_manager(self, temp_dir):
        """Create state manager with temporary persistence directory"""
        return TTDRStateManager(persistence_dir=temp_dir)
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample research requirements"""
        return ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.8,
            max_sources=10
        )
    
    @pytest.fixture
    def sample_draft(self):
        """Create sample draft"""
        structure = ResearchStructure(
            sections=[
                Section(id="intro", title="Introduction"),
                Section(id="methods", title="Methods")
            ],
            estimated_length=1000,
            complexity_level=ComplexityLevel.INTERMEDIATE
        )
        
        return Draft(
            id="draft_001",
            topic="AI Research",
            structure=structure,
            content={"intro": "Introduction content", "methods": "Methods content"},
            quality_score=0.7,
            iteration=1
        )
    
    def test_create_initial_state(self, state_manager, sample_requirements):
        """Test creation of initial workflow state"""
        topic = "Test Research Topic"
        state = state_manager.create_initial_state(topic, sample_requirements)
        
        assert state["topic"] == topic
        assert state["requirements"] == sample_requirements
        assert state["current_draft"] is None
        assert state["information_gaps"] == []
        assert state["retrieved_info"] == []
        assert state["iteration_count"] == 0
        assert state["quality_metrics"] is None
        assert state["evolution_history"] == []
        assert state["final_report"] is None
        assert state["error_log"] == []
    
    def test_validate_state_valid(self, state_manager, sample_requirements):
        """Test validation of valid state"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        assert state_manager.validate_state(state) is True
    
    def test_validate_state_missing_field(self, state_manager, sample_requirements):
        """Test validation fails for missing required field"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        del state["topic"]
        
        with pytest.raises(StateValidationError, match="Missing required field: topic"):
            state_manager.validate_state(state)
    
    def test_validate_state_invalid_topic(self, state_manager, sample_requirements):
        """Test validation fails for invalid topic"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        state["topic"] = ""
        
        with pytest.raises(StateValidationError, match="Topic must be a non-empty string"):
            state_manager.validate_state(state)
    
    def test_validate_state_invalid_iteration_count(self, state_manager, sample_requirements):
        """Test validation fails for invalid iteration count"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        state["iteration_count"] = -1
        
        with pytest.raises(StateValidationError, match="Iteration count must be a non-negative integer"):
            state_manager.validate_state(state)
    
    def test_validate_state_exceeds_max_iterations(self, state_manager, sample_requirements):
        """Test validation fails when iteration count exceeds maximum"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        state["iteration_count"] = sample_requirements.max_iterations + 1
        
        with pytest.raises(StateValidationError, match="Iteration count .* exceeds maximum"):
            state_manager.validate_state(state)
    
    def test_transform_state(self, state_manager, sample_requirements, sample_draft):
        """Test state transformation"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        
        transformations = {
            "current_draft": sample_draft,
            "iteration_count": 1
        }
        
        new_state = state_manager.transform_state(state, transformations)
        
        assert new_state["current_draft"] == sample_draft
        assert new_state["iteration_count"] == 1
        assert new_state["topic"] == state["topic"]  # Unchanged fields preserved
    
    def test_increment_iteration(self, state_manager, sample_requirements):
        """Test iteration count increment"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        assert state["iteration_count"] == 0
        
        new_state = state_manager.increment_iteration(state)
        assert new_state["iteration_count"] == 1
    
    def test_add_error(self, state_manager, sample_requirements):
        """Test adding error to error log"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        error_message = "Test error message"
        
        new_state = state_manager.add_error(state, error_message)
        
        assert len(new_state["error_log"]) == 1
        assert error_message in new_state["error_log"][0]
    
    def test_persist_and_load_state(self, state_manager, sample_requirements, sample_draft):
        """Test state persistence and loading"""
        # Create a state with some data
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        state = state_manager.transform_state(state, {
            "current_draft": sample_draft,
            "iteration_count": 2
        })
        
        # Persist the state
        state_id = "test_state"
        filepath = state_manager.persist_state(state, state_id)
        
        # Load the state
        loaded_state = state_manager.load_state(filepath)
        
        # Verify loaded state matches original
        assert loaded_state["topic"] == state["topic"]
        assert loaded_state["iteration_count"] == state["iteration_count"]
        assert loaded_state["current_draft"].id == state["current_draft"].id
    
    def test_load_nonexistent_state(self, state_manager):
        """Test loading non-existent state file"""
        with pytest.raises(StatePersistenceError, match="State file not found"):
            state_manager.load_state("nonexistent_file.pkl")
    
    def test_list_persisted_states(self, state_manager, sample_requirements):
        """Test listing persisted states"""
        # Initially no states
        assert len(state_manager.list_persisted_states()) == 0
        
        # Create and persist some states
        state1 = state_manager.create_initial_state("Topic 1", sample_requirements)
        state2 = state_manager.create_initial_state("Topic 2", sample_requirements)
        
        state_manager.persist_state(state1, "state_1")
        state_manager.persist_state(state2, "state_2")
        
        # Should have 2 states
        states = state_manager.list_persisted_states()
        assert len(states) == 2
        
        # Test filtering by prefix
        state_1_files = state_manager.list_persisted_states("state_1")
        assert len(state_1_files) == 1
        assert "state_1" in state_1_files[0]
    
    def test_cleanup_old_states(self, state_manager, sample_requirements):
        """Test cleanup of old persisted states"""
        # Create multiple states
        for i in range(5):
            state = state_manager.create_initial_state(f"Topic {i}", sample_requirements)
            state_manager.persist_state(state, f"state_{i}")
        
        # Should have 5 states
        assert len(state_manager.list_persisted_states()) == 5
        
        # Cleanup keeping only 3
        deleted_count = state_manager.cleanup_old_states(keep_count=3)
        assert deleted_count == 2
        assert len(state_manager.list_persisted_states()) == 3
    
    def test_gap_info_relationship_validation(self, state_manager, sample_requirements):
        """Test validation of gap-info relationships"""
        state = state_manager.create_initial_state("Test Topic", sample_requirements)
        
        # Add information gap
        gap = InformationGap(
            id="gap_001",
            section_id="intro",
            gap_type=GapType.CONTENT,
            description="Missing content",
            priority=Priority.HIGH
        )
        
        # Add retrieved info with valid gap reference
        source = Source(
            url="https://example.com",
            title="Test Source",
            domain="example.com",
            credibility_score=0.8
        )
        
        valid_info = RetrievedInfo(
            source=source,
            content="Retrieved content",
            relevance_score=0.9,
            credibility_score=0.8,
            gap_id="gap_001"
        )
        
        # Add retrieved info with invalid gap reference
        invalid_info = RetrievedInfo(
            source=source,
            content="Invalid content",
            relevance_score=0.9,
            credibility_score=0.8,
            gap_id="nonexistent_gap"
        )
        
        # Valid state should pass
        valid_state = state_manager.transform_state(state, {
            "information_gaps": [gap],
            "retrieved_info": [valid_info]
        })
        assert state_manager.validate_state(valid_state) is True
        
        # Invalid state should fail
        invalid_state = state_manager.transform_state(state, {
            "information_gaps": [gap],
            "retrieved_info": [invalid_info]
        })
        
        with pytest.raises(StateValidationError, match="Retrieved info references non-existent gap"):
            state_manager.validate_state(invalid_state)

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_workflow_state(self):
        """Test create_workflow_state utility function"""
        topic = "Test Topic"
        requirements = ResearchRequirements()
        
        state = create_workflow_state(topic, requirements)
        
        assert state["topic"] == topic
        assert state["requirements"] == requirements
        assert state["iteration_count"] == 0
    
    def test_validate_workflow_state(self):
        """Test validate_workflow_state utility function"""
        topic = "Test Topic"
        requirements = ResearchRequirements()
        state = create_workflow_state(topic, requirements)
        
        assert validate_workflow_state(state) is True
        
        # Test invalid state
        state["topic"] = ""
        with pytest.raises(StateValidationError):
            validate_workflow_state(state)

if __name__ == "__main__":
    pytest.main([__file__])