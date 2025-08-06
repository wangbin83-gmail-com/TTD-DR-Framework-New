"""
Test suite for TTD-DR workflow structure and basic functionality.
Tests the LangGraph workflow implementation and node execution.
"""

import pytest
import logging
from datetime import datetime
from backend.workflow.graph import (
    create_ttdr_workflow, StateGraph, CompiledGraph, WorkflowError,
    draft_generator_node, gap_analyzer_node, retrieval_engine_node,
    information_integrator_node, quality_assessor_node, quality_check_node,
    self_evolution_enhancer_node, report_synthesizer_node
)
from backend.models.core import TTDRState, ResearchRequirements, ResearchDomain, ComplexityLevel
from backend.models.state_management import create_workflow_state

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestWorkflowStructure:
    """Test the basic workflow structure and configuration"""
    
    def test_create_ttdr_workflow(self):
        """Test that the TTD-DR workflow can be created successfully"""
        workflow = create_ttdr_workflow()
        
        assert isinstance(workflow, StateGraph)
        assert workflow.entry_point == "draft_generator"
        assert "report_synthesizer" in workflow.end_nodes
        
        # Check that all required nodes are present
        expected_nodes = [
            "draft_generator", "gap_analyzer", "retrieval_engine",
            "information_integrator", "quality_assessor",
            "self_evolution_enhancer", "report_synthesizer"
        ]
        
        for node_name in expected_nodes:
            assert node_name in workflow.nodes
            assert workflow.nodes[node_name].func is not None
    
    def test_workflow_compilation(self):
        """Test that the workflow can be compiled successfully"""
        workflow = create_ttdr_workflow()
        compiled_workflow = workflow.compile()
        
        assert isinstance(compiled_workflow, CompiledGraph)
        assert compiled_workflow.graph == workflow
    
    def test_workflow_edges(self):
        """Test that workflow edges are configured correctly"""
        workflow = create_ttdr_workflow()
        
        # Check that edges exist between expected nodes
        edge_pairs = [
            ("draft_generator", "gap_analyzer"),
            ("gap_analyzer", "retrieval_engine"),
            ("retrieval_engine", "information_integrator"),
            ("information_integrator", "quality_assessor"),
            ("self_evolution_enhancer", "report_synthesizer")
        ]
        
        for from_node, to_node in edge_pairs:
            edge_exists = any(
                edge.from_node == from_node and edge.to_node == to_node
                for edge in workflow.edges
            )
            assert edge_exists, f"Missing edge: {from_node} -> {to_node}"
        
        # Check conditional edges from quality_assessor
        conditional_edges = [
            ("quality_assessor", "gap_analyzer"),
            ("quality_assessor", "self_evolution_enhancer")
        ]
        
        for from_node, to_node in conditional_edges:
            edge_exists = any(
                edge.from_node == from_node and edge.to_node == to_node
                for edge in workflow.edges
            )
            assert edge_exists, f"Missing conditional edge: {from_node} -> {to_node}"

class TestNodeFunctions:
    """Test individual node functions"""
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample TTD-DR state for testing"""
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.8
        )
        
        return create_workflow_state("Artificial Intelligence in Healthcare", requirements)
    
    def test_draft_generator_node(self, sample_state):
        """Test the draft generator node"""
        result_state = draft_generator_node(sample_state)
        
        assert result_state["current_draft"] is not None
        assert result_state["current_draft"].topic == sample_state["topic"]
        assert result_state["iteration_count"] == 0
        assert len(result_state["current_draft"].structure.sections) > 0
        assert len(result_state["current_draft"].content) > 0
    
    def test_gap_analyzer_node(self, sample_state):
        """Test the gap analyzer node"""
        # First generate a draft
        state_with_draft = draft_generator_node(sample_state)
        
        # Then analyze gaps
        result_state = gap_analyzer_node(state_with_draft)
        
        assert len(result_state["information_gaps"]) > 0
        
        # Check that gaps correspond to sections
        section_ids = {section.id for section in result_state["current_draft"].structure.sections}
        gap_section_ids = {gap.section_id for gap in result_state["information_gaps"]}
        
        assert gap_section_ids.issubset(section_ids)
    
    def test_retrieval_engine_node(self, sample_state):
        """Test the retrieval engine node"""
        # Set up state with draft and gaps
        state_with_draft = draft_generator_node(sample_state)
        state_with_gaps = gap_analyzer_node(state_with_draft)
        
        # Test retrieval
        result_state = retrieval_engine_node(state_with_gaps)
        
        assert len(result_state["retrieved_info"]) > 0
        assert len(result_state["retrieved_info"]) == len(result_state["information_gaps"])
        
        # Check that retrieved info is linked to gaps
        gap_ids = {gap.id for gap in result_state["information_gaps"]}
        retrieved_gap_ids = {info.gap_id for info in result_state["retrieved_info"] if info.gap_id}
        
        assert retrieved_gap_ids.issubset(gap_ids)
    
    def test_information_integrator_node(self, sample_state):
        """Test the information integrator node"""
        # Set up complete state
        state_with_draft = draft_generator_node(sample_state)
        state_with_gaps = gap_analyzer_node(state_with_draft)
        state_with_info = retrieval_engine_node(state_with_gaps)
        
        # Test integration
        result_state = information_integrator_node(state_with_info)
        
        assert result_state["iteration_count"] == state_with_info["iteration_count"] + 1
        assert result_state["current_draft"].iteration > state_with_info["current_draft"].iteration
        
        # Check that content was updated
        original_content_length = sum(len(content) for content in state_with_info["current_draft"].content.values())
        updated_content_length = sum(len(content) for content in result_state["current_draft"].content.values())
        
        assert updated_content_length > original_content_length
    
    def test_quality_assessor_node(self, sample_state):
        """Test the quality assessor node"""
        # Set up state with draft
        state_with_draft = draft_generator_node(sample_state)
        
        # Test quality assessment
        result_state = quality_assessor_node(state_with_draft)
        
        assert result_state["quality_metrics"] is not None
        assert 0.0 <= result_state["quality_metrics"].overall_score <= 1.0
        assert 0.0 <= result_state["quality_metrics"].completeness <= 1.0
        assert 0.0 <= result_state["quality_metrics"].coherence <= 1.0
        assert 0.0 <= result_state["quality_metrics"].accuracy <= 1.0
        assert 0.0 <= result_state["quality_metrics"].citation_quality <= 1.0
    
    def test_quality_check_node(self, sample_state):
        """Test the quality check decision node"""
        # Test with low quality (should continue iteration)
        state_with_draft = draft_generator_node(sample_state)
        state_with_quality = quality_assessor_node(state_with_draft)
        
        # Ensure quality is below threshold
        state_with_quality["quality_metrics"].overall_score = 0.5
        
        decision = quality_check_node(state_with_quality)
        assert decision == "gap_analyzer"
        
        # Test with high quality (should move to evolution)
        state_with_quality["quality_metrics"].overall_score = 0.9
        decision = quality_check_node(state_with_quality)
        assert decision == "self_evolution_enhancer"
        
        # Test with max iterations reached
        state_with_quality["iteration_count"] = state_with_quality["requirements"].max_iterations
        state_with_quality["quality_metrics"].overall_score = 0.5
        decision = quality_check_node(state_with_quality)
        assert decision == "self_evolution_enhancer"
    
    def test_self_evolution_enhancer_node(self, sample_state):
        """Test the self-evolution enhancer node"""
        # Set up state with quality metrics
        state_with_draft = draft_generator_node(sample_state)
        state_with_quality = quality_assessor_node(state_with_draft)
        
        # Test evolution
        result_state = self_evolution_enhancer_node(state_with_quality)
        
        assert len(result_state["evolution_history"]) > len(state_with_quality["evolution_history"])
        
        latest_record = result_state["evolution_history"][-1]
        assert latest_record.component == "overall_workflow"
        assert latest_record.improvement_type == "quality_optimization"
    
    def test_report_synthesizer_node(self, sample_state):
        """Test the report synthesizer node"""
        # Set up complete state
        state_with_draft = draft_generator_node(sample_state)
        state_with_gaps = gap_analyzer_node(state_with_draft)
        state_with_info = retrieval_engine_node(state_with_gaps)
        
        # Test report synthesis
        result_state = report_synthesizer_node(state_with_info)
        
        assert result_state["final_report"] is not None
        assert len(result_state["final_report"]) > 0
        assert sample_state["topic"] in result_state["final_report"]
        assert "Executive Summary" in result_state["final_report"]
        assert "Research Methodology" in result_state["final_report"]
        assert "Sources" in result_state["final_report"]

class TestWorkflowExecution:
    """Test complete workflow execution"""
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample research requirements"""
        return ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.BASIC,
            max_iterations=2,  # Keep low for testing
            quality_threshold=0.6  # Lower threshold for testing
        )
    
    def test_complete_workflow_execution(self, sample_requirements):
        """Test complete workflow execution from start to finish"""
        # Create workflow and initial state
        workflow = create_ttdr_workflow()
        compiled_workflow = workflow.compile()
        
        initial_state = create_workflow_state(
            "Machine Learning Applications", 
            sample_requirements
        )
        
        # Execute workflow
        final_state = compiled_workflow.invoke(initial_state)
        
        # Verify final state
        assert final_state["final_report"] is not None
        assert len(final_state["final_report"]) > 0
        assert final_state["current_draft"] is not None
        assert final_state["iteration_count"] >= 0
        assert len(final_state["evolution_history"]) > 0
        
        # Check execution history
        assert len(workflow.execution_history) > 0
        latest_execution = workflow.execution_history[-1]
        assert latest_execution["status"] == "completed"
        assert "draft_generator" in latest_execution["execution_path"]
        assert "report_synthesizer" in latest_execution["execution_path"]
    
    def test_workflow_iteration_logic(self, sample_requirements):
        """Test that workflow correctly handles iteration logic"""
        # Set high quality threshold to force iteration
        sample_requirements.quality_threshold = 0.95  # Very high threshold
        sample_requirements.max_iterations = 3
        
        workflow = create_ttdr_workflow()
        compiled_workflow = workflow.compile()
        
        initial_state = create_workflow_state(
            "Blockchain Technology", 
            sample_requirements
        )
        
        final_state = compiled_workflow.invoke(initial_state)
        
        # Should have gone through multiple iterations due to high quality threshold
        assert final_state["iteration_count"] > 0
        
        # Check that gap_analyzer was executed multiple times
        latest_execution = workflow.execution_history[-1]
        gap_analyzer_count = latest_execution["execution_path"].count("gap_analyzer")
        assert gap_analyzer_count >= 1  # At least one iteration should occur
    
    def test_workflow_error_handling(self):
        """Test workflow error handling"""
        # Create workflow with invalid configuration
        workflow = StateGraph(TTDRState)
        workflow.add_node("test_node", lambda x: x)
        
        # Try to compile without entry point
        with pytest.raises(WorkflowError):
            workflow.compile()
    
    def test_workflow_state_validation(self, sample_requirements):
        """Test that workflow maintains state integrity"""
        workflow = create_ttdr_workflow()
        compiled_workflow = workflow.compile()
        
        initial_state = create_workflow_state(
            "Quantum Computing", 
            sample_requirements
        )
        
        final_state = compiled_workflow.invoke(initial_state)
        
        # Validate state structure
        assert "topic" in final_state
        assert "requirements" in final_state
        assert "current_draft" in final_state
        assert "information_gaps" in final_state
        assert "retrieved_info" in final_state
        assert "iteration_count" in final_state
        assert "quality_metrics" in final_state
        assert "evolution_history" in final_state
        assert "final_report" in final_state
        assert "error_log" in final_state
        
        # Validate data types
        assert isinstance(final_state["topic"], str)
        assert isinstance(final_state["iteration_count"], int)
        assert isinstance(final_state["information_gaps"], list)
        assert isinstance(final_state["retrieved_info"], list)
        assert isinstance(final_state["evolution_history"], list)
        assert isinstance(final_state["error_log"], list)

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])