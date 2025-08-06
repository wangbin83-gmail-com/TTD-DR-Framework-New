"""
Complete workflow orchestration and execution system for TTD-DR framework.
Implements task 10.1: Create complete workflow construction and compilation.
"""

import logging
import asyncio
import json
import pickle
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, asdict

from models.core import TTDRState, ResearchRequirements, Draft, QualityMetrics
from .graph import StateGraph, CompiledGraph, WorkflowError, NodeExecutionError
from workflow.draft_generator import draft_generator_node
from workflow.retrieval_engine_node import retrieval_engine_node
from workflow.information_integrator_node import information_integrator_node
from workflow.quality_assessor_node import quality_assessor_node, quality_check_node
from workflow.self_evolution_enhancer_node import self_evolution_enhancer_node
from workflow.report_synthesizer_node import report_synthesizer_node

logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    max_execution_time: int = 1800  # 30 minutes
    enable_persistence: bool = True
    persistence_path: str = "workflow_states"
    enable_recovery: bool = True
    parallel_execution: bool = False
    debug_mode: bool = False
    checkpoint_interval: int = 1  # Save state after each node
    
@dataclass
class ExecutionMetrics:
    """Metrics collected during workflow execution"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    nodes_executed: List[str] = None
    node_durations: Dict[str, float] = None
    iterations_completed: int = 0
    final_quality_score: Optional[float] = None
    errors_encountered: List[str] = None
    memory_usage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.nodes_executed is None:
            self.nodes_executed = []
        if self.node_durations is None:
            self.node_durations = {}
        if self.errors_encountered is None:
            self.errors_encountered = []
        if self.memory_usage is None:
            self.memory_usage = {}

class WorkflowPersistenceManager:
    """Manages workflow state persistence and recovery"""
    
    def __init__(self, persistence_path: str = "workflow_states"):
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(exist_ok=True)
        
    def save_state(self, execution_id: str, state: TTDRState, node_name: str) -> bool:
        """
        Save workflow state to disk
        
        Args:
            execution_id: Unique execution identifier
            state: Current workflow state
            node_name: Name of the node that just completed
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            checkpoint_data = {
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "node_name": node_name,
                "state": self._serialize_state(state)
            }
            
            checkpoint_file = self.persistence_path / f"{execution_id}_checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.debug(f"Saved checkpoint for execution {execution_id} after node {node_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state for execution {execution_id}: {e}")
            return False
    
    def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Load workflow state from disk
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Checkpoint data if found, None otherwise
        """
        try:
            checkpoint_file = self.persistence_path / f"{execution_id}_checkpoint.json"
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Deserialize state
            checkpoint_data["state"] = self._deserialize_state(checkpoint_data["state"])
            
            logger.info(f"Loaded checkpoint for execution {execution_id}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load state for execution {execution_id}: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints
        
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        for checkpoint_file in self.persistence_path.glob("*_checkpoint.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                checkpoints.append({
                    "execution_id": data["execution_id"],
                    "timestamp": data["timestamp"],
                    "node_name": data["node_name"],
                    "file_path": str(checkpoint_file)
                })
                
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Clean up old checkpoint files
        
        Args:
            max_age_days: Maximum age of checkpoints to keep
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        cleaned_count = 0
        
        for checkpoint_file in self.persistence_path.glob("*_checkpoint.json"):
            try:
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old checkpoint: {checkpoint_file}")
                    
            except Exception as e:
                logger.warning(f"Failed to clean up checkpoint {checkpoint_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} old checkpoint files")
        return cleaned_count
    
    def _serialize_state(self, state: TTDRState) -> Dict[str, Any]:
        """Serialize TTDRState for JSON storage"""
        serialized = {}
        
        for key, value in state.items():
            if value is None:
                serialized[key] = None
            elif hasattr(value, 'dict'):  # Pydantic model
                serialized[key] = value.dict()
            elif isinstance(value, list):
                serialized[key] = [
                    item.dict() if hasattr(item, 'dict') else item
                    for item in value
                ]
            else:
                serialized[key] = value
        
        return serialized
    
    def _deserialize_state(self, serialized: Dict[str, Any]) -> TTDRState:
        """Deserialize TTDRState from JSON storage"""
        # This is a simplified deserialization - in production,
        # you'd want more robust type reconstruction
        return serialized

class WorkflowExecutionEngine:
    """Advanced workflow execution engine with error handling and recovery"""
    
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.persistence_manager = WorkflowPersistenceManager(self.config.persistence_path)
        self.active_executions: Dict[str, ExecutionMetrics] = {}
        self._shutdown_event = threading.Event()
        
    def create_ttdr_workflow(self) -> StateGraph:
        """
        Create the complete TTD-DR workflow with all nodes and edges
        
        Returns:
            Configured StateGraph for TTD-DR workflow
        """
        logger.info("Creating TTD-DR workflow with all nodes and edges")
        
        workflow = StateGraph(TTDRState)
        
        # Add all workflow nodes with descriptions
        workflow.add_node(
            "draft_generator", 
            draft_generator_node,
            "Generate initial research draft using Kimi K2"
        )
        
        workflow.add_node(
            "gap_analyzer", 
            self._create_gap_analyzer_node(),
            "Analyze current draft for information gaps"
        )
        
        workflow.add_node(
            "retrieval_engine", 
            retrieval_engine_node,
            "Retrieve information using Google Search API"
        )
        
        workflow.add_node(
            "information_integrator", 
            information_integrator_node,
            "Integrate retrieved information into draft"
        )
        
        workflow.add_node(
            "quality_assessor", 
            quality_assessor_node,
            "Assess draft quality using Kimi K2"
        )
        
        workflow.add_node(
            "self_evolution_enhancer", 
            self_evolution_enhancer_node,
            "Apply self-evolution algorithms"
        )
        
        workflow.add_node(
            "report_synthesizer", 
            report_synthesizer_node,
            "Generate final polished report"
        )
        
        # Set entry point
        workflow.set_entry_point("draft_generator")
        
        # Define workflow edges
        workflow.add_edge("draft_generator", "gap_analyzer")
        workflow.add_edge("gap_analyzer", "retrieval_engine")
        workflow.add_edge("retrieval_engine", "information_integrator")
        workflow.add_edge("information_integrator", "quality_assessor")
        
        # Add conditional edges from quality_assessor based on quality check
        workflow.add_conditional_edges(
            "quality_assessor",
            quality_check_node,
            {
                "gap_analyzer": "gap_analyzer",
                "self_evolution_enhancer": "self_evolution_enhancer"
            }
        )
        
        workflow.add_edge("self_evolution_enhancer", "report_synthesizer")
        
        # Mark report_synthesizer as end node
        workflow.add_end_node("report_synthesizer")
        
        logger.info("TTD-DR workflow created successfully with 7 nodes and conditional routing")
        return workflow
    
    def execute_workflow(self, initial_state: TTDRState, 
                        execution_id: Optional[str] = None) -> TTDRState:
        """
        Execute the complete TTD-DR workflow with error handling and recovery
        
        Args:
            initial_state: Starting state for the workflow
            execution_id: Optional execution ID for recovery
            
        Returns:
            Final state after workflow completion
            
        Raises:
            WorkflowError: If workflow execution fails
        """
        if execution_id is None:
            execution_id = f"ttdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting workflow execution: {execution_id}")
        
        # Initialize execution metrics
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            start_time=datetime.now()
        )
        self.active_executions[execution_id] = metrics
        
        try:
            # Check for existing checkpoint if recovery is enabled
            if self.config.enable_recovery:
                checkpoint = self.persistence_manager.load_state(execution_id)
                if checkpoint:
                    logger.info(f"Resuming execution from checkpoint: {checkpoint['node_name']}")
                    initial_state = checkpoint["state"]
            
            # Create and compile workflow
            workflow = self.create_ttdr_workflow()
            compiled_workflow = workflow.compile()
            
            # Execute with timeout and monitoring
            if self.config.max_execution_time > 0:
                final_state = self._execute_with_timeout(
                    compiled_workflow, initial_state, execution_id, metrics
                )
            else:
                final_state = self._execute_with_monitoring(
                    compiled_workflow, initial_state, execution_id, metrics
                )
            
            # Update final metrics
            metrics.end_time = datetime.now()
            metrics.total_duration = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.iterations_completed = final_state.get("iteration_count", 0)
            
            if final_state.get("quality_metrics"):
                metrics.final_quality_score = final_state["quality_metrics"].overall_score
            
            logger.info(f"Workflow execution completed: {execution_id}")
            logger.info(f"Total duration: {metrics.total_duration:.2f}s")
            logger.info(f"Iterations: {metrics.iterations_completed}")
            logger.info(f"Final quality: {metrics.final_quality_score}")
            
            return final_state
            
        except Exception as e:
            metrics.end_time = datetime.now()
            metrics.total_duration = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.errors_encountered.append(str(e))
            
            logger.error(f"Workflow execution failed: {execution_id} - {str(e)}")
            raise WorkflowError(f"Workflow execution failed: {str(e)}") from e
            
        finally:
            # Clean up active execution
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def _execute_with_timeout(self, compiled_workflow: CompiledGraph, 
                            initial_state: TTDRState, execution_id: str,
                            metrics: ExecutionMetrics) -> TTDRState:
        """Execute workflow with timeout protection"""
        
        def execute_workflow():
            return self._execute_with_monitoring(
                compiled_workflow, initial_state, execution_id, metrics
            )
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_workflow)
            
            try:
                return future.result(timeout=self.config.max_execution_time)
            except TimeoutError:
                logger.error(f"Workflow execution timed out: {execution_id}")
                raise WorkflowError(f"Workflow execution timed out after {self.config.max_execution_time}s")
    
    def _execute_with_monitoring(self, compiled_workflow: CompiledGraph,
                               initial_state: TTDRState, execution_id: str,
                               metrics: ExecutionMetrics) -> TTDRState:
        """Execute workflow with monitoring and checkpointing"""
        
        # Create a custom invoke method with monitoring
        original_invoke = compiled_workflow.invoke
        
        def monitored_invoke(state):
            return self._monitored_execution(
                original_invoke, state, execution_id, metrics
            )
        
        # Replace the invoke method
        compiled_workflow.invoke = monitored_invoke
        
        # Execute the workflow
        return compiled_workflow.invoke(initial_state)
    
    def _monitored_execution(self, original_invoke: Callable, state: TTDRState,
                           execution_id: str, metrics: ExecutionMetrics) -> TTDRState:
        """Execute workflow with node-level monitoring"""
        
        # This is a simplified monitoring approach
        # In a full implementation, you'd hook into the graph execution
        try:
            result = original_invoke(state)
            
            # Save checkpoint if enabled
            if self.config.enable_persistence:
                self.persistence_manager.save_state(
                    execution_id, result, "workflow_complete"
                )
            
            return result
            
        except Exception as e:
            metrics.errors_encountered.append(str(e))
            raise
    
    def _create_gap_analyzer_node(self) -> Callable[[TTDRState], TTDRState]:
        """Create gap analyzer node with error handling"""
        
        def gap_analyzer_node(state: TTDRState) -> TTDRState:
            """
            Analyze current draft for information gaps using Kimi K2
            """
            logger.info("Executing gap_analyzer_node")
            
            try:
                # Import here to avoid circular imports
                from services.kimi_k2_gap_analyzer import KimiK2InformationGapAnalyzer
                from services.kimi_k2_search_query_generator import KimiK2SearchQueryGenerator
                
                if not state.get("current_draft"):
                    logger.warning("No current draft available for gap analysis")
                    return {
                        **state,
                        "information_gaps": [],
                        "error_log": state.get("error_log", []) + ["No draft available for gap analysis"]
                    }
                
                # Initialize gap analyzer
                gap_analyzer = KimiK2InformationGapAnalyzer()
                query_generator = KimiK2SearchQueryGenerator()
                
                # Run async operations
                gaps = self._run_async_operation(
                    gap_analyzer.identify_gaps(state["current_draft"])
                )
                
                # Generate search queries for each gap
                for gap in gaps:
                    try:
                        search_queries = self._run_async_operation(
                            query_generator.generate_search_queries(
                                gap=gap,
                                topic=state["topic"],
                                domain=state["current_draft"].structure.domain,
                                max_queries=3
                            )
                        )
                        gap.search_queries = search_queries
                        
                    except Exception as e:
                        logger.error(f"Failed to generate queries for gap {gap.id}: {e}")
                        # Add fallback query
                        from models.core import SearchQuery, Priority
                        gap.search_queries = [
                            SearchQuery(
                                query=f"{state['topic']} {gap.description[:50]}",
                                priority=Priority.MEDIUM
                            )
                        ]
                
                logger.info(f"Identified {len(gaps)} information gaps with search queries")
                
                return {
                    **state,
                    "information_gaps": gaps
                }
                
            except Exception as e:
                logger.error(f"Gap analysis failed: {e}")
                
                # Fallback to simple gap identification
                from models.core import InformationGap, GapType, Priority, SearchQuery
                import uuid
                
                gaps = []
                if state.get("current_draft"):
                    for section in state["current_draft"].structure.sections:
                        gap = InformationGap(
                            id=str(uuid.uuid4()),
                            section_id=section.id,
                            gap_type=GapType.CONTENT,
                            description=f"Need more detailed information for {section.title}",
                            priority=Priority.MEDIUM,
                            search_queries=[
                                SearchQuery(
                                    query=f"{state['topic']} {section.title.lower()}",
                                    priority=Priority.MEDIUM
                                )
                            ]
                        )
                        gaps.append(gap)
                
                return {
                    **state,
                    "information_gaps": gaps,
                    "error_log": state.get("error_log", []) + [f"Gap analysis error: {str(e)}"]
                }
        
        return gap_analyzer_node
    
    def _run_async_operation(self, coro):
        """Run async operation handling event loop issues"""
        try:
            # Check if coro is already a coroutine object
            if asyncio.iscoroutine(coro):
                return asyncio.run(coro)
            else:
                # If it's already a result, return it
                return coro
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if asyncio.iscoroutine(coro):
                        return loop.run_until_complete(coro)
                    else:
                        return coro
                finally:
                    loop.close()
            else:
                raise
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a running execution
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            Execution status information
        """
        if execution_id not in self.active_executions:
            return None
        
        metrics = self.active_executions[execution_id]
        current_time = datetime.now()
        
        return {
            "execution_id": execution_id,
            "status": "running",
            "start_time": metrics.start_time.isoformat(),
            "duration": (current_time - metrics.start_time).total_seconds(),
            "nodes_executed": metrics.nodes_executed.copy(),
            "iterations_completed": metrics.iterations_completed,
            "errors_count": len(metrics.errors_encountered)
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """
        List all active executions
        
        Returns:
            List of active execution statuses
        """
        return [
            self.get_execution_status(execution_id)
            for execution_id in self.active_executions.keys()
        ]
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            True if cancellation was successful
        """
        if execution_id not in self.active_executions:
            return False
        
        # In a full implementation, you'd need to signal the execution thread
        # For now, we'll just remove it from active executions
        logger.info(f"Cancelling execution: {execution_id}")
        del self.active_executions[execution_id]
        return True

# Utility functions for workflow creation and management

def create_workflow_state(topic: str, requirements: ResearchRequirements) -> TTDRState:
    """
    Create initial workflow state
    
    Args:
        topic: Research topic
        requirements: Research requirements
        
    Returns:
        Initial TTDRState for workflow execution
    """
    return TTDRState(
        topic=topic,
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

def validate_workflow_state(state: TTDRState) -> List[str]:
    """
    Validate workflow state for execution
    
    Args:
        state: Workflow state to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not state.get("topic"):
        errors.append("Topic is required")
    
    if not state.get("requirements"):
        errors.append("Requirements are required")
    
    if state.get("iteration_count", 0) < 0:
        errors.append("Iteration count cannot be negative")
    
    return errors

# Export main classes and functions
__all__ = [
    "WorkflowExecutionEngine",
    "WorkflowConfig", 
    "ExecutionMetrics",
    "WorkflowPersistenceManager",
    "create_workflow_state",
    "validate_workflow_state"
]