"""
Simplified workflow graph implementation for TTD-DR framework.
Provides LangGraph-like functionality for Python 3.7 compatibility.
"""

from typing import Dict, List, Callable, Any, Optional, Union
from enum import Enum
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from models.core import TTDRState

logger = logging.getLogger(__name__)

class NodeStatus(str, Enum):
    """Status of workflow nodes"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class EdgeType(str, Enum):
    """Types of edges in the workflow graph"""
    NORMAL = "normal"
    CONDITIONAL = "conditional"

class WorkflowError(Exception):
    """Base exception for workflow errors"""
    pass

class NodeExecutionError(WorkflowError):
    """Raised when a node execution fails"""
    pass

class WorkflowNode:
    """Represents a single node in the workflow graph"""
    
    def __init__(self, name: str, func: Callable[[TTDRState], TTDRState], description: str = ""):
        """
        Initialize workflow node
        
        Args:
            name: Unique name for the node
            func: Function to execute for this node
            description: Optional description of the node's purpose
        """
        self.name = name
        self.func = func
        self.description = description
        self.status = NodeStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[Exception] = None
        self.execution_log: List[str] = []
    
    async def execute(self, state: TTDRState) -> TTDRState:
        """
        Execute the node function with the given state, supporting both sync and async functions.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after node execution
            
        Raises:
            NodeExecutionError: If node execution fails
        """
        try:
            self.status = NodeStatus.RUNNING
            self.start_time = datetime.now()
            self.execution_log.append(f"[{self.start_time}] Starting execution of node: {self.name}")
            
            logger.info(f"Executing node: {self.name}")
            
            # Execute the node function (sync or async)
            if asyncio.iscoroutinefunction(self.func):
                updated_state = await self.func(state)
            else:
                updated_state = self.func(state)
            
            self.end_time = datetime.now()
            execution_time = (self.end_time - self.start_time).total_seconds()
            self.execution_log.append(f"[{self.end_time}] Completed execution in {execution_time:.2f}s")
            
            self.status = NodeStatus.COMPLETED
            logger.info(f"Node {self.name} completed successfully in {execution_time:.2f}s")
            
            return updated_state
            
        except Exception as e:
            self.end_time = datetime.now()
            self.error = e
            self.status = NodeStatus.FAILED
            self.execution_log.append(f"[{self.end_time}] Failed with error: {str(e)}")
            
            logger.error(f"Node {self.name} failed: {str(e)}")
            raise NodeExecutionError(f"Node {self.name} execution failed: {str(e)}") from e
    
    def reset(self):
        """Reset node status for re-execution"""
        self.status = NodeStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.execution_log.clear()

class WorkflowEdge:
    """Represents an edge between workflow nodes"""
    
    def __init__(self, from_node: str, to_node: str, edge_type: EdgeType = EdgeType.NORMAL, 
                 condition: Optional[Callable[[TTDRState], bool]] = None):
        """
        Initialize workflow edge
        
        Args:
            from_node: Source node name
            to_node: Target node name
            edge_type: Type of edge (normal or conditional)
            condition: Optional condition function for conditional edges
        """
        self.from_node = from_node
        self.to_node = to_node
        self.edge_type = edge_type
        self.condition = condition
    
    def should_traverse(self, state: TTDRState) -> bool:
        """
        Determine if this edge should be traversed
        
        Args:
            state: Current workflow state
            
        Returns:
            True if edge should be traversed
        """
        if self.edge_type == EdgeType.NORMAL:
            return True
        elif self.edge_type == EdgeType.CONDITIONAL and self.condition:
            try:
                return self.condition(state)
            except Exception as e:
                logger.error(f"Condition evaluation failed for edge {self.from_node}->{self.to_node}: {str(e)}")
                return False
        return False

class StateGraph:
    """Simplified state graph for workflow orchestration"""
    
    def __init__(self, state_schema: type = TTDRState):
        """
        Initialize state graph
        
        Args:
            state_schema: Type of state object used in the workflow
        """
        self.state_schema = state_schema
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.entry_point: Optional[str] = None
        self.end_nodes: List[str] = []
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_node(self, name: str, func: Callable[[TTDRState], TTDRState], description: str = ""):
        """
        Add a node to the workflow graph
        
        Args:
            name: Unique name for the node
            func: Function to execute for this node
            description: Optional description of the node's purpose
        """
        if name in self.nodes:
            raise WorkflowError(f"Node {name} already exists in the graph")
        
        self.nodes[name] = WorkflowNode(name, func, description)
        logger.debug(f"Added node: {name}")
    
    def add_edge(self, from_node: str, to_node: str):
        """
        Add a normal edge between two nodes
        
        Args:
            from_node: Source node name
            to_node: Target node name
        """
        self._validate_node_exists(from_node)
        self._validate_node_exists(to_node)
        
        edge = WorkflowEdge(from_node, to_node, EdgeType.NORMAL)
        self.edges.append(edge)
        logger.debug(f"Added edge: {from_node} -> {to_node}")
    
    def add_conditional_edges(self, from_node: str, condition_func: Callable[[TTDRState], str], 
                            mapping: Dict[str, str]):
        """
        Add conditional edges from a node based on state evaluation
        
        Args:
            from_node: Source node name
            condition_func: Function that returns the key for mapping
            mapping: Dictionary mapping condition results to target nodes
        """
        self._validate_node_exists(from_node)
        
        # Store the condition function and mapping for later use
        for condition_key, to_node in mapping.items():
            self._validate_node_exists(to_node)
            
            # Create a condition function that checks for specific key
            def make_condition(key):
                def condition(state: TTDRState) -> bool:
                    try:
                        result = condition_func(state)
                        return result == key
                    except Exception:
                        return False
                return condition
            
            edge = WorkflowEdge(from_node, to_node, EdgeType.CONDITIONAL, make_condition(condition_key))
            self.edges.append(edge)
            logger.debug(f"Added conditional edge: {from_node} -> {to_node} (condition: {condition_key})")
        
        # Store the condition function for special handling
        if not hasattr(self, '_conditional_functions'):
            self._conditional_functions = {}
        self._conditional_functions[from_node] = condition_func
    
    def set_entry_point(self, node_name: str):
        """
        Set the entry point for workflow execution
        
        Args:
            node_name: Name of the starting node
        """
        self._validate_node_exists(node_name)
        self.entry_point = node_name
        logger.debug(f"Set entry point: {node_name}")
    
    def add_end_node(self, node_name: str):
        """
        Mark a node as an end node
        
        Args:
            node_name: Name of the end node
        """
        self._validate_node_exists(node_name)
        if node_name not in self.end_nodes:
            self.end_nodes.append(node_name)
            logger.debug(f"Added end node: {node_name}")
    
    def compile(self) -> 'CompiledGraph':
        """
        Compile the graph for execution
        
        Returns:
            Compiled graph ready for execution
        """
        if not self.entry_point:
            raise WorkflowError("Entry point must be set before compilation")
        
        if not self.end_nodes:
            # Auto-detect end nodes (nodes with no outgoing edges)
            outgoing_nodes = {edge.from_node for edge in self.edges}
            all_nodes = set(self.nodes.keys())
            self.end_nodes = list(all_nodes - outgoing_nodes)
            
            if not self.end_nodes:
                raise WorkflowError("No end nodes found. Graph may have cycles.")
        
        # Validate graph connectivity
        self._validate_graph()
        
        return CompiledGraph(self)
    
    def _validate_node_exists(self, node_name: str):
        """Validate that a node exists in the graph"""
        if node_name not in self.nodes:
            raise WorkflowError(f"Node {node_name} does not exist in the graph")
    
    def _validate_graph(self):
        """Validate graph structure and connectivity"""
        # Check that entry point can reach all nodes
        reachable = self._get_reachable_nodes(self.entry_point)
        all_nodes = set(self.nodes.keys())
        unreachable = all_nodes - reachable
        
        if unreachable:
            logger.warning(f"Unreachable nodes detected: {unreachable}")
    
    def _get_reachable_nodes(self, start_node: str) -> set:
        """Get all nodes reachable from the start node"""
        visited = set()
        stack = [start_node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add all target nodes from current node
            for edge in self.edges:
                if edge.from_node == current and edge.to_node not in visited:
                    stack.append(edge.to_node)
        
        return visited

class CompiledGraph:
    """Compiled workflow graph ready for execution"""
    
    def __init__(self, graph: StateGraph):
        """
        Initialize compiled graph
        
        Args:
            graph: Source state graph
        """
        self.graph = graph
        self.execution_id: Optional[str] = None
        
    async def invoke(self, initial_state: TTDRState, config: Optional[Dict[str, Any]] = None) -> TTDRState:
        """
        Execute the workflow with the given initial state
        
        Args:
            initial_state: Starting state for the workflow
            config: Optional configuration for execution
            
        Returns:
            Final state after workflow completion
        """
        self.execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = config or {}
        
        logger.info(f"Starting workflow execution: {self.execution_id}")
        
        # Reset all nodes
        for node in self.graph.nodes.values():
            node.reset()
        
        current_state = initial_state
        current_node = self.graph.entry_point
        execution_path = []
        
        try:
            while current_node and current_node not in self.graph.end_nodes:
                # Execute current node
                node = self.graph.nodes[current_node]
                execution_path.append(current_node)
                
                logger.info(f"Executing node: {current_node}")
                current_state = await node.execute(current_state)
                
                # Find next node
                next_node = self._get_next_node(current_node, current_state)
                
                if next_node is None:
                    logger.info(f"No next node found from {current_node}, ending execution")
                    break
                
                current_node = next_node
            
            # Execute final node if it's an end node
            if current_node and current_node in self.graph.end_nodes:
                node = self.graph.nodes[current_node]
                execution_path.append(current_node)
                logger.info(f"Executing final node: {current_node}")
                current_state = await node.execute(current_state)
            
            # Record execution history
            self.graph.execution_history.append({
                "execution_id": self.execution_id,
                "timestamp": datetime.now(),
                "execution_path": execution_path,
                "final_state_summary": self._summarize_state(current_state),
                "status": "completed"
            })
            
            logger.info(f"Workflow execution completed: {self.execution_id}")
            logger.info(f"Execution path: {' -> '.join(execution_path)}")
            
            return current_state
            
        except Exception as e:
            # Record failed execution
            self.graph.execution_history.append({
                "execution_id": self.execution_id,
                "timestamp": datetime.now(),
                "execution_path": execution_path,
                "error": str(e),
                "status": "failed"
            })
            
            logger.error(f"Workflow execution failed: {self.execution_id} - {str(e)}")
            raise WorkflowError(f"Workflow execution failed: {str(e)}") from e
    
    def _get_next_node(self, current_node: str, state: TTDRState) -> Optional[str]:
        """
        Determine the next node to execute based on current state
        
        Args:
            current_node: Current node name
            state: Current workflow state
            
        Returns:
            Next node name or None if no valid next node
        """
        # Find all edges from current node
        outgoing_edges = [edge for edge in self.graph.edges if edge.from_node == current_node]
        
        if not outgoing_edges:
            return None
        
        # Check conditional edges first
        for edge in outgoing_edges:
            if edge.edge_type == EdgeType.CONDITIONAL and edge.should_traverse(state):
                return edge.to_node
        
        # Then check normal edges
        for edge in outgoing_edges:
            if edge.edge_type == EdgeType.NORMAL:
                return edge.to_node
        
        return None
    
    def _summarize_state(self, state: TTDRState) -> Dict[str, Any]:
        """Create a summary of the final state for logging"""
        return {
            "topic": state.get("topic", ""),
            "iteration_count": state.get("iteration_count", 0),
            "has_draft": state.get("current_draft") is not None,
            "gaps_count": len(state.get("information_gaps", [])),
            "retrieved_info_count": len(state.get("retrieved_info", [])),
            "has_final_report": state.get("final_report") is not None,
            "error_count": len(state.get("error_log", []))
        }

# Import the actual draft generator implementation
from workflow.draft_generator import draft_generator_node

def gap_analyzer_node(state: TTDRState) -> TTDRState:
    """
    Analyze current draft for information gaps using Kimi K2
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with identified information gaps
    """
    logger.info("Executing gap_analyzer_node")
    
    try:
        from ..services.kimi_k2_gap_analyzer import KimiK2InformationGapAnalyzer
        from ..services.kimi_k2_search_query_generator import KimiK2SearchQueryGenerator
        
        if not state["current_draft"]:
            logger.warning("No current draft available for gap analysis")
            return {
                **state,
                "information_gaps": [],
                "error_log": state.get("error_log", []) + ["No draft available for gap analysis"]
            }
        
        # Initialize gap analyzer
        gap_analyzer = KimiK2InformationGapAnalyzer()
        query_generator = KimiK2SearchQueryGenerator()
        
        # Run async operations synchronously
        import asyncio
        
        # Identify information gaps
        try:
            gaps = asyncio.run(gap_analyzer.identify_gaps(state["current_draft"]))
        except RuntimeError:
            # If event loop is already running, use a different approach
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                gaps = loop.run_until_complete(gap_analyzer.identify_gaps(state["current_draft"]))
            finally:
                loop.close()
        
        # Generate search queries for each gap
        for gap in gaps:
            try:
                try:
                    search_queries = asyncio.run(query_generator.generate_search_queries(
                        gap=gap,
                        topic=state["topic"],
                        domain=state["current_draft"].structure.domain,
                        max_queries=3
                    ))
                except RuntimeError:
                    # If event loop is already running, use a different approach
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        search_queries = loop.run_until_complete(query_generator.generate_search_queries(
                            gap=gap,
                            topic=state["topic"],
                            domain=state["current_draft"].structure.domain,
                            max_queries=3
                        ))
                    finally:
                        loop.close()
                
                gap.search_queries = search_queries
                
            except Exception as e:
                logger.error(f"Failed to generate queries for gap {gap.id}: {e}")
                # Add fallback query
                from ..models.core import SearchQuery, Priority
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
        from ..models.core import InformationGap, GapType, Priority, SearchQuery
        import uuid
        
        gaps = []
        if state["current_draft"]:
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

# Import the actual retrieval engine implementation
from workflow.retrieval_engine_node import retrieval_engine_node

# Import the actual information integrator implementation
from workflow.information_integrator_node import information_integrator_node

# Import the actual quality assessor implementation
from workflow.quality_assessor_node import quality_assessor_node, quality_check_node

def self_evolution_enhancer_node(state: TTDRState) -> TTDRState:
    """
    Apply self-evolution algorithms to improve components using Kimi K2
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with evolution record
    """
    logger.info("Executing self_evolution_enhancer_node")
    
    # Placeholder implementation - will be replaced with actual Kimi K2 integration
    from ..models.core import EvolutionRecord
    
    # Create evolution record based on current performance
    if state["quality_metrics"]:
        evolution_record = EvolutionRecord(
            component="overall_workflow",
            improvement_type="quality_optimization",
            description="Applied self-evolution algorithms to improve research quality",
            performance_before=state["quality_metrics"].overall_score,
            performance_after=min(state["quality_metrics"].overall_score + 0.1, 1.0),
            parameters_changed={"iteration_strategy": "enhanced"}
        )
        
        updated_history = state["evolution_history"] + [evolution_record]
    else:
        updated_history = state["evolution_history"]
    
    return {
        **state,
        "evolution_history": updated_history
    }

def report_synthesizer_node(state: TTDRState) -> TTDRState:
    """
    Generate final research report using Kimi K2
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final report
    """
    logger.info("Executing report_synthesizer_node")
    
    # Placeholder implementation - will be replaced with actual Kimi K2 integration
    if state["current_draft"]:
        # Synthesize final report from draft content
        final_report = f"""# {state['topic']}

## Executive Summary
This research report presents a comprehensive analysis of {state['topic']}.

"""
        
        # Add all section content
        for section in state["current_draft"].structure.sections:
            if section.id in state["current_draft"].content:
                final_report += f"{state['current_draft'].content[section.id]}\n\n"
        
        # Add methodology and sources
        final_report += f"""
## Research Methodology
This report was generated using the TTD-DR framework with {state['iteration_count']} iterations.

## Sources
Total sources consulted: {len(state['retrieved_info'])}

"""
        
        # Add source bibliography
        for i, info in enumerate(state["retrieved_info"], 1):
            final_report += f"{i}. {info.source.title} - {info.source.url}\n"
        
        return {
            **state,
            "final_report": final_report
        }
    
    return state

# Utility functions for creating TTD-DR workflow
def create_ttdr_workflow() -> StateGraph:
    """
    Create the complete TTD-DR workflow with all nodes and edges
    
    Returns:
        Configured StateGraph for TTD-DR workflow
    """
    workflow = StateGraph(TTDRState)
    
    # Add all workflow nodes
    workflow.add_node("draft_generator", draft_generator_node, 
                     "Generate initial research draft using Kimi K2")
    workflow.add_node("gap_analyzer", gap_analyzer_node,
                     "Analyze current draft for information gaps")
    workflow.add_node("retrieval_engine", retrieval_engine_node,
                     "Retrieve information using Google Search API")
    workflow.add_node("information_integrator", information_integrator_node,
                     "Integrate retrieved information into draft")
    workflow.add_node("quality_assessor", quality_assessor_node,
                     "Assess draft quality using Kimi K2")
    workflow.add_node("self_evolution_enhancer", self_evolution_enhancer_node,
                     "Apply self-evolution algorithms")
    workflow.add_node("report_synthesizer", report_synthesizer_node,
                     "Generate final polished report")
    
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
        lambda state: quality_check_node(state),
        {
            "gap_analyzer": "gap_analyzer",
            "self_evolution_enhancer": "self_evolution_enhancer"
        }
    )
    
    workflow.add_edge("self_evolution_enhancer", "report_synthesizer")
    
    # Mark report_synthesizer as end node
    workflow.add_end_node("report_synthesizer")
    
    return workflow

def create_simple_workflow() -> StateGraph:
    """Create a simple linear workflow template"""
    workflow = StateGraph(TTDRState)
    return workflow

def create_iterative_workflow() -> StateGraph:
    """Create an iterative workflow template with feedback loops"""
    workflow = StateGraph(TTDRState)
    return workflow