"""
Workflow recovery and continuation strategies for TTD-DR framework.
Implements error recovery and workflow continuation strategies for task 13.1.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import uuid

from models.core import TTDRState
from services.error_handling import (
    ErrorHandlingFramework, TTDRError, WorkflowError, ErrorSeverity, 
    ErrorCategory, RecoveryStrategy, ErrorContext
)

logger = logging.getLogger(__name__)

def initialize_recovery_manager():
    """Initializes and prints debug messages."""
    print("DEBUG: Starting workflow_recovery.py execution")
    print("DEBUG: logging imported")
    print("DEBUG: Basic imports completed")
    print("DEBUG: TTDRState imported successfully")
    print("DEBUG: error_handling imports successful")
    print("DEBUG: Logger created")

class WorkflowState(str, Enum):
    """Workflow execution states"""
    INITIALIZING = "initializing"
    DRAFT_GENERATION = "draft_generation"
    GAP_ANALYSIS = "gap_analysis"
    INFORMATION_RETRIEVAL = "information_retrieval"
    INFORMATION_INTEGRATION = "information_integration"
    QUALITY_ASSESSMENT = "quality_assessment"
    SELF_EVOLUTION = "self_evolution"
    REPORT_SYNTHESIS = "report_synthesis"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class CheckpointType(str, Enum):
    """Types of workflow checkpoints"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ERROR_RECOVERY = "error_recovery"
    ITERATION_BOUNDARY = "iteration_boundary"

@dataclass
class WorkflowCheckpoint:
    """Represents a workflow checkpoint for recovery"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    state: TTDRState = field(default_factory=dict)
    workflow_state: WorkflowState = WorkflowState.INITIALIZING
    checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    node_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_context: Optional[ErrorContext] = None

@dataclass
class RecoveryPlan:
    """Defines a recovery plan for workflow continuation"""
    recovery_strategy: RecoveryStrategy
    target_checkpoint: Optional[WorkflowCheckpoint] = None
    skip_nodes: List[str] = field(default_factory=list)
    fallback_actions: List[Callable] = field(default_factory=list)
    max_retry_attempts: int = 3
    backoff_factor: float = 2.0
    timeout: Optional[float] = None

class WorkflowRecoveryManager:
    """Manages workflow recovery and continuation strategies"""
    
    def __init__(self, checkpoint_dir: str = "workflow_states", 
                 error_framework: Optional[ErrorHandlingFramework] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.error_framework = error_framework or ErrorHandlingFramework()
        
        # Active checkpoints storage
        self.checkpoints: Dict[str, WorkflowCheckpoint] = {}
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        
        # Recovery strategies for different error types
        self.recovery_strategies = self._build_recovery_strategies()
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def _build_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
        """Build default recovery strategies for different error categories"""
        return {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.API: RecoveryStrategy.RETRY,
            ErrorCategory.RATE_LIMIT: RecoveryStrategy.RETRY,
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.MANUAL,
            ErrorCategory.VALIDATION: RecoveryStrategy.FALLBACK,
            ErrorCategory.PROCESSING: RecoveryStrategy.FALLBACK,
            ErrorCategory.WORKFLOW: RecoveryStrategy.FALLBACK,
            ErrorCategory.STORAGE: RecoveryStrategy.RETRY,
            ErrorCategory.CONFIGURATION: RecoveryStrategy.MANUAL,
            ErrorCategory.UNKNOWN: RecoveryStrategy.FALLBACK
        }
    
    def create_checkpoint(self, state: TTDRState, workflow_state: WorkflowState,
                         node_name: Optional[str] = None, 
                         checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
                         metadata: Optional[Dict[str, Any]] = None) -> WorkflowCheckpoint:
        """Create a workflow checkpoint for recovery purposes"""
        checkpoint = WorkflowCheckpoint(
            state=state.copy(),
            workflow_state=workflow_state,
            node_name=node_name,
            checkpoint_type=checkpoint_type,
            metadata=metadata or {}
        )
        
        # Store checkpoint
        self.checkpoints[checkpoint.id] = checkpoint
        
        # Persist to disk
        self._persist_checkpoint(checkpoint)
        
        logger.info(f"Created checkpoint {checkpoint.id} at {workflow_state.value}")
        return checkpoint
    
    def _persist_checkpoint(self, checkpoint: WorkflowCheckpoint):
        """Persist checkpoint to disk"""
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint.id}.json"
            
            # Convert checkpoint to serializable format
            checkpoint_data = {
                "id": checkpoint.id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "state": self._serialize_state(checkpoint.state),
                "workflow_state": checkpoint.workflow_state.value,
                "checkpoint_type": checkpoint.checkpoint_type.value,
                "node_name": checkpoint.node_name,
                "metadata": checkpoint.metadata
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist checkpoint {checkpoint.id}: {e}")
    
    def _serialize_state(self, state: TTDRState) -> Dict[str, Any]:
        """Serialize TTDRState for persistence"""
        serialized = {}
        
        for key, value in state.items():
            if hasattr(value, 'dict'):
                serialized[key] = value.dict()
            elif isinstance(value, list):
                serialized[key] = [
                    item.dict() if hasattr(item, 'dict') else str(item) 
                    for item in value
                ]
            else:
                serialized[key] = value
        
        return serialized
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoints from disk"""
        try:
            for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Reconstruct checkpoint
                    checkpoint = WorkflowCheckpoint(
                        id=checkpoint_data["id"],
                        timestamp=datetime.fromisoformat(checkpoint_data["timestamp"]),
                        state=checkpoint_data["state"],
                        workflow_state=WorkflowState(checkpoint_data["workflow_state"]),
                        checkpoint_type=CheckpointType(checkpoint_data["checkpoint_type"]),
                        node_name=checkpoint_data.get("node_name"),
                        metadata=checkpoint_data.get("metadata", {})
                    )
                    
                    self.checkpoints[checkpoint.id] = checkpoint
                    
                except Exception as e:
                    logger.error(f"Failed to load checkpoint from {checkpoint_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load existing checkpoints: {e}")
    
    def get_latest_checkpoint(self, workflow_id: Optional[str] = None) -> Optional[WorkflowCheckpoint]:
        """Get the most recent checkpoint"""
        if not self.checkpoints:
            return None
        
        # Filter by workflow_id if provided
        checkpoints = self.checkpoints.values()
        if workflow_id:
            checkpoints = [cp for cp in checkpoints 
                          if cp.metadata.get("workflow_id") == workflow_id]
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda cp: cp.timestamp)
    
    def create_recovery_plan(self, error: Exception, context: Optional[ErrorContext] = None,
                           current_checkpoint: Optional[WorkflowCheckpoint] = None) -> RecoveryPlan:
        """Create a recovery plan based on the error and context"""
        # Classify the error
        error_record = self.error_framework.classifier.classify_error(error, context)
        
        # Determine recovery strategy
        recovery_strategy = self.recovery_strategies.get(
            error_record.category, RecoveryStrategy.FALLBACK
        )
        
        # Create recovery plan
        recovery_plan = RecoveryPlan(
            recovery_strategy=recovery_strategy,
            target_checkpoint=current_checkpoint
        )
        
        # Customize plan based on error type and context
        if error_record.category == ErrorCategory.NETWORK:
            recovery_plan.max_retry_attempts = 3
            recovery_plan.backoff_factor = 2.0
            recovery_plan.timeout = 60.0
            
        elif error_record.category == ErrorCategory.API:
            recovery_plan.max_retry_attempts = 3
            recovery_plan.backoff_factor = 1.5
            recovery_plan.timeout = 120.0
            
        elif error_record.category == ErrorCategory.RATE_LIMIT:
            recovery_plan.max_retry_attempts = 5
            recovery_plan.backoff_factor = 3.0
            recovery_plan.timeout = 300.0
            
        elif error_record.category == ErrorCategory.PROCESSING:
            recovery_plan.recovery_strategy = RecoveryStrategy.FALLBACK
            if context and context.component:
                recovery_plan.fallback_actions = self._get_fallback_actions(context.component)
        
        elif error_record.category == ErrorCategory.WORKFLOW:
            if context and context.operation:
                recovery_plan.skip_nodes = [context.operation]
        
        return recovery_plan
    
    def _get_fallback_actions(self, component: str) -> List[Callable]:
        """Get fallback actions for a specific component"""
        fallback_actions = []
        
        if component == "draft_generator":
            fallback_actions.append(self._fallback_draft_generation)
        elif component == "gap_analyzer":
            fallback_actions.append(self._fallback_gap_analysis)
        elif component == "retrieval_engine":
            fallback_actions.append(self._fallback_information_retrieval)
        elif component == "information_integrator":
            fallback_actions.append(self._fallback_information_integration)
        elif component == "quality_assessor":
            fallback_actions.append(self._fallback_quality_assessment)
        elif component == "self_evolution_enhancer":
            fallback_actions.append(self._fallback_self_evolution)
        elif component == "report_synthesizer":
            fallback_actions.append(self._fallback_report_synthesis)
        
        return fallback_actions
    
    def execute_recovery_plan(self, recovery_plan: RecoveryPlan, 
                            original_function: Callable = None,
                            *args, **kwargs) -> Any:
        """Execute a recovery plan"""
        if recovery_plan.recovery_strategy == RecoveryStrategy.RETRY:
            return self._execute_retry_recovery(recovery_plan, original_function, *args, **kwargs)
        
        elif recovery_plan.recovery_strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback_recovery(recovery_plan, *args, **kwargs)
        
        elif recovery_plan.recovery_strategy == RecoveryStrategy.SKIP:
            logger.warning("Skipping failed operation as per recovery plan")
            return None
        
        elif recovery_plan.recovery_strategy == RecoveryStrategy.ABORT:
            raise WorkflowError("Workflow aborted due to critical error")
        
        elif recovery_plan.recovery_strategy == RecoveryStrategy.MANUAL:
            logger.error("Manual intervention required - workflow paused")
            return None
        
        else:
            logger.error(f"Unknown recovery strategy: {recovery_plan.recovery_strategy}")
            return None
    
    def _execute_retry_recovery(self, recovery_plan: RecoveryPlan, 
                              original_function: Callable, *args, **kwargs) -> Any:
        """Execute retry-based recovery"""
        if not original_function:
            logger.warning("No original function provided for retry recovery")
            return None
            
        max_attempts = recovery_plan.max_retry_attempts
        backoff_factor = recovery_plan.backoff_factor
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    delay = backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying operation in {delay} seconds (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                
                return original_function(*args, **kwargs)
                
            except Exception as retry_error:
                if attempt == max_attempts - 1:
                    logger.error(f"All retry attempts failed: {retry_error}")
                    raise retry_error
                else:
                    logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
        
        return None
    
    def _execute_fallback_recovery(self, recovery_plan: RecoveryPlan, *args, **kwargs) -> Any:
        """Execute fallback-based recovery"""
        for fallback_action in recovery_plan.fallback_actions:
            try:
                logger.info(f"Executing fallback action: {fallback_action.__name__}")
                return fallback_action(*args, **kwargs)
            except Exception as fallback_error:
                logger.warning(f"Fallback action {fallback_action.__name__} failed: {fallback_error}")
                continue
        
        logger.error("All fallback actions failed")
        return None
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[TTDRState]:
        """Restore workflow state from a checkpoint"""
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return None
        
        try:
            restored_state = checkpoint.state.copy()
            logger.info(f"Restored workflow state from checkpoint {checkpoint_id}")
            return restored_state
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return None
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24, max_count: int = 50):
        """Clean up old checkpoints to manage storage"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Remove old checkpoints from memory
            old_checkpoint_ids = [
                cp_id for cp_id, cp in self.checkpoints.items()
                if cp.timestamp < cutoff_time
            ]
            
            # Keep only the most recent checkpoints
            if len(self.checkpoints) > max_count:
                sorted_checkpoints = sorted(
                    self.checkpoints.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True
                )
                old_checkpoint_ids.extend([
                    cp_id for cp_id, _ in sorted_checkpoints[max_count:]
                ])
            
            # Remove from memory and disk
            for cp_id in old_checkpoint_ids:
                if cp_id in self.checkpoints:
                    del self.checkpoints[cp_id]
                
                # Remove file
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{cp_id}.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            
            if old_checkpoint_ids:
                logger.info(f"Cleaned up {len(old_checkpoint_ids)} old checkpoints")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    # Simplified fallback action implementations
    def _fallback_draft_generation(self, state: TTDRState) -> TTDRState:
        """Fallback draft generation using simplified approach"""
        logger.info("Executing fallback draft generation")
        
        # Create a basic draft structure
        basic_draft = {
            "id": str(uuid.uuid4()),
            "topic": state.get("topic", "Research Topic"),
            "content": {
                "introduction": f"This research explores {state.get('topic', 'the given topic')}.",
                "main_content": "Content will be developed through iterative research.",
                "conclusion": "Conclusions will be drawn based on research findings."
            },
            "iteration": 0
        }
        
        state["current_draft"] = basic_draft
        return state
    
    def _fallback_gap_analysis(self, state: TTDRState) -> TTDRState:
        """Fallback gap analysis using basic heuristics"""
        logger.info("Executing fallback gap analysis")
        
        # Create basic gaps
        gaps = [
            {
                "id": str(uuid.uuid4()),
                "section_id": "introduction",
                "description": "Need more information for introduction",
                "priority": "medium"
            },
            {
                "id": str(uuid.uuid4()),
                "section_id": "main_content",
                "description": "Need more information for main content",
                "priority": "high"
            }
        ]
        
        state["information_gaps"] = gaps
        return state
    
    def _fallback_information_retrieval(self, state: TTDRState) -> TTDRState:
        """Fallback information retrieval using cached or default content"""
        logger.info("Executing fallback information retrieval")
        
        # Create placeholder retrieved information
        retrieved_info = []
        for gap in state.get("information_gaps", []):
            info = {
                "source": {"url": "https://example.com", "title": "Fallback Source"},
                "content": f"Fallback content for {gap.get('description', 'unknown gap')}",
                "relevance_score": 0.5,
                "gap_id": gap.get("id")
            }
            retrieved_info.append(info)
        
        state["retrieved_info"] = retrieved_info
        return state
    
    def _fallback_information_integration(self, state: TTDRState) -> TTDRState:
        """Fallback information integration using simple concatenation"""
        logger.info("Executing fallback information integration")
        
        if state.get("current_draft") and state.get("retrieved_info"):
            # Simple integration: append retrieved content to sections
            for info in state["retrieved_info"]:
                gap_id = info.get("gap_id")
                if gap_id:
                    # Find the gap and corresponding section
                    for gap in state.get("information_gaps", []):
                        if gap.get("id") == gap_id:
                            section_id = gap.get("section_id")
                            if section_id in state["current_draft"]["content"]:
                                current_content = state["current_draft"]["content"][section_id]
                                state["current_draft"]["content"][section_id] = f"{current_content}\n\n{info['content']}"
                            break
            
            state["current_draft"]["iteration"] = state["current_draft"].get("iteration", 0) + 1
        
        return state
    
    def _fallback_quality_assessment(self, state: TTDRState) -> TTDRState:
        """Fallback quality assessment using basic metrics"""
        logger.info("Executing fallback quality assessment")
        
        # Basic quality assessment
        if state.get("current_draft"):
            total_content = sum(len(content) for content in state["current_draft"]["content"].values())
            completeness = min(total_content / 1000, 1.0)  # Assume 1000 chars is complete
            
            quality_metrics = {
                "completeness": completeness,
                "coherence": 0.7,
                "accuracy": 0.6,
                "citation_quality": 0.5,
                "overall_score": (completeness + 0.7 + 0.6 + 0.5) / 4
            }
            
            state["quality_metrics"] = quality_metrics
        
        return state
    
    def _fallback_self_evolution(self, state: TTDRState) -> TTDRState:
        """Fallback self-evolution using basic improvements"""
        logger.info("Executing fallback self-evolution")
        
        # Create a basic evolution record
        evolution_record = {
            "timestamp": datetime.now().isoformat(),
            "component": "fallback_system",
            "improvement_type": "basic_enhancement",
            "description": "Applied fallback improvements",
            "performance_before": 0.5,
            "performance_after": 0.6
        }
        
        if "evolution_history" not in state:
            state["evolution_history"] = []
        state["evolution_history"].append(evolution_record)
        
        return state
    
    def _fallback_report_synthesis(self, state: TTDRState) -> TTDRState:
        """Fallback report synthesis using basic formatting"""
        logger.info("Executing fallback report synthesis")
        
        if state.get("current_draft"):
            # Create a basic final report
            report_parts = []
            report_parts.append(f"# {state['current_draft']['topic']}\n")
            
            for section_id, content in state["current_draft"]["content"].items():
                section_title = section_id.replace("_", " ").title()
                report_parts.append(f"## {section_title}\n")
                report_parts.append(f"{content}\n")
            
            state["final_report"] = "\n".join(report_parts)
        
        return state

class WorkflowRecoveryContext:
    """Context manager for automatic workflow recovery"""
    
    def __init__(self, recovery_manager: WorkflowRecoveryManager,
                 workflow_state: WorkflowState, node_name: str,
                 state: TTDRState):
        self.recovery_manager = recovery_manager
        self.workflow_state = workflow_state
        self.node_name = node_name
        self.state = state
        self.checkpoint = None
    
    def __enter__(self):
        # Create checkpoint before operation
        self.checkpoint = self.recovery_manager.create_checkpoint(
            self.state, self.workflow_state, self.node_name
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred, create recovery plan
            context = ErrorContext(
                component=self.node_name,
                operation=self.workflow_state.value,
                state=self.state
            )
            
            recovery_plan = self.recovery_manager.create_recovery_plan(
                exc_val, context, self.checkpoint
            )
            
            # Log the error and recovery plan
            logger.error(f"Error in {self.node_name}: {exc_val}")
            logger.info(f"Created recovery plan with strategy: {recovery_plan.recovery_strategy}")
            
            # Store recovery plan for later use
            self.recovery_manager.recovery_plans[self.checkpoint.id] = recovery_plan
        
        return False  # Don't suppress the exception

# Export main classes
__all__ = [
    "WorkflowRecoveryManager",
    "WorkflowCheckpoint", 
    "RecoveryPlan",
    "WorkflowState",
    "CheckpointType",
    "WorkflowRecoveryContext"
]