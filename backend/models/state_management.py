"""
State management utilities for TTD-DR LangGraph workflow.
Provides validation, transformation, and persistence mechanisms for workflow state.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pydantic import ValidationError

from .core import (
    TTDRState, Draft, InformationGap, RetrievedInfo, 
    QualityMetrics, EvolutionRecord, ResearchRequirements
)

logger = logging.getLogger(__name__)

class StateValidationError(Exception):
    """Raised when state validation fails"""
    pass

class StatePersistenceError(Exception):
    """Raised when state persistence operations fail"""
    pass

class TTDRStateManager:
    """Manages TTD-DR workflow state validation, transformation, and persistence"""
    
    def __init__(self, persistence_dir: str = "workflow_states"):
        """
        Initialize state manager with persistence directory
        
        Args:
            persistence_dir: Directory to store persisted states
        """
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
        
    def create_initial_state(self, topic: str, requirements: ResearchRequirements) -> TTDRState:
        """
        Create initial TTD-DR state for a new workflow
        
        Args:
            topic: Research topic
            requirements: Research requirements and constraints
            
        Returns:
            Initial TTDRState with default values
        """
        initial_state: TTDRState = {
            "topic": topic,
            "requirements": requirements,
            "current_draft": None,
            "information_gaps": [],
            "retrieved_info": [],
            "iteration_count": 0,
            "quality_metrics": None,
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
        
        # Validate the initial state
        self.validate_state(initial_state)
        return initial_state
    
    def validate_state(self, state: TTDRState) -> bool:
        """
        Validate TTD-DR state structure and data integrity
        
        Args:
            state: State to validate
            
        Returns:
            True if valid
            
        Raises:
            StateValidationError: If validation fails
        """
        try:
            # Check required fields
            required_fields = [
                "topic", "requirements", "current_draft", "information_gaps",
                "retrieved_info", "iteration_count", "quality_metrics",
                "evolution_history", "final_report", "error_log"
            ]
            
            for field in required_fields:
                if field not in state:
                    raise StateValidationError(f"Missing required field: {field}")
            
            # Validate field types and constraints
            if not isinstance(state["topic"], str) or not state["topic"].strip():
                raise StateValidationError("Topic must be a non-empty string")
            
            if not isinstance(state["requirements"], ResearchRequirements):
                raise StateValidationError("Requirements must be a ResearchRequirements instance")
            
            if state["current_draft"] is not None and not isinstance(state["current_draft"], Draft):
                raise StateValidationError("Current draft must be a Draft instance or None")
            
            if not isinstance(state["information_gaps"], list):
                raise StateValidationError("Information gaps must be a list")
            
            for gap in state["information_gaps"]:
                if not isinstance(gap, InformationGap):
                    raise StateValidationError("All information gaps must be InformationGap instances")
            
            if not isinstance(state["retrieved_info"], list):
                raise StateValidationError("Retrieved info must be a list")
            
            for info in state["retrieved_info"]:
                if not isinstance(info, RetrievedInfo):
                    raise StateValidationError("All retrieved info must be RetrievedInfo instances")
            
            if not isinstance(state["iteration_count"], int) or state["iteration_count"] < 0:
                raise StateValidationError("Iteration count must be a non-negative integer")
            
            if state["quality_metrics"] is not None and not isinstance(state["quality_metrics"], QualityMetrics):
                raise StateValidationError("Quality metrics must be a QualityMetrics instance or None")
            
            if not isinstance(state["evolution_history"], list):
                raise StateValidationError("Evolution history must be a list")
            
            for record in state["evolution_history"]:
                if not isinstance(record, EvolutionRecord):
                    raise StateValidationError("All evolution records must be EvolutionRecord instances")
            
            if state["final_report"] is not None and not isinstance(state["final_report"], str):
                raise StateValidationError("Final report must be a string or None")
            
            if not isinstance(state["error_log"], list):
                raise StateValidationError("Error log must be a list")
            
            # Validate business logic constraints
            max_iterations = state["requirements"].max_iterations
            if state["iteration_count"] > max_iterations:
                raise StateValidationError(f"Iteration count ({state['iteration_count']}) exceeds maximum ({max_iterations})")
            
            # Validate gap-info relationships
            gap_ids = {gap.id for gap in state["information_gaps"]}
            for info in state["retrieved_info"]:
                if info.gap_id and info.gap_id not in gap_ids:
                    raise StateValidationError(f"Retrieved info references non-existent gap: {info.gap_id}")
            
            logger.info("State validation successful")
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {str(e)}")
            raise StateValidationError(f"State validation failed: {str(e)}")
    
    def transform_state(self, state: TTDRState, transformations: Dict[str, Any]) -> TTDRState:
        """
        Apply transformations to state while maintaining validity
        
        Args:
            state: Current state
            transformations: Dictionary of field updates
            
        Returns:
            New state with transformations applied
        """
        # Create a copy of the current state
        new_state = state.copy()
        
        # Apply transformations
        for field, value in transformations.items():
            if field in new_state:
                new_state[field] = value
                logger.debug(f"Applied transformation: {field} = {type(value).__name__}")
            else:
                logger.warning(f"Attempted to transform unknown field: {field}")
        
        # Update metadata
        if "current_draft" in transformations and new_state["current_draft"]:
            new_state["current_draft"].metadata.updated_at = datetime.now()
        
        # Validate the transformed state
        self.validate_state(new_state)
        
        return new_state
    
    def increment_iteration(self, state: TTDRState) -> TTDRState:
        """
        Increment iteration count and update related metadata
        
        Args:
            state: Current state
            
        Returns:
            State with incremented iteration count
        """
        return self.transform_state(state, {
            "iteration_count": state["iteration_count"] + 1
        })
    
    def add_error(self, state: TTDRState, error_message: str) -> TTDRState:
        """
        Add error message to state error log
        
        Args:
            state: Current state
            error_message: Error message to add
            
        Returns:
            State with error added to log
        """
        timestamp = datetime.now().isoformat()
        error_entry = f"[{timestamp}] {error_message}"
        
        new_error_log = state["error_log"].copy()
        new_error_log.append(error_entry)
        
        return self.transform_state(state, {
            "error_log": new_error_log
        })
    
    def persist_state(self, state: TTDRState, state_id: str) -> str:
        """
        Persist state to disk for workflow continuity
        
        Args:
            state: State to persist
            state_id: Unique identifier for the state
            
        Returns:
            Path to persisted state file
            
        Raises:
            StatePersistenceError: If persistence fails
        """
        try:
            # Validate state before persisting
            self.validate_state(state)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{state_id}_{timestamp}.pkl"
            filepath = self.persistence_dir / filename
            
            # Serialize state using pickle for complex objects
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            # Also save a JSON version for human readability (simplified)
            json_filepath = filepath.with_suffix('.json')
            json_state = self._state_to_json_serializable(state)
            with open(json_filepath, 'w') as f:
                json.dump(json_state, f, indent=2, default=str)
            
            logger.info(f"State persisted to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to persist state: {str(e)}")
            raise StatePersistenceError(f"Failed to persist state: {str(e)}")
    
    def load_state(self, filepath: str) -> TTDRState:
        """
        Load persisted state from disk
        
        Args:
            filepath: Path to persisted state file
            
        Returns:
            Loaded TTDRState
            
        Raises:
            StatePersistenceError: If loading fails
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise StatePersistenceError(f"State file not found: {filepath}")
            
            # Load state from pickle file
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Validate loaded state
            self.validate_state(state)
            
            logger.info(f"State loaded from {filepath}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            raise StatePersistenceError(f"Failed to load state: {str(e)}")
    
    def list_persisted_states(self, state_id_prefix: Optional[str] = None) -> List[str]:
        """
        List all persisted state files
        
        Args:
            state_id_prefix: Optional prefix to filter by state ID
            
        Returns:
            List of state file paths
        """
        pattern = f"{state_id_prefix}_*.pkl" if state_id_prefix else "*.pkl"
        state_files = list(self.persistence_dir.glob(pattern))
        return [str(f) for f in sorted(state_files, reverse=True)]  # Most recent first
    
    def cleanup_old_states(self, keep_count: int = 10) -> int:
        """
        Clean up old persisted states, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent states to keep
            
        Returns:
            Number of states deleted
        """
        all_states = self.list_persisted_states()
        if len(all_states) <= keep_count:
            return 0
        
        states_to_delete = all_states[keep_count:]
        deleted_count = 0
        
        for state_path in states_to_delete:
            try:
                Path(state_path).unlink()
                # Also delete corresponding JSON file
                json_path = Path(state_path).with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete state file {state_path}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} old state files")
        return deleted_count
    
    def _state_to_json_serializable(self, state: TTDRState) -> Dict[str, Any]:
        """
        Convert state to JSON-serializable format for human-readable persistence
        
        Args:
            state: State to convert
            
        Returns:
            JSON-serializable dictionary
        """
        json_state = {}
        
        for key, value in state.items():
            if value is None:
                json_state[key] = None
            elif isinstance(value, (str, int, float, bool)):
                json_state[key] = value
            elif isinstance(value, list):
                json_state[key] = [
                    item.dict() if hasattr(item, 'dict') else str(item) 
                    for item in value
                ]
            elif hasattr(value, 'dict'):
                json_state[key] = value.dict()
            else:
                json_state[key] = str(value)
        
        return json_state

# Utility functions for common state operations
def create_workflow_state(topic: str, requirements: ResearchRequirements) -> TTDRState:
    """
    Convenience function to create initial workflow state
    
    Args:
        topic: Research topic
        requirements: Research requirements
        
    Returns:
        Initial TTDRState
    """
    manager = TTDRStateManager()
    return manager.create_initial_state(topic, requirements)

def validate_workflow_state(state: TTDRState) -> bool:
    """
    Convenience function to validate workflow state
    
    Args:
        state: State to validate
        
    Returns:
        True if valid
        
    Raises:
        StateValidationError: If validation fails
    """
    manager = TTDRStateManager()
    return manager.validate_state(state)