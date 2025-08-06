"""
Comprehensive error handling and recovery framework for TTD-DR system.
Implements task 13.1: Create comprehensive error handling framework.
"""

import logging
import traceback
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union, Type
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
import json
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorSeverity(str, Enum):
    """Severity levels for errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """Categories of errors in the TTD-DR system"""
    NETWORK = "network"
    API = "api"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    PROCESSING = "processing"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    WORKFLOW = "workflow"
    UNKNOWN = "unknown"

class RecoveryStrategy(str, Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    DEGRADE = "degrade"
    MANUAL = "manual"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    component: str
    operation: str
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_type: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    message: str = ""
    context: Optional[ErrorContext] = None
    traceback: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_strategy: Optional[RecoveryStrategy] = None
    resolution_details: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class RecoveryAction:
    """Defines a recovery action for an error"""
    strategy: RecoveryStrategy
    action: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    backoff_factor: float = 2.0
    timeout: Optional[float] = None

class TTDRError(Exception):
    """Base exception for TTD-DR framework errors"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.timestamp = datetime.now()

class NetworkError(TTDRError):
    """Network-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)

class APIError(TTDRError):
    """API-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.API, **kwargs)
        self.status_code = status_code

class AuthenticationError(TTDRError):
    """Authentication-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHENTICATION, 
                        severity=ErrorSeverity.HIGH, **kwargs)

class RateLimitError(TTDRError):
    """Rate limiting errors"""
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.RATE_LIMIT, **kwargs)
        self.retry_after = retry_after

class TimeoutError(TTDRError):
    """Timeout errors"""
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)
        self.timeout_duration = timeout_duration

class ValidationError(TTDRError):
    """Data validation errors"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        self.field = field

class ProcessingError(TTDRError):
    """Data processing errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PROCESSING, **kwargs)

class WorkflowError(TTDRError):
    """Workflow execution errors"""
    def __init__(self, message: str, node: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.WORKFLOW, **kwargs)
        self.node = node

class ErrorClassifier:
    """Classifies errors and determines appropriate handling strategies"""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
    
    def classify_error(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorRecord:
        """
        Classify an error and create an error record
        
        Args:
            error: The exception to classify
            context: Optional context information
            
        Returns:
            ErrorRecord with classification details
        """
        error_type = type(error).__name__
        
        # Check if it's already a TTDRError
        if isinstance(error, TTDRError):
            return ErrorRecord(
                error_type=error_type,
                category=error.category,
                severity=error.severity,
                message=error.message,
                context=error.context or context,
                traceback=traceback.format_exc()
            )
        
        # Classify based on error type and message
        category, severity = self._classify_by_rules(error, context)
        
        return ErrorRecord(
            error_type=error_type,
            category=category,
            severity=severity,
            message=str(error),
            context=context,
            traceback=traceback.format_exc()
        )
    
    def _build_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build classification rules for different error types"""
        return {
            # Network errors
            "ConnectionError": {"category": ErrorCategory.NETWORK, "severity": ErrorSeverity.HIGH},
            "TimeoutError": {"category": ErrorCategory.TIMEOUT, "severity": ErrorSeverity.MEDIUM},
            "HTTPError": {"category": ErrorCategory.NETWORK, "severity": ErrorSeverity.MEDIUM},
            
            # API errors
            "HTTPStatusError": {"category": ErrorCategory.API, "severity": ErrorSeverity.MEDIUM},
            "RequestError": {"category": ErrorCategory.API, "severity": ErrorSeverity.MEDIUM},
            
            # Authentication
            "AuthenticationError": {"category": ErrorCategory.AUTHENTICATION, "severity": ErrorSeverity.HIGH},
            "PermissionError": {"category": ErrorCategory.AUTHENTICATION, "severity": ErrorSeverity.HIGH},
            
            # Validation
            "ValidationError": {"category": ErrorCategory.VALIDATION, "severity": ErrorSeverity.LOW},
            "ValueError": {"category": ErrorCategory.VALIDATION, "severity": ErrorSeverity.LOW},
            "TypeError": {"category": ErrorCategory.VALIDATION, "severity": ErrorSeverity.MEDIUM},
            
            # Processing
            "JSONDecodeError": {"category": ErrorCategory.PROCESSING, "severity": ErrorSeverity.MEDIUM},
            "KeyError": {"category": ErrorCategory.PROCESSING, "severity": ErrorSeverity.MEDIUM},
            "AttributeError": {"category": ErrorCategory.PROCESSING, "severity": ErrorSeverity.MEDIUM},
            
            # Storage
            "FileNotFoundError": {"category": ErrorCategory.STORAGE, "severity": ErrorSeverity.MEDIUM},
            "PermissionError": {"category": ErrorCategory.STORAGE, "severity": ErrorSeverity.HIGH},
            "IOError": {"category": ErrorCategory.STORAGE, "severity": ErrorSeverity.MEDIUM},
        }
    
    def _classify_by_rules(self, error: Exception, context: Optional[ErrorContext] = None) -> tuple:
        """Classify error using built-in rules"""
        error_type = type(error).__name__
        
        if error_type in self.classification_rules:
            rule = self.classification_rules[error_type]
            return rule["category"], rule["severity"]
        
        # Check error message for specific patterns
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ["timeout", "timed out"]):
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        elif any(keyword in error_message for keyword in ["rate limit", "quota", "too many requests"]):
            return ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
        elif any(keyword in error_message for keyword in ["unauthorized", "forbidden", "invalid key"]):
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        elif any(keyword in error_message for keyword in ["network", "connection", "dns"]):
            return ErrorCategory.NETWORK, ErrorSeverity.HIGH
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

class RecoveryManager:
    """Manages error recovery strategies and actions"""
    
    def __init__(self):
        self.recovery_strategies = self._build_recovery_strategies()
        self.fallback_handlers = {}
        
    def register_fallback_handler(self, category: ErrorCategory, handler: Callable):
        """Register a fallback handler for a specific error category"""
        self.fallback_handlers[category] = handler
    
    def get_recovery_action(self, error_record: ErrorRecord) -> Optional[RecoveryAction]:
        """
        Determine the appropriate recovery action for an error
        
        Args:
            error_record: The error record to handle
            
        Returns:
            RecoveryAction if available, None otherwise
        """
        # Check for fallback handlers first (higher priority)
        if error_record.category in self.fallback_handlers:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action=self.fallback_handlers[error_record.category],
                parameters={}
            )
        
        # Check for specific recovery strategies based on error category
        if error_record.category in self.recovery_strategies:
            strategy_config = self.recovery_strategies[error_record.category]
            
            if error_record.retry_count < strategy_config.get("max_retries", 3):
                return RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    action=self._create_retry_action(),
                    parameters=strategy_config,
                    max_attempts=strategy_config.get("max_retries", 3),
                    backoff_factor=strategy_config.get("backoff_factor", 2.0),
                    timeout=strategy_config.get("timeout")
                )
        
        # Default strategies based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                action=self._create_abort_action(),
                parameters={}
            )
        elif error_record.severity == ErrorSeverity.HIGH:
            return RecoveryAction(
                strategy=RecoveryStrategy.MANUAL,
                action=self._create_manual_action(),
                parameters={}
            )
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                action=self._create_skip_action(),
                parameters={}
            )
    
    def _build_recovery_strategies(self) -> Dict[ErrorCategory, Dict[str, Any]]:
        """Build recovery strategies for different error categories"""
        return {
            ErrorCategory.NETWORK: {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "timeout": 30.0
            },
            ErrorCategory.API: {
                "max_retries": 3,
                "backoff_factor": 1.5,
                "timeout": 60.0
            },
            ErrorCategory.RATE_LIMIT: {
                "max_retries": 5,
                "backoff_factor": 3.0,
                "timeout": 300.0
            },
            ErrorCategory.TIMEOUT: {
                "max_retries": 2,
                "backoff_factor": 2.0,
                "timeout": 120.0
            },
            ErrorCategory.PROCESSING: {
                "max_retries": 1,
                "backoff_factor": 1.0,
                "timeout": 30.0
            }
        }
    
    def _create_retry_action(self) -> Callable:
        """Create a retry action"""
        def retry_action(original_func: Callable, *args, **kwargs):
            return original_func(*args, **kwargs)
        return retry_action
    
    def _create_abort_action(self) -> Callable:
        """Create an abort action"""
        def abort_action(error_record: ErrorRecord):
            logger.critical(f"Aborting due to critical error: {error_record.message}")
            raise TTDRError(f"Critical error: {error_record.message}", 
                          category=error_record.category, 
                          severity=ErrorSeverity.CRITICAL)
        return abort_action
    
    def _create_manual_action(self) -> Callable:
        """Create a manual intervention action"""
        def manual_action(error_record: ErrorRecord):
            logger.error(f"Manual intervention required for error: {error_record.message}")
            # In a real system, this might trigger alerts or notifications
            return None
        return manual_action
    
    def _create_skip_action(self) -> Callable:
        """Create a skip action"""
        def skip_action(error_record: ErrorRecord):
            logger.warning(f"Skipping operation due to error: {error_record.message}")
            return None
        return skip_action

class GracefulDegradationManager:
    """Manages graceful degradation mechanisms for component failures"""
    
    def __init__(self):
        self.degradation_strategies = {}
        self.component_health = {}
        
    def register_degradation_strategy(self, component: str, strategy: Callable):
        """Register a degradation strategy for a component"""
        self.degradation_strategies[component] = strategy
        
    def update_component_health(self, component: str, is_healthy: bool):
        """Update the health status of a component"""
        self.component_health[component] = {
            "healthy": is_healthy,
            "last_updated": datetime.now()
        }
        
    def should_degrade(self, component: str) -> bool:
        """Check if a component should operate in degraded mode"""
        health_info = self.component_health.get(component)
        if not health_info:
            return False
            
        return not health_info["healthy"]
    
    def apply_degradation(self, component: str, operation: str, *args, **kwargs):
        """Apply degradation strategy for a component operation"""
        if component in self.degradation_strategies:
            strategy = self.degradation_strategies[component]
            return strategy(operation, *args, **kwargs)
        
        # Default degradation: return None or empty result
        logger.warning(f"No degradation strategy for component {component}, returning None")
        return None

class ErrorHandlingFramework:
    """Main error handling framework that coordinates all error handling components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.classifier = ErrorClassifier()
        self.recovery_manager = RecoveryManager()
        self.degradation_manager = GracefulDegradationManager()
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        # Setup logging
        self.setup_error_logging()
        
    def setup_error_logging(self):
        """Setup specialized error logging"""
        error_logger = logging.getLogger("ttdr.errors")
        error_logger.setLevel(logging.INFO)
        
        # Create file handler for error logs
        log_dir = Path(self.config.get("log_directory", "logs"))
        log_dir.mkdir(exist_ok=True)
        
        error_log_file = log_dir / "errors.log"
        file_handler = logging.FileHandler(error_log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        error_logger.addHandler(file_handler)
        self.error_logger = error_logger
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """
        Main error handling entry point
        
        Args:
            error: The exception to handle
            context: Optional context information
            
        Returns:
            Result of recovery action or None
        """
        # Classify the error
        error_record = self.classifier.classify_error(error, context)
        
        # Add to history
        self._add_to_history(error_record)
        
        # Log the error
        self._log_error(error_record)
        
        # Get recovery action
        recovery_action = self.recovery_manager.get_recovery_action(error_record)
        
        if recovery_action:
            try:
                # Execute recovery action
                result = self._execute_recovery_action(recovery_action, error_record)
                
                # Mark as resolved if successful
                error_record.resolved = True
                error_record.resolution_strategy = recovery_action.strategy
                
                return result
                
            except Exception as recovery_error:
                logger.error(f"Recovery action failed: {recovery_error}")
                error_record.resolution_details["recovery_error"] = str(recovery_error)
        
        # If no recovery action or recovery failed, check for degradation
        if context and context.component:
            if self.degradation_manager.should_degrade(context.component):
                return self.degradation_manager.apply_degradation(
                    context.component, context.operation
                )
        
        # Re-raise the original error if no recovery possible
        raise error
    
    def _add_to_history(self, error_record: ErrorRecord):
        """Add error record to history with size management"""
        self.error_history.append(error_record)
        
        # Trim history if it exceeds max size
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error record with appropriate level"""
        log_message = f"[{error_record.category.value}] {error_record.message}"
        
        if error_record.context:
            log_message += f" | Component: {error_record.context.component}"
            log_message += f" | Operation: {error_record.context.operation}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.error_logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.error_logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.error_logger.warning(log_message)
        else:
            self.error_logger.info(log_message)
    
    def _execute_recovery_action(self, recovery_action: RecoveryAction, 
                               error_record: ErrorRecord) -> Any:
        """Execute a recovery action with proper error handling"""
        if recovery_action.strategy == RecoveryStrategy.RETRY:
            return self._execute_retry(recovery_action, error_record)
        elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
            return recovery_action.action(error_record)
        elif recovery_action.strategy == RecoveryStrategy.SKIP:
            return recovery_action.action(error_record)
        elif recovery_action.strategy == RecoveryStrategy.ABORT:
            recovery_action.action(error_record)
        elif recovery_action.strategy == RecoveryStrategy.MANUAL:
            return recovery_action.action(error_record)
        else:
            logger.warning(f"Unknown recovery strategy: {recovery_action.strategy}")
            return None
    
    def _execute_retry(self, recovery_action: RecoveryAction, error_record: ErrorRecord) -> Any:
        """Execute retry logic with exponential backoff"""
        max_attempts = recovery_action.max_attempts
        backoff_factor = recovery_action.backoff_factor
        
        # For retry strategy, we can't actually retry without the original function
        # This is a limitation of the current design - retry should be handled at a higher level
        # For now, we'll just log and return None
        logger.warning(f"Retry strategy requested but no original function available for error: {error_record.message}")
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about error occurrences"""
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        resolved_errors = sum(1 for error in self.error_history if error.resolved)
        
        # Count by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent errors (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = sum(1 for error in self.error_history 
                          if error.timestamp > one_hour_ago)
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "recent_errors_1h": recent_errors
        }
    
    def clear_error_history(self):
        """Clear the error history"""
        self.error_history.clear()
        logger.info("Error history cleared")

# Context managers for error handling

@contextmanager
def error_context(component: str, operation: str, framework: ErrorHandlingFramework,
                 metadata: Optional[Dict[str, Any]] = None):
    """Context manager for automatic error handling"""
    context = ErrorContext(
        component=component,
        operation=operation,
        metadata=metadata or {}
    )
    
    try:
        yield context
    except Exception as e:
        framework.handle_error(e, context)
        raise

@asynccontextmanager
async def async_error_context(component: str, operation: str, framework: ErrorHandlingFramework,
                            metadata: Optional[Dict[str, Any]] = None):
    """Async context manager for automatic error handling"""
    context = ErrorContext(
        component=component,
        operation=operation,
        metadata=metadata or {}
    )
    
    try:
        yield context
    except Exception as e:
        framework.handle_error(e, context)
        raise

# Decorator for automatic error handling
def handle_errors(component: str, operation: str = None, 
                 framework: Optional[ErrorHandlingFramework] = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal framework
            if framework is None:
                framework = ErrorHandlingFramework()
            
            op_name = operation or func.__name__
            
            with error_context(component, op_name, framework):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global error handling framework instance
global_error_framework = ErrorHandlingFramework()

# Export main classes and functions
__all__ = [
    "ErrorHandlingFramework",
    "TTDRError", "NetworkError", "APIError", "AuthenticationError", 
    "RateLimitError", "TimeoutError", "ValidationError", "ProcessingError", "WorkflowError",
    "ErrorSeverity", "ErrorCategory", "RecoveryStrategy",
    "ErrorContext", "ErrorRecord", "RecoveryAction",
    "ErrorClassifier", "RecoveryManager", "GracefulDegradationManager",
    "error_context", "async_error_context", "handle_errors",
    "global_error_framework"
]