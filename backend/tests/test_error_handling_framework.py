"""
Comprehensive tests for the error handling framework.
Tests task 13.1: Create comprehensive error handling framework.
"""

import pytest
import logging
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from services.error_handling import (
    ErrorHandlingFramework, ErrorClassifier, RecoveryManager, GracefulDegradationManager,
    TTDRError, NetworkError, APIError, AuthenticationError, RateLimitError, 
    TimeoutError, ValidationError, ProcessingError, WorkflowError,
    ErrorSeverity, ErrorCategory, RecoveryStrategy, ErrorContext, ErrorRecord, RecoveryAction,
    error_context, async_error_context, handle_errors, global_error_framework
)

class TestErrorClassifier:
    """Test error classification functionality"""
    
    def setup_method(self):
        self.classifier = ErrorClassifier()
    
    def test_classify_ttdr_error(self):
        """Test classification of TTDRError instances"""
        error = NetworkError("Connection failed", severity=ErrorSeverity.HIGH)
        context = ErrorContext(component="test", operation="connect")
        
        record = self.classifier.classify_error(error, context)
        
        assert record.error_type == "NetworkError"
        assert record.category == ErrorCategory.NETWORK
        assert record.severity == ErrorSeverity.HIGH
        assert record.message == "Connection failed"
        assert record.context == context
    
    def test_classify_standard_exceptions(self):
        """Test classification of standard Python exceptions"""
        test_cases = [
            (ConnectionError("Network issue"), ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            (ValueError("Invalid value"), ErrorCategory.VALIDATION, ErrorSeverity.LOW),
            (KeyError("Missing key"), ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM),
            (FileNotFoundError("File missing"), ErrorCategory.STORAGE, ErrorSeverity.MEDIUM),
        ]
        
        for error, expected_category, expected_severity in test_cases:
            record = self.classifier.classify_error(error)
            assert record.category == expected_category
            assert record.severity == expected_severity
    
    def test_classify_by_message_patterns(self):
        """Test classification based on error message patterns"""
        test_cases = [
            (Exception("Connection timeout"), ErrorCategory.TIMEOUT),
            (Exception("Rate limit exceeded"), ErrorCategory.RATE_LIMIT),
            (Exception("Unauthorized access"), ErrorCategory.AUTHENTICATION),
            (Exception("Network error occurred"), ErrorCategory.NETWORK),
        ]
        
        for error, expected_category in test_cases:
            record = self.classifier.classify_error(error)
            assert record.category == expected_category
    
    def test_unknown_error_classification(self):
        """Test classification of unknown errors"""
        error = Exception("Unknown error")
        record = self.classifier.classify_error(error)
        
        assert record.category == ErrorCategory.UNKNOWN
        assert record.severity == ErrorSeverity.MEDIUM

class TestRecoveryManager:
    """Test recovery management functionality"""
    
    def setup_method(self):
        self.recovery_manager = RecoveryManager()
    
    def test_get_recovery_action_retry(self):
        """Test retry recovery action generation"""
        error_record = ErrorRecord(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retry_count=0
        )
        
        action = self.recovery_manager.get_recovery_action(error_record)
        
        assert action is not None
        assert action.strategy == RecoveryStrategy.RETRY
        assert action.max_attempts == 3
        assert action.backoff_factor == 2.0
    
    def test_get_recovery_action_max_retries_exceeded(self):
        """Test recovery when max retries exceeded"""
        error_record = ErrorRecord(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retry_count=5
        )
        
        action = self.recovery_manager.get_recovery_action(error_record)
        
        # Should not retry when max retries exceeded
        assert action is None or action.strategy != RecoveryStrategy.RETRY
    
    def test_get_recovery_action_critical_severity(self):
        """Test recovery for critical errors"""
        error_record = ErrorRecord(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.CRITICAL
        )
        
        action = self.recovery_manager.get_recovery_action(error_record)
        
        assert action is not None
        assert action.strategy == RecoveryStrategy.ABORT
    
    def test_register_fallback_handler(self):
        """Test registering and using fallback handlers"""
        mock_handler = Mock(return_value="fallback_result")
        self.recovery_manager.register_fallback_handler(ErrorCategory.PROCESSING, mock_handler)
        
        error_record = ErrorRecord(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM
        )
        
        action = self.recovery_manager.get_recovery_action(error_record)
        
        assert action is not None
        assert action.strategy == RecoveryStrategy.FALLBACK
        assert action.action == mock_handler

class TestGracefulDegradationManager:
    """Test graceful degradation functionality"""
    
    def setup_method(self):
        self.degradation_manager = GracefulDegradationManager()
    
    def test_register_and_apply_degradation_strategy(self):
        """Test registering and applying degradation strategies"""
        mock_strategy = Mock(return_value="degraded_result")
        self.degradation_manager.register_degradation_strategy("test_component", mock_strategy)
        
        # Mark component as unhealthy
        self.degradation_manager.update_component_health("test_component", False)
        
        # Check degradation should be applied
        assert self.degradation_manager.should_degrade("test_component")
        
        # Apply degradation
        result = self.degradation_manager.apply_degradation("test_component", "test_operation", "arg1", key="value")
        
        assert result == "degraded_result"
        mock_strategy.assert_called_once_with("test_operation", "arg1", key="value")
    
    def test_healthy_component_no_degradation(self):
        """Test that healthy components don't degrade"""
        self.degradation_manager.update_component_health("healthy_component", True)
        
        assert not self.degradation_manager.should_degrade("healthy_component")
    
    def test_unknown_component_no_degradation(self):
        """Test that unknown components don't degrade"""
        assert not self.degradation_manager.should_degrade("unknown_component")

class TestErrorHandlingFramework:
    """Test the main error handling framework"""
    
    def setup_method(self):
        # Use temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        config = {"log_directory": self.temp_dir}
        self.framework = ErrorHandlingFramework(config)
    
    def teardown_method(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_handle_error_with_recovery(self):
        """Test error handling with successful recovery"""
        # Register a fallback handler
        mock_handler = Mock(return_value="recovered_result")
        self.framework.recovery_manager.register_fallback_handler(ErrorCategory.PROCESSING, mock_handler)
        
        error = ProcessingError("Processing failed")
        context = ErrorContext(component="test", operation="process")
        
        result = self.framework.handle_error(error, context)
        
        assert result == "recovered_result"
        mock_handler.assert_called_once()
        
        # Check error was recorded
        assert len(self.framework.error_history) == 1
        assert self.framework.error_history[0].resolved
    
    def test_handle_error_without_recovery(self):
        """Test error handling when no recovery is possible"""
        error = Exception("Unrecoverable error")
        context = ErrorContext(component="test", operation="fail")
        
        with pytest.raises(Exception, match="Unrecoverable error"):
            self.framework.handle_error(error, context)
        
        # Check error was recorded
        assert len(self.framework.error_history) == 1
        assert not self.framework.error_history[0].resolved
    
    def test_error_history_management(self):
        """Test error history size management"""
        # Set small max history size
        self.framework.max_history_size = 3
        
        # Add more errors than max size
        for i in range(5):
            try:
                self.framework.handle_error(Exception(f"Error {i}"))
            except:
                pass
        
        # Should only keep the most recent errors
        assert len(self.framework.error_history) == 3
        assert self.framework.error_history[-1].message == "Error 4"
    
    def test_get_error_statistics(self):
        """Test error statistics generation"""
        # Add some test errors
        errors = [
            NetworkError("Network error 1"),
            APIError("API error 1"),
            NetworkError("Network error 2"),
        ]
        
        for error in errors:
            try:
                self.framework.handle_error(error)
            except:
                pass
        
        stats = self.framework.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["category_breakdown"]["network"] == 2
        assert stats["category_breakdown"]["api"] == 1
        assert stats["resolution_rate"] == 0  # No recovery handlers registered
    
    def test_clear_error_history(self):
        """Test clearing error history"""
        # Add some errors
        for i in range(3):
            try:
                self.framework.handle_error(Exception(f"Error {i}"))
            except:
                pass
        
        assert len(self.framework.error_history) == 3
        
        self.framework.clear_error_history()
        
        assert len(self.framework.error_history) == 0

class TestErrorContextManagers:
    """Test error context managers and decorators"""
    
    def setup_method(self):
        self.framework = ErrorHandlingFramework()
    
    def test_error_context_manager_success(self):
        """Test error context manager with successful operation"""
        with error_context("test_component", "test_operation", self.framework) as ctx:
            assert ctx.component == "test_component"
            assert ctx.operation == "test_operation"
            result = "success"
        
        assert result == "success"
    
    def test_error_context_manager_with_error(self):
        """Test error context manager with error"""
        with pytest.raises(ValueError):
            with error_context("test_component", "test_operation", self.framework):
                raise ValueError("Test error")
        
        # Error should be recorded
        assert len(self.framework.error_history) == 1
        assert self.framework.error_history[0].context.component == "test_component"
    
    @pytest.mark.asyncio
    async def test_async_error_context_manager(self):
        """Test async error context manager"""
        async with async_error_context("async_component", "async_operation", self.framework) as ctx:
            assert ctx.component == "async_component"
            assert ctx.operation == "async_operation"
            result = "async_success"
        
        assert result == "async_success"
    
    def test_handle_errors_decorator_success(self):
        """Test error handling decorator with successful function"""
        @handle_errors("decorated_component", "decorated_operation", self.framework)
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5
    
    def test_handle_errors_decorator_with_error(self):
        """Test error handling decorator with error"""
        @handle_errors("decorated_component", "decorated_operation", self.framework)
        def failing_function():
            raise ValueError("Decorator test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Error should be recorded
        assert len(self.framework.error_history) == 1
        assert self.framework.error_history[0].context.component == "decorated_component"

class TestSpecificErrorTypes:
    """Test specific error types and their properties"""
    
    def test_network_error(self):
        """Test NetworkError properties"""
        error = NetworkError("Connection failed", severity=ErrorSeverity.HIGH)
        
        assert error.category == ErrorCategory.NETWORK
        assert error.severity == ErrorSeverity.HIGH
        assert error.message == "Connection failed"
    
    def test_api_error_with_status_code(self):
        """Test APIError with status code"""
        error = APIError("API request failed", status_code=404)
        
        assert error.category == ErrorCategory.API
        assert error.status_code == 404
    
    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after"""
        error = RateLimitError("Rate limit exceeded", retry_after=60.0)
        
        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.retry_after == 60.0
    
    def test_timeout_error_with_duration(self):
        """Test TimeoutError with timeout duration"""
        error = TimeoutError("Operation timed out", timeout_duration=30.0)
        
        assert error.category == ErrorCategory.TIMEOUT
        assert error.timeout_duration == 30.0
    
    def test_validation_error_with_field(self):
        """Test ValidationError with field information"""
        error = ValidationError("Invalid field value", field="email")
        
        assert error.category == ErrorCategory.VALIDATION
        assert error.field == "email"
    
    def test_workflow_error_with_node(self):
        """Test WorkflowError with node information"""
        error = WorkflowError("Node execution failed", node="draft_generator")
        
        assert error.category == ErrorCategory.WORKFLOW
        assert error.node == "draft_generator"

class TestErrorRecoveryExecution:
    """Test error recovery execution scenarios"""
    
    def setup_method(self):
        self.framework = ErrorHandlingFramework()
    
    def test_retry_recovery_success(self):
        """Test successful retry recovery"""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        # Create recovery action
        recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            action=flaky_function,
            max_attempts=3,
            backoff_factor=1.0  # No delay for testing
        )
        
        error_record = ErrorRecord(category=ErrorCategory.NETWORK)
        
        result = self.framework._execute_recovery_action(recovery_action, error_record)
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_recovery_failure(self):
        """Test retry recovery that ultimately fails"""
        def always_failing_function():
            raise NetworkError("Persistent failure")
        
        recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            action=always_failing_function,
            max_attempts=2,
            backoff_factor=1.0
        )
        
        error_record = ErrorRecord(category=ErrorCategory.NETWORK)
        
        with pytest.raises(NetworkError, match="Persistent failure"):
            self.framework._execute_recovery_action(recovery_action, error_record)
    
    def test_fallback_recovery(self):
        """Test fallback recovery execution"""
        def fallback_function(error_record):
            return f"fallback_result_for_{error_record.category.value}"
        
        recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action=fallback_function
        )
        
        error_record = ErrorRecord(category=ErrorCategory.PROCESSING)
        
        result = self.framework._execute_recovery_action(recovery_action, error_record)
        
        assert result == "fallback_result_for_processing"

class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        config = {"log_directory": self.temp_dir}
        self.framework = ErrorHandlingFramework(config)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow"""
        # Register fallback handler
        def processing_fallback(error_record):
            return "processed_with_fallback"
        
        self.framework.recovery_manager.register_fallback_handler(
            ErrorCategory.PROCESSING, processing_fallback
        )
        
        # Register degradation strategy
        def component_degradation(operation, *args, **kwargs):
            return "degraded_result"
        
        self.framework.degradation_manager.register_degradation_strategy(
            "test_component", component_degradation
        )
        
        # Mark component as unhealthy
        self.framework.degradation_manager.update_component_health("test_component", False)
        
        # Test error handling
        error = ProcessingError("Processing failed")
        context = ErrorContext(component="test_component", operation="process")
        
        result = self.framework.handle_error(error, context)
        
        assert result == "processed_with_fallback"
        
        # Verify error was recorded and resolved
        assert len(self.framework.error_history) == 1
        assert self.framework.error_history[0].resolved
        assert self.framework.error_history[0].resolution_strategy == RecoveryStrategy.FALLBACK
    
    def test_error_logging_integration(self):
        """Test error logging integration"""
        error = NetworkError("Network connection failed", severity=ErrorSeverity.HIGH)
        context = ErrorContext(component="network_client", operation="connect")
        
        try:
            self.framework.handle_error(error, context)
        except:
            pass
        
        # Check that log file was created
        log_file = Path(self.temp_dir) / "errors.log"
        assert log_file.exists()
        
        # Check log content
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert "Network connection failed" in log_content
            assert "network_client" in log_content

if __name__ == "__main__":
    pytest.main([__file__, "-v"])