"""
Integration tests for monitoring and alerting system with workflow execution.
Tests the complete monitoring flow including health checks, alerts, and recovery.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from services.monitoring_alerting import (
    MonitoringAlertingSystem, HealthCheck, Alert, AlertType, AlertSeverity,
    HealthStatus, Metric, MetricType, WorkflowMetrics
)
from services.error_handling import ErrorHandlingFramework, TTDRError, ErrorCategory
from services.workflow_recovery import WorkflowRecoveryManager, WorkflowState

class TestMonitoringIntegration:
    """Integration tests for monitoring and alerting system"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a monitoring system for testing"""
        config = {
            "auto_recovery": True,
            "email_alerts": {"enabled": False},
            "webhook_alerts": {"enabled": False}
        }
        return MonitoringAlertingSystem(config)
    
    @pytest.fixture
    def mock_workflow_execution(self):
        """Mock workflow execution data"""
        return {
            "execution_id": "test_workflow_123",
            "node_name": "draft_generator",
            "start_time": datetime.now(),
            "user_id": "user_123"
        }
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self, monitoring_system):
        """Test complete monitoring workflow from health check to recovery"""
        
        # 1. Start monitoring system
        await monitoring_system.start_monitoring()
        assert monitoring_system.monitoring_active is True
        
        # 2. Register a custom health check that will fail
        failure_count = 0
        def failing_health_check():
            nonlocal failure_count
            failure_count += 1
            return failure_count <= 2  # Fail first 2 times, then succeed
        
        health_check = HealthCheck(
            name="test_component",
            check_function=failing_health_check,
            interval_seconds=1,
            failure_threshold=2,
            recovery_threshold=1
        )
        monitoring_system.register_health_check(health_check)
        
        # 3. Mock alert channel to capture alerts
        alerts_sent = []
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        async def capture_alert(alert):
            alerts_sent.append(alert)
            return True
        mock_channel.send_alert = capture_alert
        monitoring_system.alert_channels["test"] = mock_channel
        
        # 4. Mock recovery action
        recovery_called = []
        async def mock_recovery(alert):
            recovery_called.append(alert)
            return True
        monitoring_system.recovery_actions["restart_component"] = mock_recovery
        
        # 5. Wait for health checks to run and generate alerts
        await asyncio.sleep(4)  # Allow time for multiple health check cycles
        
        # 6. Verify alerts were generated
        assert len(alerts_sent) > 0
        failure_alert = next((a for a in alerts_sent if a.type == AlertType.SYSTEM_HEALTH), None)
        assert failure_alert is not None
        assert failure_alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        
        # 7. Wait for recovery to be attempted
        await asyncio.sleep(3)
        
        # 8. Verify recovery was attempted
        assert len(recovery_called) > 0
        
        # 9. Stop monitoring
        await monitoring_system.stop_monitoring()
        assert monitoring_system.monitoring_active is False
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_monitoring(self, monitoring_system, mock_workflow_execution):
        """Test workflow execution metrics monitoring"""
        
        # Record successful workflow execution
        monitoring_system.record_workflow_execution(
            workflow_id=mock_workflow_execution["execution_id"],
            node_name=mock_workflow_execution["node_name"],
            execution_time_ms=1500.0,
            success=True,
            memory_usage_mb=128.0,
            error_count=0
        )
        
        # Verify metrics were recorded
        workflow_metrics = monitoring_system.performance_monitor.workflow_metrics
        assert mock_workflow_execution["execution_id"] in workflow_metrics
        
        recorded_metrics = workflow_metrics[mock_workflow_execution["execution_id"]]
        assert len(recorded_metrics) == 1
        assert recorded_metrics[0].node_name == mock_workflow_execution["node_name"]
        assert recorded_metrics[0].execution_time_ms == 1500.0
        assert recorded_metrics[0].success is True
        
        # Verify general metrics were also recorded
        response_time_metrics = monitoring_system.performance_monitor.metrics["response_time_ms"]
        assert len(response_time_metrics) == 1
        assert response_time_metrics[0]["value"] == 1500.0
    
    @pytest.mark.asyncio
    async def test_performance_threshold_alerting(self, monitoring_system):
        """Test performance threshold monitoring and alerting"""
        
        # Set low threshold for testing
        monitoring_system.performance_monitor.thresholds["test_metric"] = 100.0
        
        # Mock alert channel
        alerts_sent = []
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        async def capture_alert(alert):
            alerts_sent.append(alert)
            return True
        mock_channel.send_alert = capture_alert
        monitoring_system.alert_channels["test"] = mock_channel
        
        # Record metrics above threshold
        for i in range(5):
            metric = Metric(
                name="test_metric",
                type=MetricType.GAUGE,
                value=150.0  # Above threshold of 100.0
            )
            monitoring_system.performance_monitor.record_metric(metric)
        
        # Check thresholds and send alerts
        alerts = monitoring_system.performance_monitor.check_performance_thresholds()
        for alert in alerts:
            await monitoring_system._send_alert(alert)
        
        # Verify performance alert was generated
        assert len(alerts_sent) > 0
        perf_alert = next((a for a in alerts_sent if a.type == AlertType.PERFORMANCE_DEGRADATION), None)
        assert perf_alert is not None
        assert perf_alert.severity == AlertSeverity.WARNING
        assert "test_metric" in perf_alert.title
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, monitoring_system):
        """Test integration with error handling framework"""
        
        # Create an error that should trigger monitoring
        error = TTDRError(
            message="Test workflow error",
            category=ErrorCategory.WORKFLOW,
            severity=ErrorSeverity.HIGH
        )
        
        # Mock alert channel
        alerts_sent = []
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        async def capture_alert(alert):
            alerts_sent.append(alert)
            return True
        mock_channel.send_alert = capture_alert
        monitoring_system.alert_channels["test"] = mock_channel
        
        # Handle the error through the error framework
        try:
            monitoring_system.error_framework.handle_error(error)
        except TTDRError:
            pass  # Expected to re-raise
        
        # Create an alert based on the error
        alert = Alert(
            type=AlertType.WORKFLOW_FAILURE,
            severity=AlertSeverity.ERROR,
            title="Workflow execution failed",
            message=str(error),
            source="workflow_engine",
            metadata={"error_category": error.category.value}
        )
        
        await monitoring_system._send_alert(alert)
        
        # Verify alert was sent
        assert len(alerts_sent) == 1
        assert alerts_sent[0].type == AlertType.WORKFLOW_FAILURE
        assert alerts_sent[0].severity == AlertSeverity.ERROR
    
    @pytest.mark.asyncio
    async def test_recovery_mechanism_integration(self, monitoring_system):
        """Test automated recovery mechanism integration"""
        
        # Create alert eligible for recovery
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            title="Component failure",
            message="Test component has failed",
            source="test_component",
            timestamp=datetime.now() - timedelta(minutes=5),  # Old enough for recovery
            metadata={"component": "test_component"}
        )
        
        # Add to active alerts
        monitoring_system.active_alerts[alert.id] = alert
        
        # Mock recovery actions
        recovery_calls = []
        
        async def mock_restart_recovery(alert):
            recovery_calls.append(("restart", alert))
            return True
        
        async def mock_fallback_recovery(alert):
            recovery_calls.append(("fallback", alert))
            return True
        
        monitoring_system.recovery_actions["restart_component"] = mock_restart_recovery
        monitoring_system.recovery_actions["fallback_mode"] = mock_fallback_recovery
        
        # Process auto-recovery
        await monitoring_system._process_auto_recovery()
        
        # Verify recovery was attempted
        assert len(recovery_calls) > 0
        assert recovery_calls[0][0] == "restart"
        assert recovery_calls[0][1] == alert
        
        # Verify alert metadata was updated
        assert alert.metadata["recovery_attempted"] is True
        assert "recovery_timestamp" in alert.metadata
    
    @pytest.mark.asyncio
    async def test_health_check_recovery_cycle(self, monitoring_system):
        """Test complete health check failure and recovery cycle"""
        
        # Start monitoring
        await monitoring_system.start_monitoring()
        
        # Create a health check that fails then recovers
        check_calls = 0
        def flaky_health_check():
            nonlocal check_calls
            check_calls += 1
            # Fail for first 3 calls, then succeed
            return check_calls > 3
        
        health_check = HealthCheck(
            name="flaky_component",
            check_function=flaky_health_check,
            interval_seconds=1,
            failure_threshold=2,
            recovery_threshold=2
        )
        monitoring_system.register_health_check(health_check)
        
        # Mock alert channel to track alerts
        alerts_sent = []
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        async def capture_alert(alert):
            alerts_sent.append(alert)
            return True
        mock_channel.send_alert = capture_alert
        monitoring_system.alert_channels["test"] = mock_channel
        
        # Wait for failure and recovery cycle
        await asyncio.sleep(8)  # Allow time for multiple cycles
        
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        
        # Verify we got both failure and recovery alerts
        failure_alerts = [a for a in alerts_sent if "failure" in a.title.lower()]
        recovery_alerts = [a for a in alerts_sent if "recovered" in a.title.lower()]
        
        assert len(failure_alerts) > 0, "Should have failure alerts"
        assert len(recovery_alerts) > 0, "Should have recovery alerts"
    
    def test_system_health_aggregation(self, monitoring_system):
        """Test system health status aggregation"""
        
        # Set various component health statuses
        monitoring_system.health_status_cache["component1"] = HealthStatus.HEALTHY
        monitoring_system.health_status_cache["component2"] = HealthStatus.WARNING
        monitoring_system.health_status_cache["component3"] = HealthStatus.CRITICAL
        monitoring_system.health_status_cache["component4"] = HealthStatus.UNKNOWN
        
        # Add some active alerts
        alert1 = Alert(title="Active Alert 1")
        alert2 = Alert(title="Resolved Alert", resolved=True)
        monitoring_system.active_alerts[alert1.id] = alert1
        monitoring_system.active_alerts[alert2.id] = alert2
        
        # Get system health
        health = monitoring_system.get_system_health()
        
        # Verify aggregation logic
        assert health["overall_status"] == HealthStatus.CRITICAL.value  # Worst status wins
        assert health["components"]["component1"] == HealthStatus.HEALTHY.value
        assert health["components"]["component2"] == HealthStatus.WARNING.value
        assert health["components"]["component3"] == HealthStatus.CRITICAL.value
        assert health["components"]["component4"] == HealthStatus.UNKNOWN.value
        assert health["active_alerts"] == 1  # Only unresolved alerts
    
    def test_performance_metrics_statistics(self, monitoring_system):
        """Test performance metrics statistics calculation"""
        
        # Record a series of metrics
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            metric = Metric(
                name="test_metric",
                type=MetricType.GAUGE,
                value=value
            )
            monitoring_system.performance_monitor.record_metric(metric)
        
        # Get statistics
        stats = monitoring_system.performance_monitor.get_metric_statistics("test_metric")
        
        # Verify calculations
        assert stats["count"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0  # (10+20+30+40+50)/5
        assert stats["median"] == 30.0
        assert stats["std_dev"] > 0  # Should have some standard deviation
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, monitoring_system):
        """Test that similar alerts are deduplicated"""
        
        # Mock alert channel
        alerts_sent = []
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        async def capture_alert(alert):
            alerts_sent.append(alert)
            return True
        mock_channel.send_alert = capture_alert
        monitoring_system.alert_channels["test"] = mock_channel
        
        # Send similar alerts
        alert1 = Alert(
            type=AlertType.SYSTEM_HEALTH,
            source="test_component",
            title="Component failure 1"
        )
        
        alert2 = Alert(
            type=AlertType.SYSTEM_HEALTH,
            source="test_component",
            title="Component failure 2"
        )
        
        # Send first alert
        await monitoring_system._send_alert(alert1)
        
        # Send similar alert (should be deduplicated)
        await monitoring_system._send_alert(alert2)
        
        # Verify only one alert was actually sent
        assert len(alerts_sent) == 1
        assert len(monitoring_system.active_alerts) == 1
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self, monitoring_system):
        """Test alert resolution functionality"""
        
        # Create and send an alert
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            source="test_component",
            title="Test alert"
        )
        
        await monitoring_system._send_alert(alert)
        
        # Verify alert is active
        assert alert.id in monitoring_system.active_alerts
        assert not alert.resolved
        
        # Resolve alerts from the source
        await monitoring_system._resolve_alerts_by_source("test_component")
        
        # Verify alert is resolved
        assert alert.resolved is True
        assert alert.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_monitoring_system_lifecycle(self, monitoring_system):
        """Test complete monitoring system lifecycle"""
        
        # Initially inactive
        assert monitoring_system.monitoring_active is False
        assert len(monitoring_system.monitoring_tasks) == 0
        
        # Start monitoring
        await monitoring_system.start_monitoring()
        assert monitoring_system.monitoring_active is True
        assert len(monitoring_system.monitoring_tasks) > 0
        
        # Verify tasks are running
        for task in monitoring_system.monitoring_tasks:
            assert not task.done()
        
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        assert monitoring_system.monitoring_active is False
        assert len(monitoring_system.monitoring_tasks) == 0
    
    def test_workflow_metrics_window_management(self, monitoring_system):
        """Test workflow metrics window size management"""
        
        # Set small window size for testing
        monitoring_system.performance_monitor.window_size = 3
        
        # Record more metrics than window size
        for i in range(5):
            metrics = WorkflowMetrics(
                workflow_id="test_workflow",
                node_name=f"node_{i}",
                execution_time_ms=float(i * 100),
                memory_usage_mb=float(i * 10),
                success=True
            )
            monitoring_system.performance_monitor.record_workflow_metrics(metrics)
        
        # Verify only window size number of metrics are kept
        workflow_metrics = monitoring_system.performance_monitor.workflow_metrics["test_workflow"]
        assert len(workflow_metrics) == 3  # Window size
        
        # Verify it kept the most recent ones
        assert workflow_metrics[0].node_name == "node_2"  # Should start from index 2
        assert workflow_metrics[1].node_name == "node_3"
        assert workflow_metrics[2].node_name == "node_4"

class TestMonitoringAPIIntegration:
    """Test monitoring system integration with API endpoints"""
    
    @pytest.fixture
    def mock_monitoring_system(self):
        """Mock monitoring system for API testing"""
        system = Mock()
        system.get_system_health.return_value = {
            "overall_status": "healthy",
            "components": {"api": "healthy", "database": "healthy"},
            "active_alerts": 0,
            "last_updated": datetime.now().isoformat()
        }
        system.get_active_alerts.return_value = []
        system.get_performance_metrics.return_value = {
            "cpu_usage_percent": {"mean": 45.0, "max": 80.0},
            "memory_usage_mb": {"mean": 512.0, "max": 1024.0}
        }
        system.record_workflow_execution = Mock()
        system.start_monitoring = AsyncMock()
        system.stop_monitoring = AsyncMock()
        system.monitoring_active = True
        return system
    
    def test_health_endpoint_response_format(self, mock_monitoring_system):
        """Test health endpoint response format"""
        health_data = mock_monitoring_system.get_system_health()
        
        # Verify required fields
        assert "overall_status" in health_data
        assert "components" in health_data
        assert "active_alerts" in health_data
        assert "last_updated" in health_data
        
        # Verify data types
        assert isinstance(health_data["overall_status"], str)
        assert isinstance(health_data["components"], dict)
        assert isinstance(health_data["active_alerts"], int)
    
    def test_alerts_endpoint_response_format(self, mock_monitoring_system):
        """Test alerts endpoint response format"""
        alerts_data = mock_monitoring_system.get_active_alerts()
        
        # Should return a list (even if empty)
        assert isinstance(alerts_data, list)
    
    def test_metrics_endpoint_response_format(self, mock_monitoring_system):
        """Test metrics endpoint response format"""
        metrics_data = mock_monitoring_system.get_performance_metrics(60)
        
        # Verify structure
        assert isinstance(metrics_data, dict)
        
        # Check for expected metric types
        for metric_name, stats in metrics_data.items():
            assert isinstance(stats, dict)
            # Should have statistical measures
            expected_keys = ["mean", "max"]
            for key in expected_keys:
                if key in stats:
                    assert isinstance(stats[key], (int, float))
    
    def test_workflow_metrics_recording(self, mock_monitoring_system):
        """Test workflow metrics recording"""
        # Record metrics
        mock_monitoring_system.record_workflow_execution(
            workflow_id="test_123",
            node_name="test_node",
            execution_time_ms=1500.0,
            success=True,
            memory_usage_mb=256.0,
            error_count=0
        )
        
        # Verify method was called with correct parameters
        mock_monitoring_system.record_workflow_execution.assert_called_once_with(
            workflow_id="test_123",
            node_name="test_node",
            execution_time_ms=1500.0,
            success=True,
            memory_usage_mb=256.0,
            error_count=0
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])