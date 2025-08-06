"""
Comprehensive tests for monitoring and alerting systems.
Tests task 13.2: Build monitoring and alerting systems.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from services.monitoring_alerting import (
    MonitoringAlertingSystem, HealthCheck, HealthCheckResult, HealthStatus,
    Alert, AlertSeverity, AlertType, Metric, MetricType, WorkflowMetrics,
    LogAlertChannel, EmailAlertChannel, WebhookAlertChannel,
    PerformanceMonitor
)
from services.error_handling import ErrorCategory, ErrorSeverity as ErrorSev

class TestHealthCheck:
    """Test health check functionality"""
    
    def test_health_check_creation(self):
        """Test health check creation"""
        check_func = Mock(return_value=True)
        health_check = HealthCheck(
            name="test_check",
            check_function=check_func,
            interval_seconds=30,
            timeout_seconds=10
        )
        
        assert health_check.name == "test_check"
        assert health_check.check_function == check_func
        assert health_check.interval_seconds == 30
        assert health_check.timeout_seconds == 10
        assert health_check.enabled is True
    
    def test_health_check_result_creation(self):
        """Test health check result creation"""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            duration_ms=150.5
        )
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.duration_ms == 150.5
        assert isinstance(result.timestamp, datetime)

class TestAlert:
    """Test alert functionality"""
    
    def test_alert_creation(self):
        """Test alert creation"""
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="test_system"
        )
        
        assert alert.type == AlertType.SYSTEM_HEALTH
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.source == "test_system"
        assert alert.resolved is False
        assert alert.resolved_at is None
        assert isinstance(alert.timestamp, datetime)
        assert len(alert.id) > 0
    
    def test_alert_resolution(self):
        """Test alert resolution"""
        alert = Alert(
            title="Test Alert",
            message="Test message"
        )
        
        assert alert.resolved is False
        assert alert.resolved_at is None
        
        # Simulate resolution
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        assert alert.resolved is True
        assert alert.resolved_at is not None

class TestMetric:
    """Test metric functionality"""
    
    def test_metric_creation(self):
        """Test metric creation"""
        metric = Metric(
            name="cpu_usage",
            type=MetricType.GAUGE,
            value=75.5,
            tags={"host": "server1", "region": "us-east"}
        )
        
        assert metric.name == "cpu_usage"
        assert metric.type == MetricType.GAUGE
        assert metric.value == 75.5
        assert metric.tags["host"] == "server1"
        assert metric.tags["region"] == "us-east"
        assert isinstance(metric.timestamp, datetime)
    
    def test_workflow_metrics_creation(self):
        """Test workflow metrics creation"""
        metrics = WorkflowMetrics(
            workflow_id="wf_123",
            node_name="draft_generator",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            success=True,
            error_count=0
        )
        
        assert metrics.workflow_id == "wf_123"
        assert metrics.node_name == "draft_generator"
        assert metrics.execution_time_ms == 1500.0
        assert metrics.memory_usage_mb == 256.0
        assert metrics.success is True
        assert metrics.error_count == 0

class TestAlertChannels:
    """Test alert channel implementations"""
    
    @pytest.mark.asyncio
    async def test_log_alert_channel(self):
        """Test log alert channel"""
        channel = LogAlertChannel("log", {"enabled": True})
        
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message"
        )
        
        with patch('services.monitoring_alerting.logger') as mock_logger:
            result = await channel.send_alert(alert)
            
            assert result is True
            mock_logger.log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_email_alert_channel_no_config(self):
        """Test email alert channel without SMTP config"""
        channel = EmailAlertChannel("email", {"enabled": True})
        
        alert = Alert(title="Test Alert", message="Test message")
        
        with patch('services.monitoring_alerting.logger') as mock_logger:
            result = await channel.send_alert(alert)
            
            assert result is False
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_email_alert_channel_with_config(self):
        """Test email alert channel with SMTP config"""
        config = {
            "enabled": True,
            "smtp": {
                "from_email": "alerts@test.com"
            },
            "recipients": ["admin@test.com"]
        }
        channel = EmailAlertChannel("email", config)
        
        alert = Alert(title="Test Alert", message="Test message")
        
        with patch('services.monitoring_alerting.logger') as mock_logger:
            result = await channel.send_alert(alert)
            
            assert result is True
            mock_logger.info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_webhook_alert_channel_no_url(self):
        """Test webhook alert channel without URL"""
        channel = WebhookAlertChannel("webhook", {"enabled": True})
        
        alert = Alert(title="Test Alert", message="Test message")
        
        with patch('services.monitoring_alerting.logger') as mock_logger:
            result = await channel.send_alert(alert)
            
            assert result is False
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_webhook_alert_channel_success(self):
        """Test successful webhook alert"""
        config = {
            "enabled": True,
            "url": "https://webhook.test.com/alerts"
        }
        channel = WebhookAlertChannel("webhook", config)
        
        alert = Alert(title="Test Alert", message="Test message")
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await channel.send_alert(alert)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_webhook_alert_channel_failure(self):
        """Test failed webhook alert"""
        config = {
            "enabled": True,
            "url": "https://webhook.test.com/alerts"
        }
        channel = WebhookAlertChannel("webhook", config)
        
        alert = Alert(title="Test Alert", message="Test message")
        
        mock_response = AsyncMock()
        mock_response.status = 500
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await channel.send_alert(alert)
            
            assert result is False

class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def test_performance_monitor_creation(self):
        """Test performance monitor creation"""
        monitor = PerformanceMonitor(window_size=50)
        
        assert monitor.window_size == 50
        assert len(monitor.metrics) == 0
        assert len(monitor.workflow_metrics) == 0
    
    def test_record_metric(self):
        """Test recording metrics"""
        monitor = PerformanceMonitor()
        
        metric = Metric(
            name="cpu_usage",
            type=MetricType.GAUGE,
            value=75.0
        )
        
        monitor.record_metric(metric)
        
        assert "cpu_usage" in monitor.metrics
        assert len(monitor.metrics["cpu_usage"]) == 1
        assert monitor.metrics["cpu_usage"][0]["value"] == 75.0
    
    def test_record_workflow_metrics(self):
        """Test recording workflow metrics"""
        monitor = PerformanceMonitor()
        
        metrics = WorkflowMetrics(
            workflow_id="wf_123",
            node_name="test_node",
            execution_time_ms=1000.0,
            memory_usage_mb=100.0,
            success=True
        )
        
        monitor.record_workflow_metrics(metrics)
        
        assert "wf_123" in monitor.workflow_metrics
        assert len(monitor.workflow_metrics["wf_123"]) == 1
        assert monitor.workflow_metrics["wf_123"][0].node_name == "test_node"
    
    def test_get_metric_statistics(self):
        """Test getting metric statistics"""
        monitor = PerformanceMonitor()
        
        # Record multiple metrics
        for i in range(10):
            metric = Metric(
                name="test_metric",
                type=MetricType.GAUGE,
                value=float(i * 10)
            )
            monitor.record_metric(metric)
        
        stats = monitor.get_metric_statistics("test_metric")
        
        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 90.0
        assert stats["mean"] == 45.0
        assert stats["median"] == 45.0
    
    def test_check_performance_thresholds(self):
        """Test performance threshold checking"""
        monitor = PerformanceMonitor()
        
        # Set a low threshold for testing
        monitor.thresholds["test_metric"] = 50.0
        
        # Record metrics above threshold
        for i in range(5):
            metric = Metric(
                name="test_metric",
                type=MetricType.GAUGE,
                value=60.0  # Above threshold
            )
            monitor.record_metric(metric)
        
        alerts = monitor.check_performance_thresholds()
        
        assert len(alerts) == 1
        assert alerts[0].type == AlertType.PERFORMANCE_DEGRADATION
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "test_metric" in alerts[0].title

class TestMonitoringAlertingSystem:
    """Test main monitoring and alerting system"""
    
    def test_system_creation(self):
        """Test monitoring system creation"""
        system = MonitoringAlertingSystem()
        
        assert system.monitoring_active is False
        assert len(system.health_checks) > 0  # Default health checks
        assert len(system.alert_channels) > 0  # Default alert channels
        assert len(system.recovery_actions) > 0  # Default recovery actions
    
    def test_register_health_check(self):
        """Test registering health checks"""
        system = MonitoringAlertingSystem()
        
        check_func = Mock(return_value=True)
        health_check = HealthCheck(
            name="custom_check",
            check_function=check_func
        )
        
        system.register_health_check(health_check)
        
        assert "custom_check" in system.health_checks
        assert system.health_checks["custom_check"] == health_check
    
    def test_register_alert_channel(self):
        """Test registering alert channels"""
        system = MonitoringAlertingSystem()
        
        channel = LogAlertChannel("custom_log", {"enabled": True})
        system.register_alert_channel("custom_log", channel)
        
        assert "custom_log" in system.alert_channels
        assert system.alert_channels["custom_log"] == channel
    
    def test_register_recovery_action(self):
        """Test registering recovery actions"""
        system = MonitoringAlertingSystem()
        
        recovery_func = Mock()
        system.register_recovery_action("custom_recovery", recovery_func)
        
        assert "custom_recovery" in system.recovery_actions
        assert system.recovery_actions["custom_recovery"] == recovery_func
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        system = MonitoringAlertingSystem()
        
        assert system.monitoring_active is False
        assert len(system.monitoring_tasks) == 0
        
        # Start monitoring
        await system.start_monitoring()
        
        assert system.monitoring_active is True
        assert len(system.monitoring_tasks) > 0
        
        # Stop monitoring
        await system.stop_monitoring()
        
        assert system.monitoring_active is False
        assert len(system.monitoring_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alerts"""
        system = MonitoringAlertingSystem()
        
        # Mock alert channels
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        mock_channel.send_alert = AsyncMock(return_value=True)
        system.alert_channels["mock"] = mock_channel
        
        alert = Alert(
            title="Test Alert",
            message="Test message"
        )
        
        await system._send_alert(alert)
        
        assert alert.id in system.active_alerts
        assert alert in system.alert_history
        mock_channel.send_alert.assert_called_once_with(alert)
    
    def test_find_similar_active_alert(self):
        """Test finding similar active alerts"""
        system = MonitoringAlertingSystem()
        
        # Add an active alert
        existing_alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            source="test_source",
            title="Existing Alert"
        )
        system.active_alerts[existing_alert.id] = existing_alert
        
        # Create similar alert
        similar_alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            source="test_source",
            title="Similar Alert"
        )
        
        found_alert = system._find_similar_active_alert(similar_alert)
        
        assert found_alert == existing_alert
    
    @pytest.mark.asyncio
    async def test_resolve_alerts_by_source(self):
        """Test resolving alerts by source"""
        system = MonitoringAlertingSystem()
        
        # Add alerts from same source
        alert1 = Alert(source="test_source", title="Alert 1")
        alert2 = Alert(source="test_source", title="Alert 2")
        alert3 = Alert(source="other_source", title="Alert 3")
        
        system.active_alerts[alert1.id] = alert1
        system.active_alerts[alert2.id] = alert2
        system.active_alerts[alert3.id] = alert3
        
        await system._resolve_alerts_by_source("test_source")
        
        assert alert1.resolved is True
        assert alert2.resolved is True
        assert alert3.resolved is False
    
    @pytest.mark.asyncio
    async def test_should_attempt_recovery(self):
        """Test recovery attempt decision logic"""
        system = MonitoringAlertingSystem()
        
        # Test recoverable alert type
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            timestamp=datetime.now() - timedelta(minutes=5)  # Old enough
        )
        
        should_recover = await system._should_attempt_recovery(alert)
        assert should_recover is True
        
        # Test non-recoverable alert type
        alert.type = AlertType.CUSTOM
        should_recover = await system._should_attempt_recovery(alert)
        assert should_recover is False
        
        # Test already attempted recovery
        alert.type = AlertType.SYSTEM_HEALTH
        alert.metadata["recovery_attempted"] = True
        should_recover = await system._should_attempt_recovery(alert)
        assert should_recover is False
        
        # Test too recent alert
        alert.metadata.pop("recovery_attempted", None)
        alert.timestamp = datetime.now()  # Too recent
        should_recover = await system._should_attempt_recovery(alert)
        assert should_recover is False
    
    @pytest.mark.asyncio
    async def test_attempt_recovery(self):
        """Test attempting recovery"""
        system = MonitoringAlertingSystem()
        
        # Mock recovery action
        mock_recovery = AsyncMock(return_value=True)
        system.recovery_actions["restart_component"] = mock_recovery
        
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            title="Test Alert"
        )
        
        await system._attempt_recovery(alert)
        
        assert alert.metadata["recovery_attempted"] is True
        assert "recovery_timestamp" in alert.metadata
        mock_recovery.assert_called_once_with(alert)
    
    def test_get_system_health(self):
        """Test getting system health status"""
        system = MonitoringAlertingSystem()
        
        # Set some health statuses
        system.health_status_cache["component1"] = HealthStatus.HEALTHY
        system.health_status_cache["component2"] = HealthStatus.WARNING
        system.health_status_cache["component3"] = HealthStatus.CRITICAL
        
        # Add some active alerts
        alert1 = Alert(title="Alert 1")
        alert2 = Alert(title="Alert 2", resolved=True)
        system.active_alerts[alert1.id] = alert1
        system.active_alerts[alert2.id] = alert2
        
        health = system.get_system_health()
        
        assert health["overall_status"] == HealthStatus.CRITICAL.value
        assert health["components"]["component1"] == HealthStatus.HEALTHY.value
        assert health["components"]["component2"] == HealthStatus.WARNING.value
        assert health["components"]["component3"] == HealthStatus.CRITICAL.value
        assert health["active_alerts"] == 1  # Only unresolved alerts
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        system = MonitoringAlertingSystem()
        
        # Add alerts
        alert1 = Alert(title="Active Alert")
        alert2 = Alert(title="Resolved Alert", resolved=True)
        system.active_alerts[alert1.id] = alert1
        system.active_alerts[alert2.id] = alert2
        
        active_alerts = system.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0]["title"] == "Active Alert"
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics"""
        system = MonitoringAlertingSystem()
        
        # Record some metrics
        for i in range(5):
            metric = Metric(
                name="cpu_usage_percent",
                type=MetricType.GAUGE,
                value=float(i * 10)
            )
            system.performance_monitor.record_metric(metric)
        
        metrics = system.get_performance_metrics()
        
        assert "cpu_usage_percent" in metrics
        assert metrics["cpu_usage_percent"]["count"] == 5
    
    def test_record_workflow_execution(self):
        """Test recording workflow execution"""
        system = MonitoringAlertingSystem()
        
        system.record_workflow_execution(
            workflow_id="wf_123",
            node_name="test_node",
            execution_time_ms=1500.0,
            success=True,
            memory_usage_mb=256.0,
            error_count=0
        )
        
        # Check workflow metrics
        assert "wf_123" in system.performance_monitor.workflow_metrics
        workflow_metrics = system.performance_monitor.workflow_metrics["wf_123"]
        assert len(workflow_metrics) == 1
        assert workflow_metrics[0].node_name == "test_node"
        
        # Check general metrics
        assert "response_time_ms" in system.performance_monitor.metrics
        response_metrics = system.performance_monitor.metrics["response_time_ms"]
        assert len(response_metrics) == 1
        assert response_metrics[0]["value"] == 1500.0

class TestHealthCheckImplementations:
    """Test default health check implementations"""
    
    def test_system_health_check(self):
        """Test system health check"""
        system = MonitoringAlertingSystem()
        
        result = system._check_system_health()
        
        # Should pass basic checks
        assert isinstance(result, bool)
    
    def test_api_health_check(self):
        """Test API health check"""
        system = MonitoringAlertingSystem()
        
        with patch('services.monitoring_alerting.router'):
            result = system._check_api_health()
            assert result is True
    
    def test_storage_health_check(self):
        """Test storage health check"""
        system = MonitoringAlertingSystem()
        
        result = system._check_storage_health()
        
        # Should be able to read/write to storage
        assert isinstance(result, bool)
    
    def test_external_services_health_check(self):
        """Test external services health check"""
        system = MonitoringAlertingSystem()
        
        result = system._check_external_services_health()
        
        # Currently just returns True
        assert result is True

class TestRecoveryActions:
    """Test recovery action implementations"""
    
    @pytest.mark.asyncio
    async def test_restart_component_recovery(self):
        """Test component restart recovery"""
        system = MonitoringAlertingSystem()
        
        alert = Alert(
            title="Component Failure",
            metadata={"health_check": "test_component"}
        )
        
        result = await system._restart_component_recovery(alert)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_clear_cache_recovery(self):
        """Test cache clear recovery"""
        system = MonitoringAlertingSystem()
        
        # Add some metrics to clear
        metric = Metric(name="test", type=MetricType.GAUGE, value=1.0)
        system.performance_monitor.record_metric(metric)
        
        alert = Alert(title="Performance Issue")
        
        result = await system._clear_cache_recovery(alert)
        
        assert result is True
        assert len(system.performance_monitor.metrics) == 0
    
    @pytest.mark.asyncio
    async def test_scale_resources_recovery(self):
        """Test resource scaling recovery"""
        system = MonitoringAlertingSystem()
        
        alert = Alert(title="Resource Exhaustion")
        
        result = await system._scale_resources_recovery(alert)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_enable_fallback_mode_recovery(self):
        """Test fallback mode recovery"""
        system = MonitoringAlertingSystem()
        
        alert = Alert(
            title="Component Failure",
            metadata={"component": "test_component"}
        )
        
        result = await system._enable_fallback_mode_recovery(alert)
        
        assert result is True
        
        # Check that component health was updated
        component_health = system.error_framework.degradation_manager.component_health.get("test_component")
        assert component_health is not None
        assert component_health["healthy"] is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        system = MonitoringAlertingSystem()
        
        alert = Alert(title="API Failure")
        
        result = await system._circuit_breaker_recovery(alert)
        
        assert result is True

class TestIntegration:
    """Integration tests for monitoring and alerting"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow"""
        system = MonitoringAlertingSystem()
        
        # Register a failing health check
        failing_check = Mock(return_value=False)
        health_check = HealthCheck(
            name="failing_check",
            check_function=failing_check,
            interval_seconds=1,  # Fast for testing
            failure_threshold=2
        )
        system.register_health_check(health_check)
        
        # Mock alert channel
        mock_channel = AsyncMock()
        mock_channel.enabled = True
        mock_channel.send_alert = AsyncMock(return_value=True)
        system.alert_channels["mock"] = mock_channel
        
        # Start monitoring briefly
        await system.start_monitoring()
        
        # Wait for health checks to run
        await asyncio.sleep(3)
        
        # Stop monitoring
        await system.stop_monitoring()
        
        # Should have generated alerts
        assert len(system.active_alerts) > 0
        
        # Mock channel should have been called
        assert mock_channel.send_alert.call_count > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        system = MonitoringAlertingSystem()
        
        # Set low threshold for testing
        system.performance_monitor.thresholds["test_metric"] = 10.0
        
        # Record metrics above threshold
        for i in range(5):
            metric = Metric(
                name="test_metric",
                type=MetricType.GAUGE,
                value=20.0  # Above threshold
            )
            system.performance_monitor.record_metric(metric)
        
        # Check thresholds
        alerts = system.performance_monitor.check_performance_thresholds()
        
        assert len(alerts) > 0
        assert alerts[0].type == AlertType.PERFORMANCE_DEGRADATION
    
    @pytest.mark.asyncio
    async def test_recovery_integration(self):
        """Test recovery integration"""
        system = MonitoringAlertingSystem()
        
        # Create alert eligible for recovery
        alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            title="System Failure",
            timestamp=datetime.now() - timedelta(minutes=5)
        )
        system.active_alerts[alert.id] = alert
        
        # Mock recovery action
        mock_recovery = AsyncMock(return_value=True)
        system.recovery_actions["restart_component"] = mock_recovery
        
        # Process auto-recovery
        await system._process_auto_recovery()
        
        # Recovery should have been attempted
        assert alert.metadata.get("recovery_attempted") is True
        mock_recovery.assert_called_once_with(alert)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])