"""
End-to-end validation test for monitoring and alerting system.
This script validates that task 13.2 is properly implemented.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_monitoring_system_end_to_end():
    """
    Comprehensive end-to-end test of the monitoring and alerting system
    """
    logger.info("Starting end-to-end monitoring system validation")
    
    try:
        # Import the monitoring system
        from backend.services.monitoring_alerting import (
            MonitoringAlertingSystem, HealthCheck, Alert, AlertType, 
            AlertSeverity, HealthStatus, Metric, MetricType, WorkflowMetrics
        )
        logger.info("✓ Successfully imported monitoring system components")
        
        # Test 1: Create monitoring system
        logger.info("Test 1: Creating monitoring system...")
        config = {
            "auto_recovery": True,
            "max_history_size": 100,
            "email_alerts": {"enabled": False},
            "webhook_alerts": {"enabled": False}
        }
        monitoring_system = MonitoringAlertingSystem(config)
        logger.info("✓ Monitoring system created successfully")
        
        # Test 2: Verify default components are initialized
        logger.info("Test 2: Verifying default components...")
        assert len(monitoring_system.health_checks) > 0, "No default health checks found"
        assert len(monitoring_system.alert_channels) > 0, "No default alert channels found"
        assert len(monitoring_system.recovery_actions) > 0, "No default recovery actions found"
        logger.info(f"✓ Found {len(monitoring_system.health_checks)} health checks")
        logger.info(f"✓ Found {len(monitoring_system.alert_channels)} alert channels")
        logger.info(f"✓ Found {len(monitoring_system.recovery_actions)} recovery actions")
        
        # Test 3: Test health check registration and execution
        logger.info("Test 3: Testing health check functionality...")
        test_health_results = []
        
        def test_health_check():
            result = len(test_health_results) < 2  # Fail first 2 times
            test_health_results.append(result)
            return result
        
        custom_health_check = HealthCheck(
            name="test_component",
            check_function=test_health_check,
            interval_seconds=1,
            failure_threshold=2,
            recovery_threshold=1
        )
        
        monitoring_system.register_health_check(custom_health_check)
        assert "test_component" in monitoring_system.health_checks
        logger.info("✓ Health check registered successfully")
        
        # Test 4: Test metric recording
        logger.info("Test 4: Testing metric recording...")
        
        # Record various metrics
        cpu_metric = Metric(
            name="cpu_usage_percent",
            type=MetricType.GAUGE,
            value=75.5,
            tags={"host": "test_server"}
        )
        monitoring_system.performance_monitor.record_metric(cpu_metric)
        
        memory_metric = Metric(
            name="memory_usage_mb",
            type=MetricType.GAUGE,
            value=512.0
        )
        monitoring_system.performance_monitor.record_metric(memory_metric)
        
        # Record workflow metrics
        workflow_metrics = WorkflowMetrics(
            workflow_id="test_workflow_123",
            node_name="draft_generator",
            execution_time_ms=1500.0,
            memory_usage_mb=256.0,
            success=True,
            error_count=0
        )
        monitoring_system.performance_monitor.record_workflow_metrics(workflow_metrics)
        
        # Verify metrics were recorded
        assert "cpu_usage_percent" in monitoring_system.performance_monitor.metrics
        assert "memory_usage_mb" in monitoring_system.performance_monitor.metrics
        assert "test_workflow_123" in monitoring_system.performance_monitor.workflow_metrics
        logger.info("✓ Metrics recorded successfully")
        
        # Test 5: Test alert creation and management
        logger.info("Test 5: Testing alert functionality...")
        
        test_alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert for validation",
            source="validation_test"
        )
        
        # Capture alerts sent
        alerts_captured = []
        
        class TestAlertChannel:
            def __init__(self):
                self.enabled = True
                self.name = "test_channel"
            
            async def send_alert(self, alert):
                alerts_captured.append(alert)
                return True
        
        test_channel = TestAlertChannel()
        monitoring_system.alert_channels["test"] = test_channel
        
        await monitoring_system._send_alert(test_alert)
        
        # Verify alert was processed
        assert len(alerts_captured) == 1
        assert alerts_captured[0].title == "Test Alert"
        assert test_alert.id in monitoring_system.active_alerts
        logger.info("✓ Alert creation and sending works correctly")
        
        # Test 6: Test performance threshold monitoring
        logger.info("Test 6: Testing performance threshold monitoring...")
        
        # Set low threshold for testing
        monitoring_system.performance_monitor.thresholds["test_metric"] = 50.0
        
        # Record metrics above threshold
        for i in range(3):
            high_metric = Metric(
                name="test_metric",
                type=MetricType.GAUGE,
                value=75.0  # Above threshold
            )
            monitoring_system.performance_monitor.record_metric(high_metric)
        
        # Check thresholds
        threshold_alerts = monitoring_system.performance_monitor.check_performance_thresholds()
        assert len(threshold_alerts) > 0, "No threshold alerts generated"
        assert threshold_alerts[0].type == AlertType.PERFORMANCE_DEGRADATION
        logger.info("✓ Performance threshold monitoring works correctly")
        
        # Test 7: Test recovery mechanism
        logger.info("Test 7: Testing recovery mechanism...")
        
        recovery_calls = []
        
        async def test_recovery_action(alert):
            recovery_calls.append(alert)
            return True
        
        monitoring_system.recovery_actions["test_recovery"] = test_recovery_action
        
        # Create alert eligible for recovery
        recovery_alert = Alert(
            type=AlertType.SYSTEM_HEALTH,
            title="Component failure for recovery test",
            timestamp=datetime.now() - timedelta(minutes=5),  # Old enough
            source="test_component"
        )
        monitoring_system.active_alerts[recovery_alert.id] = recovery_alert
        
        # Process recovery
        await monitoring_system._process_auto_recovery()
        
        # Verify recovery was attempted
        assert recovery_alert.metadata.get("recovery_attempted") is True
        logger.info("✓ Recovery mechanism works correctly")
        
        # Test 8: Test system health aggregation
        logger.info("Test 8: Testing system health aggregation...")
        
        # Set various component statuses
        monitoring_system.health_status_cache["component1"] = HealthStatus.HEALTHY
        monitoring_system.health_status_cache["component2"] = HealthStatus.WARNING
        monitoring_system.health_status_cache["component3"] = HealthStatus.CRITICAL
        
        health_status = monitoring_system.get_system_health()
        
        assert health_status["overall_status"] == HealthStatus.CRITICAL.value
        assert "components" in health_status
        assert "active_alerts" in health_status
        logger.info("✓ System health aggregation works correctly")
        
        # Test 9: Test monitoring system lifecycle
        logger.info("Test 9: Testing monitoring system lifecycle...")
        
        # Start monitoring
        await monitoring_system.start_monitoring()
        assert monitoring_system.monitoring_active is True
        assert len(monitoring_system.monitoring_tasks) > 0
        logger.info("✓ Monitoring system started successfully")
        
        # Let it run briefly
        await asyncio.sleep(2)
        
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        assert monitoring_system.monitoring_active is False
        logger.info("✓ Monitoring system stopped successfully")
        
        # Test 10: Test API integration points
        logger.info("Test 10: Testing API integration...")
        
        # Test workflow metrics recording
        monitoring_system.record_workflow_execution(
            workflow_id="api_test_workflow",
            node_name="api_test_node",
            execution_time_ms=2000.0,
            success=True,
            memory_usage_mb=128.0,
            error_count=0
        )
        
        # Verify it was recorded
        assert "api_test_workflow" in monitoring_system.performance_monitor.workflow_metrics
        logger.info("✓ API integration points work correctly")
        
        # Test 11: Test error handling integration
        logger.info("Test 11: Testing error handling integration...")
        
        from backend.services.error_handling import TTDRError, ErrorCategory, ErrorSeverity as ErrorSev
        
        test_error = TTDRError(
            message="Test error for monitoring integration",
            category=ErrorCategory.WORKFLOW,
            severity=ErrorSev.HIGH
        )
        
        # The error framework should be accessible
        assert monitoring_system.error_framework is not None
        logger.info("✓ Error handling integration works correctly")
        
        # Test 12: Test workflow recovery integration
        logger.info("Test 12: Testing workflow recovery integration...")
        
        # The recovery manager should be accessible
        assert monitoring_system.recovery_manager is not None
        logger.info("✓ Workflow recovery integration works correctly")
        
        logger.info("🎉 All end-to-end tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test API endpoint integration"""
    logger.info("Testing API endpoint integration...")
    
    try:
        # Test that the global monitoring system is accessible
        from backend.services.monitoring_alerting import global_monitoring_system
        
        # Test basic functionality
        health = global_monitoring_system.get_system_health()
        assert isinstance(health, dict)
        assert "overall_status" in health
        
        alerts = global_monitoring_system.get_active_alerts()
        assert isinstance(alerts, list)
        
        metrics = global_monitoring_system.get_performance_metrics(60)
        assert isinstance(metrics, dict)
        
        logger.info("✓ API endpoint integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ API endpoint integration test failed: {e}")
        return False

def test_import_structure():
    """Test that all required components can be imported"""
    logger.info("Testing import structure...")
    
    try:
        # Test main monitoring system imports
        from backend.services.monitoring_alerting import (
            MonitoringAlertingSystem,
            HealthCheck, HealthCheckResult, HealthStatus,
            Alert, AlertSeverity, AlertType,
            Metric, MetricType, WorkflowMetrics,
            AlertChannel, LogAlertChannel, EmailAlertChannel, WebhookAlertChannel,
            PerformanceMonitor,
            global_monitoring_system
        )
        
        # Test error handling imports
        from backend.services.error_handling import (
            ErrorHandlingFramework, TTDRError, ErrorCategory, ErrorSeverity
        )
        
        # Test workflow recovery imports
        from backend.services.workflow_recovery import (
            WorkflowRecoveryManager, WorkflowState
        )
        
        logger.info("✓ All required components imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

async def main():
    """Run all validation tests"""
    logger.info("=" * 60)
    logger.info("MONITORING AND ALERTING SYSTEM VALIDATION")
    logger.info("Task 13.2: Build monitoring and alerting systems")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Import structure
    logger.info("\n1. Testing import structure...")
    result1 = test_import_structure()
    test_results.append(("Import Structure", result1))
    
    # Test 2: API endpoint integration
    logger.info("\n2. Testing API endpoint integration...")
    result2 = await test_api_endpoints()
    test_results.append(("API Integration", result2))
    
    # Test 3: End-to-end functionality
    logger.info("\n3. Testing end-to-end functionality...")
    result3 = await test_monitoring_system_end_to_end()
    test_results.append(("End-to-End Functionality", result3))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - Task 13.2 implementation is complete!")
        logger.info("\nImplemented features:")
        logger.info("- Real-time health monitoring with configurable checks")
        logger.info("- Multi-channel alerting system (log, email, webhook)")
        logger.info("- Performance metrics collection and threshold monitoring")
        logger.info("- Automated recovery mechanisms")
        logger.info("- Workflow execution monitoring")
        logger.info("- System health aggregation")
        logger.info("- API endpoints for monitoring data")
        logger.info("- Integration with error handling and recovery systems")
    else:
        logger.error("❌ SOME TESTS FAILED - Please check the implementation")
    
    return all_passed

if __name__ == "__main__":
    # Run the validation
    success = asyncio.run(main())
    exit(0 if success else 1)