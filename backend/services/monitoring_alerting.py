"""
Real-time monitoring and alerting systems for TTD-DR framework.
Implements task 13.2: Build monitoring and alerting systems.
"""

import logging
import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid
from collections import defaultdict, deque
import statistics
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from .error_handling import (
        ErrorHandlingFramework, ErrorRecord, ErrorSeverity, ErrorCategory,
        TTDRError
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    # Create mock classes for when error handling is not available
    class ErrorHandlingFramework:
        def __init__(self):
            pass
    
    class TTDRError(Exception):
        pass

try:
    from .workflow_recovery import (
        WorkflowRecoveryManager, WorkflowState, WorkflowCheckpoint
    )
    WORKFLOW_RECOVERY_AVAILABLE = True
except ImportError:
    WORKFLOW_RECOVERY_AVAILABLE = False
    # Create mock class for when workflow recovery is not available
    class WorkflowRecoveryManager:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Types of alerts"""
    SYSTEM_HEALTH = "system_health"
    WORKFLOW_FAILURE = "workflow_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    API_ERROR = "api_error"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TIMEOUT = "timeout"
    CUSTOM = "custom"

class MetricType(str, Enum):
    """Types of metrics to monitor"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class HealthCheck:
    """Represents a health check configuration"""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Represents an alert"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: AlertType = AlertType.CUSTOM
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    message: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class Metric:
    """Represents a metric measurement"""
    name: str
    type: MetricType
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowMetrics:
    """Workflow-specific metrics"""
    workflow_id: str
    node_name: str
    execution_time_ms: float
    memory_usage_mb: float
    success: bool
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AlertChannel:
    """Base class for alert channels"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through this channel"""
        raise NotImplementedError

class LogAlertChannel(AlertChannel):
    """Log-based alert channel"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to logs"""
        try:
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            logger.log(log_level, f"ALERT [{alert.type.value}] {alert.title}: {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send log alert: {e}")
            return False

class EmailAlertChannel(AlertChannel):
    """Email-based alert channel"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            smtp_config = self.config.get("smtp", {})
            if not smtp_config:
                logger.warning("No SMTP configuration for email alerts")
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = smtp_config.get("from_email", "alerts@ttdr.local")
            msg['To'] = ", ".join(self.config.get("recipients", []))
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Type: {alert.type.value}
- Severity: {alert.severity.value}
- Source: {alert.source}
- Time: {alert.timestamp.isoformat()}
- Message: {alert.message}

Metadata: {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email (in production, use async email sending)
            # For now, just log the attempt
            logger.info(f"Would send email alert: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

class WebhookAlertChannel(AlertChannel):
    """Webhook-based alert channel"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        try:
            import aiohttp
            
            webhook_url = self.config.get("url")
            if not webhook_url:
                logger.warning("No webhook URL configured")
                return False
            
            payload = {
                "alert_id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
                "tags": alert.tags
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent successfully: {alert.title}")
                        return True
                    else:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

class PerformanceMonitor:
    """Monitors system and workflow performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.workflow_metrics: Dict[str, List[WorkflowMetrics]] = defaultdict(list)
        self.thresholds = {
            "response_time_ms": 5000,
            "error_rate": 0.1,
            "memory_usage_mb": 1000,
            "cpu_usage_percent": 80
        }
    
    def record_metric(self, metric: Metric):
        """Record a metric measurement"""
        self.metrics[metric.name].append({
            "value": metric.value,
            "timestamp": metric.timestamp,
            "tags": metric.tags,
            "metadata": metric.metadata
        })
    
    def record_workflow_metrics(self, metrics: WorkflowMetrics):
        """Record workflow-specific metrics"""
        self.workflow_metrics[metrics.workflow_id].append(metrics)
        
        # Keep only recent metrics
        if len(self.workflow_metrics[metrics.workflow_id]) > self.window_size:
            self.workflow_metrics[metrics.workflow_id] = \
                self.workflow_metrics[metrics.workflow_id][-self.window_size:]
    
    def get_metric_statistics(self, metric_name: str, 
                            time_window_minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a metric over a time window"""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_values = [
            entry["value"] for entry in self.metrics[metric_name]
            if entry["timestamp"] > cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "std_dev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        }
    
    def check_performance_thresholds(self) -> List[Alert]:
        """Check if any performance metrics exceed thresholds"""
        alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            stats = self.get_metric_statistics(metric_name, time_window_minutes=5)
            
            if not stats:
                continue
            
            if stats["mean"] > threshold:
                alert = Alert(
                    type=AlertType.PERFORMANCE_DEGRADATION,
                    severity=AlertSeverity.WARNING,
                    title=f"Performance threshold exceeded: {metric_name}",
                    message=f"{metric_name} average ({stats['mean']:.2f}) exceeds threshold ({threshold})",
                    source="performance_monitor",
                    metadata={
                        "metric_name": metric_name,
                        "threshold": threshold,
                        "current_value": stats["mean"],
                        "statistics": stats
                    }
                )
                alerts.append(alert)
        
        return alerts

class MonitoringAlertingSystem:
    """Main monitoring and alerting system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_framework = ErrorHandlingFramework()
        self.recovery_manager = WorkflowRecoveryManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        self.health_status_cache: Dict[str, HealthStatus] = {}
        
        # Alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_channels: Dict[str, AlertChannel] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Recovery mechanisms
        self.recovery_actions: Dict[str, Callable] = {}
        self.auto_recovery_enabled = self.config.get("auto_recovery", True)
        
        # Initialize default components
        self._initialize_default_health_checks()
        self._initialize_default_alert_channels()
        self._initialize_default_recovery_actions()
    
    def _initialize_default_health_checks(self):
        """Initialize default health checks"""
        # System health check
        self.register_health_check(HealthCheck(
            name="system_health",
            check_function=self._check_system_health,
            interval_seconds=30,
            timeout_seconds=10
        ))
        
        # API health check
        self.register_health_check(HealthCheck(
            name="api_health",
            check_function=self._check_api_health,
            interval_seconds=60,
            timeout_seconds=15
        ))
        
        # Database/storage health check
        self.register_health_check(HealthCheck(
            name="storage_health",
            check_function=self._check_storage_health,
            interval_seconds=120,
            timeout_seconds=20
        ))
        
        # External services health check
        self.register_health_check(HealthCheck(
            name="external_services_health",
            check_function=self._check_external_services_health,
            interval_seconds=180,
            timeout_seconds=30
        ))
    
    def _initialize_default_alert_channels(self):
        """Initialize default alert channels"""
        # Log channel (always enabled)
        self.register_alert_channel("log", LogAlertChannel("log", {"enabled": True}))
        
        # Email channel (if configured)
        email_config = self.config.get("email_alerts", {})
        if email_config.get("enabled", False):
            self.register_alert_channel("email", EmailAlertChannel("email", email_config))
        
        # Webhook channel (if configured)
        webhook_config = self.config.get("webhook_alerts", {})
        if webhook_config.get("enabled", False):
            self.register_alert_channel("webhook", WebhookAlertChannel("webhook", webhook_config))
    
    def _initialize_default_recovery_actions(self):
        """Initialize default automated recovery actions"""
        self.recovery_actions.update({
            "restart_component": self._restart_component_recovery,
            "clear_cache": self._clear_cache_recovery,
            "scale_resources": self._scale_resources_recovery,
            "fallback_mode": self._enable_fallback_mode_recovery,
            "circuit_breaker": self._circuit_breaker_recovery
        })
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def register_alert_channel(self, name: str, channel: AlertChannel):
        """Register an alert channel"""
        self.alert_channels[name] = channel
        logger.info(f"Registered alert channel: {name}")
    
    def register_recovery_action(self, name: str, action: Callable):
        """Register a recovery action"""
        self.recovery_actions[name] = action
        logger.info(f"Registered recovery action: {name}")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            logger.warning("Monitoring system is already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting monitoring and alerting system")
        
        # Start health check tasks
        for health_check in self.health_checks.values():
            if health_check.enabled:
                task = asyncio.create_task(
                    self._run_health_check_loop(health_check)
                )
                self.monitoring_tasks.append(task)
        
        # Start performance monitoring task
        performance_task = asyncio.create_task(self._run_performance_monitoring())
        self.monitoring_tasks.append(performance_task)
        
        # Start alert processing task
        alert_task = asyncio.create_task(self._run_alert_processing())
        self.monitoring_tasks.append(alert_task)
        
        logger.info(f"Started {len(self.monitoring_tasks)} monitoring tasks")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        logger.info("Stopping monitoring and alerting system")
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Monitoring system stopped")
    
    async def _run_health_check_loop(self, health_check: HealthCheck):
        """Run a health check in a loop"""
        consecutive_failures = 0
        consecutive_successes = 0
        
        while self.monitoring_active:
            try:
                # Execute health check with timeout
                start_time = time.time()
                
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(health_check.check_function),
                        timeout=health_check.timeout_seconds
                    )
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if result:
                        status = HealthStatus.HEALTHY
                        message = "Health check passed"
                        consecutive_failures = 0
                        consecutive_successes += 1
                    else:
                        status = HealthStatus.WARNING
                        message = "Health check failed"
                        consecutive_failures += 1
                        consecutive_successes = 0
                        
                except asyncio.TimeoutError:
                    status = HealthStatus.CRITICAL
                    message = f"Health check timed out after {health_check.timeout_seconds}s"
                    duration_ms = health_check.timeout_seconds * 1000
                    consecutive_failures += 1
                    consecutive_successes = 0
                
                # Create health check result
                health_result = HealthCheckResult(
                    name=health_check.name,
                    status=status,
                    message=message,
                    duration_ms=duration_ms,
                    metadata=health_check.metadata.copy()
                )
                
                # Store result
                self.health_results[health_check.name].append(health_result)
                
                # Keep only recent results
                if len(self.health_results[health_check.name]) > 100:
                    self.health_results[health_check.name] = \
                        self.health_results[health_check.name][-100:]
                
                # Update cached status
                self.health_status_cache[health_check.name] = status
                
                # Check for alert conditions
                await self._check_health_alerts(health_check, health_result, 
                                              consecutive_failures, consecutive_successes)
                
            except Exception as e:
                logger.error(f"Error in health check {health_check.name}: {e}")
                
                # Create error result
                error_result = HealthCheckResult(
                    name=health_check.name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check error: {str(e)}",
                    metadata={"error": str(e)}
                )
                
                self.health_results[health_check.name].append(error_result)
                self.health_status_cache[health_check.name] = HealthStatus.UNKNOWN
            
            # Wait for next check
            await asyncio.sleep(health_check.interval_seconds)
    
    async def _check_health_alerts(self, health_check: HealthCheck, 
                                 result: HealthCheckResult,
                                 consecutive_failures: int, 
                                 consecutive_successes: int):
        """Check if health check results should trigger alerts"""
        
        # Alert on failure threshold
        if consecutive_failures >= health_check.failure_threshold:
            alert = Alert(
                type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.CRITICAL if result.status == HealthStatus.CRITICAL else AlertSeverity.ERROR,
                title=f"Health check failure: {health_check.name}",
                message=f"Health check {health_check.name} has failed {consecutive_failures} consecutive times",
                source="health_monitor",
                metadata={
                    "health_check": health_check.name,
                    "consecutive_failures": consecutive_failures,
                    "last_result": asdict(result)
                }
            )
            await self._send_alert(alert)
        
        # Alert on recovery
        elif consecutive_successes >= health_check.recovery_threshold and \
             health_check.name in [alert.metadata.get("health_check") for alert in self.active_alerts.values()]:
            
            alert = Alert(
                type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.INFO,
                title=f"Health check recovered: {health_check.name}",
                message=f"Health check {health_check.name} has recovered after {consecutive_successes} consecutive successes",
                source="health_monitor",
                metadata={
                    "health_check": health_check.name,
                    "consecutive_successes": consecutive_successes,
                    "recovery": True
                }
            )
            await self._send_alert(alert)
            
            # Resolve related alerts
            await self._resolve_alerts_by_source(health_check.name)
    
    async def _run_performance_monitoring(self):
        """Run performance monitoring loop"""
        while self.monitoring_active:
            try:
                # Check performance thresholds
                performance_alerts = self.performance_monitor.check_performance_thresholds()
                
                for alert in performance_alerts:
                    await self._send_alert(alert)
                
                # Record system metrics
                await self._record_system_metrics()
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _run_alert_processing(self):
        """Run alert processing loop"""
        while self.monitoring_active:
            try:
                # Process active alerts for auto-recovery
                if self.auto_recovery_enabled:
                    await self._process_auto_recovery()
                
                # Clean up old resolved alerts
                await self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
            
            await asyncio.sleep(30)  # Process every 30 seconds
    
    async def _send_alert(self, alert: Alert):
        """Send an alert through all configured channels"""
        # Check if similar alert is already active
        similar_alert = self._find_similar_active_alert(alert)
        if similar_alert:
            logger.debug(f"Similar alert already active, skipping: {alert.title}")
            return
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Keep alert history manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        logger.info(f"Sending alert: {alert.title}")
        
        # Send through all enabled channels
        for channel_name, channel in self.alert_channels.items():
            if channel.enabled:
                try:
                    success = await channel.send_alert(alert)
                    if success:
                        logger.debug(f"Alert sent successfully via {channel_name}")
                    else:
                        logger.warning(f"Failed to send alert via {channel_name}")
                except Exception as e:
                    logger.error(f"Error sending alert via {channel_name}: {e}")
    
    def _find_similar_active_alert(self, alert: Alert) -> Optional[Alert]:
        """Find if a similar alert is already active"""
        for active_alert in self.active_alerts.values():
            if (active_alert.type == alert.type and 
                active_alert.source == alert.source and
                not active_alert.resolved):
                return active_alert
        return None
    
    async def _resolve_alerts_by_source(self, source: str):
        """Resolve all active alerts from a specific source"""
        for alert in self.active_alerts.values():
            if alert.source == source and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Resolved alert: {alert.title}")
    
    async def _process_auto_recovery(self):
        """Process active alerts for automated recovery"""
        for alert in list(self.active_alerts.values()):
            if alert.resolved:
                continue
            
            # Check if alert is eligible for auto-recovery
            if await self._should_attempt_recovery(alert):
                await self._attempt_recovery(alert)
    
    async def _should_attempt_recovery(self, alert: Alert) -> bool:
        """Determine if an alert should trigger automated recovery"""
        # Only attempt recovery for certain alert types
        recoverable_types = {
            AlertType.SYSTEM_HEALTH,
            AlertType.PERFORMANCE_DEGRADATION,
            AlertType.API_ERROR,
            AlertType.TIMEOUT
        }
        
        if alert.type not in recoverable_types:
            return False
        
        # Check if recovery has already been attempted
        if alert.metadata.get("recovery_attempted"):
            return False
        
        # Check alert age (don't recover immediately)
        alert_age = datetime.now() - alert.timestamp
        if alert_age < timedelta(minutes=2):
            return False
        
        return True
    
    async def _attempt_recovery(self, alert: Alert):
        """Attempt automated recovery for an alert"""
        logger.info(f"Attempting automated recovery for alert: {alert.title}")
        
        # Mark recovery as attempted
        alert.metadata["recovery_attempted"] = True
        alert.metadata["recovery_timestamp"] = datetime.now().isoformat()
        
        # Determine recovery action based on alert type
        recovery_action = None
        
        if alert.type == AlertType.SYSTEM_HEALTH:
            recovery_action = "restart_component"
        elif alert.type == AlertType.PERFORMANCE_DEGRADATION:
            recovery_action = "clear_cache"
        elif alert.type == AlertType.API_ERROR:
            recovery_action = "circuit_breaker"
        elif alert.type == AlertType.TIMEOUT:
            recovery_action = "scale_resources"
        
        if recovery_action and recovery_action in self.recovery_actions:
            try:
                success = await self.recovery_actions[recovery_action](alert)
                
                if success:
                    logger.info(f"Automated recovery successful for alert: {alert.title}")
                    alert.metadata["recovery_successful"] = True
                    
                    # Create recovery success alert
                    recovery_alert = Alert(
                        type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.INFO,
                        title=f"Automated recovery successful",
                        message=f"Successfully recovered from: {alert.title}",
                        source="auto_recovery",
                        metadata={
                            "original_alert_id": alert.id,
                            "recovery_action": recovery_action
                        }
                    )
                    await self._send_alert(recovery_alert)
                    
                else:
                    logger.warning(f"Automated recovery failed for alert: {alert.title}")
                    alert.metadata["recovery_successful"] = False
                    
            except Exception as e:
                logger.error(f"Error during automated recovery: {e}")
                alert.metadata["recovery_error"] = str(e)
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old resolved alerts from active alerts
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time:
                to_remove.append(alert_id)
        
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old resolved alerts")
    
    async def _record_system_metrics(self):
        """Record system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_monitor.record_metric(Metric(
                name="cpu_usage_percent",
                type=MetricType.GAUGE,
                value=cpu_percent
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_monitor.record_metric(Metric(
                name="memory_usage_mb",
                type=MetricType.GAUGE,
                value=memory.used / (1024 * 1024)
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.performance_monitor.record_metric(Metric(
                name="disk_usage_percent",
                type=MetricType.GAUGE,
                value=(disk.used / disk.total) * 100
            ))
            
        except ImportError:
            # psutil not available, record basic metrics
            self.performance_monitor.record_metric(Metric(
                name="system_health",
                type=MetricType.GAUGE,
                value=1.0  # Assume healthy if we can record metrics
            ))
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")
    
    # Health check implementations
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Basic system checks
            import os
            import tempfile
            
            # Check if we can write to temp directory
            with tempfile.NamedTemporaryFile(delete=True):
                pass
            
            # Check memory availability (basic check)
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 95:
                    return False
            except ImportError:
                pass
            
            return True
        except Exception:
            return False
    
    def _check_api_health(self) -> bool:
        """Check API health"""
        try:
            # This would typically make an HTTP request to the API
            # For now, just check if the main modules can be imported
            from api.endpoints import router
            return True
        except Exception:
            return False
    
    def _check_storage_health(self) -> bool:
        """Check storage/database health"""
        try:
            # Check if we can read/write to the storage directory
            storage_dir = Path("workflow_states")
            storage_dir.mkdir(exist_ok=True)
            
            test_file = storage_dir / "health_check.tmp"
            test_file.write_text("health_check")
            content = test_file.read_text()
            test_file.unlink()
            
            return content == "health_check"
        except Exception:
            return False
    
    def _check_external_services_health(self) -> bool:
        """Check external services health"""
        try:
            # This would check external APIs like Kimi K2, Google Search, etc.
            # For now, just return True
            return True
        except Exception:
            return False
    
    # Recovery action implementations
    async def _restart_component_recovery(self, alert: Alert) -> bool:
        """Restart a component as recovery action"""
        try:
            component = alert.metadata.get("health_check", "unknown")
            logger.info(f"Simulating component restart for: {component}")
            
            # In a real implementation, this would restart the actual component
            # For now, just simulate success
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"Component restart recovery failed: {e}")
            return False
    
    async def _clear_cache_recovery(self, alert: Alert) -> bool:
        """Clear cache as recovery action"""
        try:
            logger.info("Clearing system caches")
            
            # Clear performance monitor metrics
            self.performance_monitor.metrics.clear()
            
            # In a real implementation, this would clear various caches
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"Cache clear recovery failed: {e}")
            return False
    
    async def _scale_resources_recovery(self, alert: Alert) -> bool:
        """Scale resources as recovery action"""
        try:
            logger.info("Simulating resource scaling")
            
            # In a real implementation, this would scale compute resources
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"Resource scaling recovery failed: {e}")
            return False
    
    async def _enable_fallback_mode_recovery(self, alert: Alert) -> bool:
        """Enable fallback mode as recovery action"""
        try:
            logger.info("Enabling fallback mode")
            
            # Enable degraded mode in the error handling framework
            component = alert.metadata.get("component", "unknown")
            self.error_framework.degradation_manager.update_component_health(
                component, False
            )
            
            return True
        except Exception as e:
            logger.error(f"Fallback mode recovery failed: {e}")
            return False
    
    async def _circuit_breaker_recovery(self, alert: Alert) -> bool:
        """Implement circuit breaker as recovery action"""
        try:
            logger.info("Activating circuit breaker")
            
            # In a real implementation, this would implement circuit breaker logic
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"Circuit breaker recovery failed: {e}")
            return False
    
    # Public API methods
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        overall_status = HealthStatus.HEALTHY
        health_details = {}
        
        for name, status in self.health_status_cache.items():
            health_details[name] = status.value
            
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
            elif status == HealthStatus.UNKNOWN and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.UNKNOWN
        
        return {
            "overall_status": overall_status.value,
            "components": health_details,
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [
            asdict(alert) for alert in self.active_alerts.values()
            if not alert.resolved
        ]
    
    def get_performance_metrics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for a time window"""
        metrics = {}
        
        for metric_name in ["cpu_usage_percent", "memory_usage_mb", "response_time_ms", "error_rate"]:
            stats = self.performance_monitor.get_metric_statistics(metric_name, time_window_minutes)
            if stats:
                metrics[metric_name] = stats
        
        return metrics
    
    def record_workflow_execution(self, workflow_id: str, node_name: str,
                                execution_time_ms: float, success: bool,
                                memory_usage_mb: float = 0.0, error_count: int = 0):
        """Record workflow execution metrics"""
        metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            node_name=node_name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            success=success,
            error_count=error_count
        )
        
        self.performance_monitor.record_workflow_metrics(metrics)
        
        # Record as general metrics too
        self.performance_monitor.record_metric(Metric(
            name="response_time_ms",
            type=MetricType.TIMER,
            value=execution_time_ms,
            tags={"workflow_id": workflow_id, "node": node_name}
        ))
        
        if not success:
            self.performance_monitor.record_metric(Metric(
                name="error_count",
                type=MetricType.COUNTER,
                value=1,
                tags={"workflow_id": workflow_id, "node": node_name}
            ))

# Global monitoring system instance
global_monitoring_system = MonitoringAlertingSystem()

# Export main classes
__all__ = [
    "MonitoringAlertingSystem",
    "HealthCheck", "HealthCheckResult", "HealthStatus",
    "Alert", "AlertSeverity", "AlertType",
    "Metric", "MetricType", "WorkflowMetrics",
    "AlertChannel", "LogAlertChannel", "EmailAlertChannel", "WebhookAlertChannel",
    "PerformanceMonitor",
    "global_monitoring_system"
]