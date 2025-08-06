"""
Workflow monitoring and debugging tools for TTD-DR framework.
Implements task 10.2: Build workflow monitoring and debugging tools.
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import traceback
import sys

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from models.core import TTDRState, QualityMetrics
from .graph import WorkflowNode, NodeStatus

logger = logging.getLogger(__name__)

@dataclass
class NodeExecutionTrace:
    """Detailed execution trace for a workflow node"""
    node_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: NodeStatus = NodeStatus.PENDING
    input_state_summary: Dict[str, Any] = None
    output_state_summary: Dict[str, Any] = None
    memory_usage_before: Optional[float] = None
    memory_usage_after: Optional[float] = None
    cpu_usage: Optional[float] = None
    error_details: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.input_state_summary is None:
            self.input_state_summary = {}
        if self.output_state_summary is None:
            self.output_state_summary = {}
        if self.warnings is None:
            self.warnings = []

@dataclass
class WorkflowExecutionTrace:
    """Complete execution trace for a workflow run"""
    execution_id: str
    topic: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    status: str = "running"
    node_traces: List[NodeExecutionTrace] = None
    iteration_count: int = 0
    final_quality_score: Optional[float] = None
    total_errors: int = 0
    total_warnings: int = 0
    memory_peak: Optional[float] = None
    execution_path: List[str] = None
    
    def __post_init__(self):
        if self.node_traces is None:
            self.node_traces = []
        if self.execution_path is None:
            self.execution_path = []

class PerformanceProfiler:
    """Performance profiling and optimization tools"""
    
    def __init__(self):
        self.node_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.cpu_snapshots: List[Dict[str, Any]] = []
        self.bottleneck_analysis: Dict[str, Any] = {}
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_system_resources(self, interval: float):
        """Monitor system resources in background thread"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - system resource monitoring disabled")
            return
            
        while self._monitoring_active:
            try:
                # Memory usage
                memory_info = psutil.virtual_memory()
                memory_snapshot = {
                    "timestamp": datetime.now(),
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "percent": memory_info.percent,
                    "used": memory_info.used
                }
                self.memory_snapshots.append(memory_snapshot)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_snapshot = {
                    "timestamp": datetime.now(),
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count()
                }
                self.cpu_snapshots.append(cpu_snapshot)
                
                # Keep only recent snapshots (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.memory_snapshots = [
                    s for s in self.memory_snapshots 
                    if s["timestamp"] > cutoff_time
                ]
                self.cpu_snapshots = [
                    s for s in self.cpu_snapshots 
                    if s["timestamp"] > cutoff_time
                ]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    def record_node_performance(self, node_name: str, duration: float, 
                              memory_usage: Optional[float] = None):
        """Record performance metrics for a node execution"""
        self.node_performance_history[node_name].append(duration)
        
        # Keep only recent performance data (last 100 executions)
        if len(self.node_performance_history[node_name]) > 100:
            self.node_performance_history[node_name] = \
                self.node_performance_history[node_name][-100:]
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks in the workflow"""
        analysis = {
            "timestamp": datetime.now(),
            "node_performance": {},
            "system_bottlenecks": {},
            "recommendations": []
        }
        
        # Analyze node performance
        for node_name, durations in self.node_performance_history.items():
            if not durations:
                continue
                
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
            
            analysis["node_performance"][node_name] = {
                "average_duration": avg_duration,
                "max_duration": max_duration,
                "min_duration": min_duration,
                "variance": variance,
                "execution_count": len(durations),
                "stability_score": 1.0 - (variance / (avg_duration ** 2)) if avg_duration > 0 else 0.0
            }
        
        # Identify slowest nodes
        if analysis["node_performance"]:
            slowest_node = max(
                analysis["node_performance"].items(),
                key=lambda x: x[1]["average_duration"]
            )
            analysis["slowest_node"] = {
                "name": slowest_node[0],
                "average_duration": slowest_node[1]["average_duration"]
            }
        
        # Analyze system resource usage
        if self.memory_snapshots:
            recent_memory = self.memory_snapshots[-10:]  # Last 10 snapshots
            avg_memory_usage = sum(s["percent"] for s in recent_memory) / len(recent_memory)
            max_memory_usage = max(s["percent"] for s in recent_memory)
            
            analysis["system_bottlenecks"]["memory"] = {
                "average_usage_percent": avg_memory_usage,
                "peak_usage_percent": max_memory_usage,
                "is_bottleneck": max_memory_usage > 85.0
            }
        
        if self.cpu_snapshots:
            recent_cpu = self.cpu_snapshots[-10:]  # Last 10 snapshots
            avg_cpu_usage = sum(s["cpu_percent"] for s in recent_cpu) / len(recent_cpu)
            max_cpu_usage = max(s["cpu_percent"] for s in recent_cpu)
            
            analysis["system_bottlenecks"]["cpu"] = {
                "average_usage_percent": avg_cpu_usage,
                "peak_usage_percent": max_cpu_usage,
                "is_bottleneck": max_cpu_usage > 90.0
            }
        
        # Generate recommendations
        recommendations = []
        
        # Node-specific recommendations
        for node_name, perf in analysis["node_performance"].items():
            if perf["average_duration"] > 30.0:  # Slow node (>30s)
                recommendations.append(
                    f"Node '{node_name}' is slow (avg: {perf['average_duration']:.1f}s) - consider optimization"
                )
            
            if perf["stability_score"] < 0.7:  # Unstable performance
                recommendations.append(
                    f"Node '{node_name}' has unstable performance - investigate variability"
                )
        
        # System resource recommendations
        memory_info = analysis["system_bottlenecks"].get("memory", {})
        if memory_info.get("is_bottleneck", False):
            recommendations.append(
                f"High memory usage detected ({memory_info['peak_usage_percent']:.1f}%) - consider memory optimization"
            )
        
        cpu_info = analysis["system_bottlenecks"].get("cpu", {})
        if cpu_info.get("is_bottleneck", False):
            recommendations.append(
                f"High CPU usage detected ({cpu_info['peak_usage_percent']:.1f}%) - consider parallel processing"
            )
        
        analysis["recommendations"] = recommendations
        self.bottleneck_analysis = analysis
        
        return analysis
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        summary = {
            "monitoring_active": self._monitoring_active,
            "nodes_monitored": len(self.node_performance_history),
            "total_executions": sum(len(durations) for durations in self.node_performance_history.values()),
            "memory_snapshots": len(self.memory_snapshots),
            "cpu_snapshots": len(self.cpu_snapshots)
        }
        
        if self.node_performance_history:
            all_durations = []
            for durations in self.node_performance_history.values():
                all_durations.extend(durations)
            
            if all_durations:
                summary["overall_performance"] = {
                    "average_node_duration": sum(all_durations) / len(all_durations),
                    "total_execution_time": sum(all_durations),
                    "fastest_execution": min(all_durations),
                    "slowest_execution": max(all_durations)
                }
        
        return summary

class WorkflowDebugger:
    """Advanced debugging tools for workflow execution"""
    
    def __init__(self):
        self.execution_traces: Dict[str, WorkflowExecutionTrace] = {}
        self.breakpoints: Dict[str, List[Callable]] = defaultdict(list)  # node_name -> [condition_functions]
        self.state_snapshots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.debug_mode = False
        self.step_mode = False
        self.current_execution_id: Optional[str] = None
        
    def enable_debug_mode(self):
        """Enable debug mode for detailed tracing"""
        self.debug_mode = True
        logger.info("Debug mode enabled")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_mode = False
        logger.info("Debug mode disabled")
    
    def set_breakpoint(self, node_name: str, condition: Optional[Callable[[TTDRState], bool]] = None):
        """Set a breakpoint at a specific node"""
        if condition is None:
            condition = lambda state: True  # Always break
        
        self.breakpoints[node_name].append(condition)
        logger.info(f"Breakpoint set for node: {node_name}")
    
    def clear_breakpoints(self, node_name: Optional[str] = None):
        """Clear breakpoints for a specific node or all nodes"""
        if node_name:
            self.breakpoints[node_name].clear()
            logger.info(f"Breakpoints cleared for node: {node_name}")
        else:
            self.breakpoints.clear()
            logger.info("All breakpoints cleared")
    
    def should_break(self, node_name: str, state: TTDRState) -> bool:
        """Check if execution should break at this node"""
        if not self.debug_mode:
            return False
        
        for condition in self.breakpoints.get(node_name, []):
            try:
                if condition(state):
                    logger.info(f"Breakpoint hit at node: {node_name}")
                    return True
            except Exception as e:
                logger.error(f"Error evaluating breakpoint condition: {e}")
        
        return False
    
    def start_execution_trace(self, execution_id: str, topic: str):
        """Start tracing a workflow execution"""
        trace = WorkflowExecutionTrace(
            execution_id=execution_id,
            topic=topic,
            start_time=datetime.now()
        )
        self.execution_traces[execution_id] = trace
        self.current_execution_id = execution_id
        logger.info(f"Started execution trace: {execution_id}")
    
    def trace_node_execution(self, node_name: str, execution_id: str, 
                           input_state: TTDRState, start_time: datetime):
        """Start tracing a node execution"""
        if execution_id not in self.execution_traces:
            logger.warning(f"No execution trace found for: {execution_id}")
            return
        
        # Capture memory usage before execution
        memory_before = None
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                pass
        
        node_trace = NodeExecutionTrace(
            node_name=node_name,
            execution_id=execution_id,
            start_time=start_time,
            status=NodeStatus.RUNNING,
            input_state_summary=self._summarize_state(input_state),
            memory_usage_before=memory_before
        )
        
        self.execution_traces[execution_id].node_traces.append(node_trace)
        self.execution_traces[execution_id].execution_path.append(node_name)
        
        # Take state snapshot if in debug mode
        if self.debug_mode:
            self.state_snapshots[execution_id].append({
                "node_name": node_name,
                "timestamp": start_time,
                "state": self._deep_copy_state(input_state)
            })
    
    def complete_node_trace(self, node_name: str, execution_id: str, 
                          output_state: TTDRState, end_time: datetime,
                          error: Optional[Exception] = None):
        """Complete a node execution trace"""
        if execution_id not in self.execution_traces:
            return
        
        # Find the most recent node trace for this node
        node_traces = self.execution_traces[execution_id].node_traces
        node_trace = None
        for trace in reversed(node_traces):
            if trace.node_name == node_name and trace.end_time is None:
                node_trace = trace
                break
        
        if not node_trace:
            logger.warning(f"No active node trace found for: {node_name}")
            return
        
        # Complete the trace
        node_trace.end_time = end_time
        node_trace.duration = (end_time - node_trace.start_time).total_seconds()
        node_trace.output_state_summary = self._summarize_state(output_state)
        
        # Capture memory usage after execution
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                node_trace.memory_usage_after = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                pass
        
        # Set status and error details
        if error:
            node_trace.status = NodeStatus.FAILED
            node_trace.error_details = str(error)
            self.execution_traces[execution_id].total_errors += 1
        else:
            node_trace.status = NodeStatus.COMPLETED
        
        # Update execution trace
        execution_trace = self.execution_traces[execution_id]
        if output_state.get("quality_metrics"):
            execution_trace.final_quality_score = output_state["quality_metrics"].overall_score
        execution_trace.iteration_count = output_state.get("iteration_count", 0)
    
    def complete_execution_trace(self, execution_id: str, final_state: TTDRState,
                               status: str = "completed"):
        """Complete a workflow execution trace"""
        if execution_id not in self.execution_traces:
            return
        
        trace = self.execution_traces[execution_id]
        trace.end_time = datetime.now()
        trace.total_duration = (trace.end_time - trace.start_time).total_seconds()
        trace.status = status
        
        if final_state.get("quality_metrics"):
            trace.final_quality_score = final_state["quality_metrics"].overall_score
        trace.iteration_count = final_state.get("iteration_count", 0)
        
        # Calculate memory peak
        memory_usages = []
        for node_trace in trace.node_traces:
            if node_trace.memory_usage_before:
                memory_usages.append(node_trace.memory_usage_before)
            if node_trace.memory_usage_after:
                memory_usages.append(node_trace.memory_usage_after)
        
        if memory_usages:
            trace.memory_peak = max(memory_usages)
        
        logger.info(f"Completed execution trace: {execution_id}")
    
    def get_execution_trace(self, execution_id: str) -> Optional[WorkflowExecutionTrace]:
        """Get execution trace for a specific execution"""
        return self.execution_traces.get(execution_id)
    
    def get_node_trace_history(self, node_name: str, limit: int = 10) -> List[NodeExecutionTrace]:
        """Get execution history for a specific node"""
        node_traces = []
        
        for execution_trace in self.execution_traces.values():
            for node_trace in execution_trace.node_traces:
                if node_trace.node_name == node_name:
                    node_traces.append(node_trace)
        
        # Sort by start time and return most recent
        node_traces.sort(key=lambda x: x.start_time, reverse=True)
        return node_traces[:limit]
    
    def inspect_state(self, execution_id: str, node_name: Optional[str] = None) -> Dict[str, Any]:
        """Inspect workflow state at a specific point"""
        if execution_id not in self.state_snapshots:
            return {"error": "No state snapshots found for execution"}
        
        snapshots = self.state_snapshots[execution_id]
        
        if node_name:
            # Find snapshot for specific node
            for snapshot in reversed(snapshots):
                if snapshot["node_name"] == node_name:
                    return {
                        "node_name": node_name,
                        "timestamp": snapshot["timestamp"],
                        "state": snapshot["state"]
                    }
            return {"error": f"No state snapshot found for node: {node_name}"}
        else:
            # Return most recent snapshot
            if snapshots:
                latest = snapshots[-1]
                return {
                    "node_name": latest["node_name"],
                    "timestamp": latest["timestamp"],
                    "state": latest["state"]
                }
            return {"error": "No state snapshots available"}
    
    def generate_execution_report(self, execution_id: str) -> Dict[str, Any]:
        """Generate a comprehensive execution report"""
        if execution_id not in self.execution_traces:
            return {"error": "Execution trace not found"}
        
        trace = self.execution_traces[execution_id]
        
        report = {
            "execution_summary": {
                "execution_id": execution_id,
                "topic": trace.topic,
                "start_time": trace.start_time.isoformat(),
                "end_time": trace.end_time.isoformat() if trace.end_time else None,
                "total_duration": trace.total_duration,
                "status": trace.status,
                "iteration_count": trace.iteration_count,
                "final_quality_score": trace.final_quality_score,
                "total_errors": trace.total_errors,
                "total_warnings": trace.total_warnings,
                "memory_peak": trace.memory_peak
            },
            "execution_path": trace.execution_path,
            "node_performance": [],
            "error_analysis": [],
            "performance_insights": []
        }
        
        # Analyze node performance
        for node_trace in trace.node_traces:
            node_perf = {
                "node_name": node_trace.node_name,
                "duration": node_trace.duration,
                "status": node_trace.status.value,
                "memory_usage": {
                    "before": node_trace.memory_usage_before,
                    "after": node_trace.memory_usage_after,
                    "delta": (node_trace.memory_usage_after - node_trace.memory_usage_before) 
                            if (node_trace.memory_usage_before and node_trace.memory_usage_after) else None
                }
            }
            report["node_performance"].append(node_perf)
            
            # Collect errors
            if node_trace.error_details:
                report["error_analysis"].append({
                    "node_name": node_trace.node_name,
                    "error": node_trace.error_details,
                    "timestamp": node_trace.start_time.isoformat()
                })
        
        # Generate performance insights
        if report["node_performance"]:
            durations = [p["duration"] for p in report["node_performance"] if p["duration"]]
            if durations:
                total_time = sum(durations)
                slowest_node = max(report["node_performance"], key=lambda x: x["duration"] or 0)
                
                report["performance_insights"] = [
                    f"Total execution time: {total_time:.2f}s",
                    f"Slowest node: {slowest_node['node_name']} ({slowest_node['duration']:.2f}s)",
                    f"Average node duration: {total_time / len(durations):.2f}s"
                ]
                
                if trace.memory_peak:
                    report["performance_insights"].append(f"Peak memory usage: {trace.memory_peak:.1f} MB")
        
        return report
    
    def _summarize_state(self, state: TTDRState) -> Dict[str, Any]:
        """Create a summary of workflow state for tracing"""
        return {
            "topic": state.get("topic", ""),
            "iteration_count": state.get("iteration_count", 0),
            "has_draft": state.get("current_draft") is not None,
            "gaps_count": len(state.get("information_gaps", [])),
            "retrieved_info_count": len(state.get("retrieved_info", [])),
            "has_quality_metrics": state.get("quality_metrics") is not None,
            "evolution_history_count": len(state.get("evolution_history", [])),
            "has_final_report": state.get("final_report") is not None,
            "error_count": len(state.get("error_log", []))
        }
    
    def _deep_copy_state(self, state: TTDRState) -> Dict[str, Any]:
        """Create a deep copy of state for debugging (simplified)"""
        # This is a simplified version - in production you'd want more sophisticated copying
        try:
            return json.loads(json.dumps(state, default=str))
        except Exception:
            return self._summarize_state(state)

class WorkflowMonitor:
    """Comprehensive workflow monitoring system"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.debugger = WorkflowDebugger()
        self.monitoring_enabled = False
        self.log_file: Optional[Path] = None
        self.real_time_callbacks: List[Callable] = []
        
    def enable_monitoring(self, log_file: Optional[str] = None, 
                         enable_profiling: bool = True,
                         enable_debugging: bool = False):
        """Enable comprehensive monitoring"""
        self.monitoring_enabled = True
        
        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if enable_profiling:
            self.profiler.start_monitoring()
        
        if enable_debugging:
            self.debugger.enable_debug_mode()
        
        logger.info("Workflow monitoring enabled")
    
    def disable_monitoring(self):
        """Disable monitoring"""
        self.monitoring_enabled = False
        self.profiler.stop_monitoring()
        self.debugger.disable_debug_mode()
        logger.info("Workflow monitoring disabled")
    
    def add_real_time_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for real-time monitoring updates"""
        self.real_time_callbacks.append(callback)
    
    def monitor_node_execution(self, node_name: str, execution_id: str,
                             input_state: TTDRState, output_state: TTDRState,
                             duration: float, error: Optional[Exception] = None):
        """Monitor a node execution"""
        if not self.monitoring_enabled:
            return
        
        # Record performance
        self.profiler.record_node_performance(node_name, duration)
        
        # Create monitoring event
        event = {
            "type": "node_execution",
            "timestamp": datetime.now().isoformat(),
            "execution_id": execution_id,
            "node_name": node_name,
            "duration": duration,
            "status": "failed" if error else "completed",
            "error": str(error) if error else None,
            "input_summary": self.debugger._summarize_state(input_state),
            "output_summary": self.debugger._summarize_state(output_state)
        }
        
        # Log to file if configured
        if self.log_file:
            self._log_event(event)
        
        # Notify real-time callbacks
        for callback in self.real_time_callbacks:
            try:
                callback("node_execution", event)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
    
    def monitor_workflow_execution(self, execution_id: str, topic: str,
                                 final_state: TTDRState, total_duration: float,
                                 status: str = "completed"):
        """Monitor a complete workflow execution"""
        if not self.monitoring_enabled:
            return
        
        event = {
            "type": "workflow_execution",
            "timestamp": datetime.now().isoformat(),
            "execution_id": execution_id,
            "topic": topic,
            "total_duration": total_duration,
            "status": status,
            "final_summary": self.debugger._summarize_state(final_state)
        }
        
        # Log to file if configured
        if self.log_file:
            self._log_event(event)
        
        # Notify real-time callbacks
        for callback in self.real_time_callbacks:
            try:
                callback("workflow_execution", event)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": {
                "enabled": self.monitoring_enabled,
                "profiling_active": self.profiler._monitoring_active,
                "debug_mode": self.debugger.debug_mode
            },
            "performance_summary": self.profiler.get_performance_summary(),
            "recent_executions": [],
            "system_health": self._get_system_health(),
            "alerts": self._get_alerts()
        }
        
        # Add recent execution traces
        recent_traces = sorted(
            self.debugger.execution_traces.values(),
            key=lambda x: x.start_time,
            reverse=True
        )[:10]
        
        for trace in recent_traces:
            dashboard["recent_executions"].append({
                "execution_id": trace.execution_id,
                "topic": trace.topic,
                "start_time": trace.start_time.isoformat(),
                "duration": trace.total_duration,
                "status": trace.status,
                "quality_score": trace.final_quality_score,
                "iteration_count": trace.iteration_count
            })
        
        return dashboard
    
    def _log_event(self, event: Dict[str, Any]):
        """Log monitoring event to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to log monitoring event: {e}")
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        if not PSUTIL_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "psutil not available - system monitoring disabled"
            }
            
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                "memory_usage_percent": memory.percent,
                "cpu_usage_percent": cpu_percent,
                "available_memory_gb": memory.available / (1024**3),
                "status": "healthy" if memory.percent < 85 and cpu_percent < 90 else "warning"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get current system alerts"""
        alerts = []
        
        # Performance alerts
        bottleneck_analysis = self.profiler.bottleneck_analysis
        if bottleneck_analysis:
            for recommendation in bottleneck_analysis.get("recommendations", []):
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": recommendation,
                    "timestamp": datetime.now().isoformat()
                })
        
        # System resource alerts
        system_health = self._get_system_health()
        if system_health.get("status") == "warning":
            if system_health.get("memory_usage_percent", 0) > 85:
                alerts.append({
                    "type": "system",
                    "severity": "warning",
                    "message": f"High memory usage: {system_health['memory_usage_percent']:.1f}%",
                    "timestamp": datetime.now().isoformat()
                })
            
            if system_health.get("cpu_usage_percent", 0) > 90:
                alerts.append({
                    "type": "system",
                    "severity": "warning",
                    "message": f"High CPU usage: {system_health['cpu_usage_percent']:.1f}%",
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts

# Export main classes
__all__ = [
    "NodeExecutionTrace",
    "WorkflowExecutionTrace", 
    "PerformanceProfiler",
    "WorkflowDebugger",
    "WorkflowMonitor"
]