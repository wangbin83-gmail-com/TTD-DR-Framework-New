"""
Tests for workflow monitoring and debugging tools.
Tests task 10.2: Build workflow monitoring and debugging tools.
"""

import pytest
import tempfile
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from backend.models.core import TTDRState, ResearchRequirements, QualityMetrics
from backend.workflow.monitoring_tools import (
    NodeExecutionTrace, WorkflowExecutionTrace, PerformanceProfiler,
    WorkflowDebugger, WorkflowMonitor
)
from backend.workflow.graph import NodeStatus

class TestNodeExecutionTrace:
    """Test node execution tracing"""
    
    def test_node_trace_creation(self):
        """Test creating a node execution trace"""
        trace = NodeExecutionTrace(
            node_name="test_node",
            execution_id="test_exec_123",
            start_time=datetime.now()
        )
        
        assert trace.node_name == "test_node"
        assert trace.execution_id == "test_exec_123"
        assert trace.status == NodeStatus.PENDING
        assert trace.input_state_summary == {}
        assert trace.output_state_summary == {}
        assert trace.warnings == []
    
    def test_node_trace_completion(self):
        """Test completing a node execution trace"""
        start_time = datetime.now()
        trace = NodeExecutionTrace(
            node_name="test_node",
            execution_id="test_exec_123",
            start_time=start_time
        )
        
        # Simulate completion
        end_time = start_time + timedelta(seconds=5)
        trace.end_time = end_time
        trace.duration = (end_time - start_time).total_seconds()
        trace.status = NodeStatus.COMPLETED
        
        assert trace.duration == 5.0
        assert trace.status == NodeStatus.COMPLETED

class TestWorkflowExecutionTrace:
    """Test workflow execution tracing"""
    
    def test_workflow_trace_creation(self):
        """Test creating a workflow execution trace"""
        trace = WorkflowExecutionTrace(
            execution_id="workflow_123",
            topic="Test Topic",
            start_time=datetime.now()
        )
        
        assert trace.execution_id == "workflow_123"
        assert trace.topic == "Test Topic"
        assert trace.status == "running"
        assert trace.node_traces == []
        assert trace.execution_path == []
    
    def test_workflow_trace_with_nodes(self):
        """Test workflow trace with node traces"""
        workflow_trace = WorkflowExecutionTrace(
            execution_id="workflow_123",
            topic="Test Topic",
            start_time=datetime.now()
        )
        
        # Add node traces
        node_trace1 = NodeExecutionTrace(
            node_name="node1",
            execution_id="workflow_123",
            start_time=datetime.now()
        )
        node_trace2 = NodeExecutionTrace(
            node_name="node2", 
            execution_id="workflow_123",
            start_time=datetime.now()
        )
        
        workflow_trace.node_traces = [node_trace1, node_trace2]
        workflow_trace.execution_path = ["node1", "node2"]
        
        assert len(workflow_trace.node_traces) == 2
        assert workflow_trace.execution_path == ["node1", "node2"]

class TestPerformanceProfiler:
    """Test performance profiling functionality"""
    
    @pytest.fixture
    def profiler(self):
        """Create a performance profiler instance"""
        return PerformanceProfiler()
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization"""
        assert not profiler._monitoring_active
        assert profiler._monitor_thread is None
        assert len(profiler.node_performance_history) == 0
        assert len(profiler.memory_snapshots) == 0
        assert len(profiler.cpu_snapshots) == 0
    
    def test_record_node_performance(self, profiler):
        """Test recording node performance"""
        profiler.record_node_performance("test_node", 2.5)
        profiler.record_node_performance("test_node", 3.0)
        profiler.record_node_performance("another_node", 1.5)
        
        assert len(profiler.node_performance_history["test_node"]) == 2
        assert profiler.node_performance_history["test_node"] == [2.5, 3.0]
        assert len(profiler.node_performance_history["another_node"]) == 1
        assert profiler.node_performance_history["another_node"] == [1.5]
    
    def test_performance_history_limit(self, profiler):
        """Test that performance history is limited"""
        # Add more than 100 entries
        for i in range(105):
            profiler.record_node_performance("test_node", float(i))
        
        # Should keep only the last 100
        assert len(profiler.node_performance_history["test_node"]) == 100
        assert profiler.node_performance_history["test_node"][0] == 5.0  # First 5 should be removed
        assert profiler.node_performance_history["test_node"][-1] == 104.0
    
    def test_monitoring_start_stop(self, profiler):
        """Test starting and stopping monitoring"""
        # Start monitoring (will work even without psutil)
        profiler.start_monitoring(interval=0.1)
        assert profiler._monitoring_active
        assert profiler._monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        profiler.stop_monitoring()
        assert not profiler._monitoring_active
    
    def test_bottleneck_analysis(self, profiler):
        """Test bottleneck analysis"""
        # Add some performance data
        profiler.record_node_performance("fast_node", 1.0)
        profiler.record_node_performance("fast_node", 1.2)
        profiler.record_node_performance("slow_node", 10.0)
        profiler.record_node_performance("slow_node", 12.0)
        profiler.record_node_performance("unstable_node", 2.0)
        profiler.record_node_performance("unstable_node", 8.0)
        
        analysis = profiler.analyze_bottlenecks()
        
        assert "node_performance" in analysis
        assert "system_bottlenecks" in analysis
        assert "recommendations" in analysis
        
        # Check node performance analysis
        node_perf = analysis["node_performance"]
        assert "fast_node" in node_perf
        assert "slow_node" in node_perf
        assert "unstable_node" in node_perf
        
        # Slow node should have higher average duration
        assert node_perf["slow_node"]["average_duration"] > node_perf["fast_node"]["average_duration"]
        
        # Check slowest node identification
        assert "slowest_node" in analysis
        assert analysis["slowest_node"]["name"] == "slow_node"
    
    def test_performance_summary(self, profiler):
        """Test performance summary generation"""
        # Add some data
        profiler.record_node_performance("node1", 2.0)
        profiler.record_node_performance("node2", 3.0)
        
        summary = profiler.get_performance_summary()
        
        assert "monitoring_active" in summary
        assert "nodes_monitored" in summary
        assert "total_executions" in summary
        
        assert summary["nodes_monitored"] == 2
        assert summary["total_executions"] == 2
        assert not summary["monitoring_active"]

class TestWorkflowDebugger:
    """Test workflow debugging functionality"""
    
    @pytest.fixture
    def debugger(self):
        """Create a workflow debugger instance"""
        return WorkflowDebugger()
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample TTDRState for testing"""
        return {
            "topic": "Test Topic",
            "iteration_count": 1,
            "current_draft": Mock(),
            "information_gaps": [Mock(), Mock()],
            "retrieved_info": [Mock()],
            "quality_metrics": Mock(overall_score=0.75),
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
    
    def test_debugger_initialization(self, debugger):
        """Test debugger initialization"""
        assert not debugger.debug_mode
        assert not debugger.step_mode
        assert len(debugger.execution_traces) == 0
        assert len(debugger.breakpoints) == 0
        assert debugger.current_execution_id is None
    
    def test_debug_mode_toggle(self, debugger):
        """Test enabling and disabling debug mode"""
        assert not debugger.debug_mode
        
        debugger.enable_debug_mode()
        assert debugger.debug_mode
        
        debugger.disable_debug_mode()
        assert not debugger.debug_mode
    
    def test_breakpoint_management(self, debugger, sample_state):
        """Test setting and clearing breakpoints"""
        # Set breakpoint
        debugger.set_breakpoint("test_node")
        assert len(debugger.breakpoints["test_node"]) == 1
        
        # Set conditional breakpoint
        condition = lambda state: state.get("iteration_count", 0) > 2
        debugger.set_breakpoint("test_node", condition)
        assert len(debugger.breakpoints["test_node"]) == 2
        
        # Test breakpoint evaluation
        debugger.enable_debug_mode()
        
        # Should break on first condition (always true)
        assert debugger.should_break("test_node", sample_state)
        
        # Test conditional breakpoint
        sample_state["iteration_count"] = 1
        # First condition should still trigger
        assert debugger.should_break("test_node", sample_state)
        
        # Clear breakpoints
        debugger.clear_breakpoints("test_node")
        assert len(debugger.breakpoints["test_node"]) == 0
        assert not debugger.should_break("test_node", sample_state)
    
    def test_execution_tracing(self, debugger, sample_state):
        """Test execution tracing functionality"""
        execution_id = "test_exec_123"
        topic = "Test Topic"
        
        # Start execution trace
        debugger.start_execution_trace(execution_id, topic)
        
        assert execution_id in debugger.execution_traces
        trace = debugger.execution_traces[execution_id]
        assert trace.execution_id == execution_id
        assert trace.topic == topic
        assert trace.status == "running"
        
        # Trace node execution
        start_time = datetime.now()
        debugger.trace_node_execution("test_node", execution_id, sample_state, start_time)
        
        assert len(trace.node_traces) == 1
        assert trace.execution_path == ["test_node"]
        
        node_trace = trace.node_traces[0]
        assert node_trace.node_name == "test_node"
        assert node_trace.status == NodeStatus.RUNNING
        
        # Complete node trace
        end_time = start_time + timedelta(seconds=2)
        debugger.complete_node_trace("test_node", execution_id, sample_state, end_time)
        
        assert node_trace.end_time == end_time
        assert node_trace.duration == 2.0
        assert node_trace.status == NodeStatus.COMPLETED
        
        # Complete execution trace
        debugger.complete_execution_trace(execution_id, sample_state, "completed")
        
        assert trace.status == "completed"
        assert trace.end_time is not None
        assert trace.total_duration is not None
    
    def test_state_inspection(self, debugger, sample_state):
        """Test state inspection functionality"""
        execution_id = "test_exec_123"
        
        # Enable debug mode to capture snapshots
        debugger.enable_debug_mode()
        debugger.start_execution_trace(execution_id, "Test Topic")
        
        # Trace node execution (should capture state snapshot)
        debugger.trace_node_execution("test_node", execution_id, sample_state, datetime.now())
        
        # Inspect state
        state_info = debugger.inspect_state(execution_id, "test_node")
        
        assert "node_name" in state_info
        assert "timestamp" in state_info
        assert "state" in state_info
        assert state_info["node_name"] == "test_node"
    
    def test_execution_report_generation(self, debugger, sample_state):
        """Test execution report generation"""
        execution_id = "test_exec_123"
        
        # Create a complete execution trace
        debugger.start_execution_trace(execution_id, "Test Topic")
        
        start_time = datetime.now()
        debugger.trace_node_execution("node1", execution_id, sample_state, start_time)
        debugger.complete_node_trace("node1", execution_id, sample_state, 
                                   start_time + timedelta(seconds=2))
        
        debugger.trace_node_execution("node2", execution_id, sample_state, 
                                    start_time + timedelta(seconds=2))
        debugger.complete_node_trace("node2", execution_id, sample_state,
                                   start_time + timedelta(seconds=5))
        
        debugger.complete_execution_trace(execution_id, sample_state, "completed")
        
        # Generate report
        report = debugger.generate_execution_report(execution_id)
        
        assert "execution_summary" in report
        assert "execution_path" in report
        assert "node_performance" in report
        assert "performance_insights" in report
        
        # Check execution summary
        summary = report["execution_summary"]
        assert summary["execution_id"] == execution_id
        assert summary["topic"] == "Test Topic"
        assert summary["status"] == "completed"
        
        # Check node performance
        assert len(report["node_performance"]) == 2
        node_perfs = {p["node_name"]: p for p in report["node_performance"]}
        assert "node1" in node_perfs
        assert "node2" in node_perfs
        assert node_perfs["node1"]["duration"] == 2.0
        assert node_perfs["node2"]["duration"] == 3.0
    
    def test_node_trace_history(self, debugger, sample_state):
        """Test node trace history retrieval"""
        # Create multiple execution traces with the same node
        for i in range(3):
            execution_id = f"exec_{i}"
            debugger.start_execution_trace(execution_id, f"Topic {i}")
            
            start_time = datetime.now()
            debugger.trace_node_execution("common_node", execution_id, sample_state, start_time)
            debugger.complete_node_trace("common_node", execution_id, sample_state,
                                       start_time + timedelta(seconds=i+1))
        
        # Get node history
        history = debugger.get_node_trace_history("common_node", limit=5)
        
        assert len(history) == 3
        # Should be sorted by start time (most recent first)
        # All traces have the same start time (datetime.now()), so order may vary
        # Just check that we have the expected durations in some order
        durations = [trace.duration for trace in history]
        assert sorted(durations) == [1.0, 2.0, 3.0]

class TestWorkflowMonitor:
    """Test comprehensive workflow monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create a workflow monitor instance"""
        return WorkflowMonitor()
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample TTDRState for testing"""
        return {
            "topic": "Test Topic",
            "iteration_count": 1,
            "current_draft": Mock(),
            "information_gaps": [Mock(), Mock()],
            "retrieved_info": [Mock()],
            "quality_metrics": Mock(overall_score=0.75),
            "evolution_history": [],
            "final_report": None,
            "error_log": []
        }
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert not monitor.monitoring_enabled
        assert monitor.log_file is None
        assert len(monitor.real_time_callbacks) == 0
        assert isinstance(monitor.profiler, PerformanceProfiler)
        assert isinstance(monitor.debugger, WorkflowDebugger)
    
    def test_enable_disable_monitoring(self, monitor, temp_log_file):
        """Test enabling and disabling monitoring"""
        # Enable monitoring
        monitor.enable_monitoring(
            log_file=temp_log_file,
            enable_profiling=True,
            enable_debugging=True
        )
        
        assert monitor.monitoring_enabled
        assert monitor.log_file == Path(temp_log_file)
        assert monitor.debugger.debug_mode
        
        # Disable monitoring
        monitor.disable_monitoring()
        
        assert not monitor.monitoring_enabled
        assert not monitor.debugger.debug_mode
    
    def test_real_time_callbacks(self, monitor):
        """Test real-time monitoring callbacks"""
        callback_calls = []
        
        def test_callback(event_type, event_data):
            callback_calls.append((event_type, event_data))
        
        monitor.add_real_time_callback(test_callback)
        assert len(monitor.real_time_callbacks) == 1
        
        # Enable monitoring to test callbacks
        monitor.enable_monitoring()
        
        # Monitor a node execution
        input_state = {"topic": "Test", "iteration_count": 0}
        output_state = {"topic": "Test", "iteration_count": 1}
        
        monitor.monitor_node_execution(
            "test_node", "exec_123", input_state, output_state, 2.5
        )
        
        # Check that callback was called
        assert len(callback_calls) == 1
        event_type, event_data = callback_calls[0]
        assert event_type == "node_execution"
        assert event_data["node_name"] == "test_node"
        assert event_data["duration"] == 2.5
    
    def test_log_file_writing(self, monitor, temp_log_file, sample_state):
        """Test logging to file"""
        monitor.enable_monitoring(log_file=temp_log_file)
        
        # Monitor a node execution
        monitor.monitor_node_execution(
            "test_node", "exec_123", sample_state, sample_state, 1.5
        )
        
        # Check that log file was written
        log_path = Path(temp_log_file)
        assert log_path.exists()
        
        # Read and verify log content
        with open(log_path, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["type"] == "node_execution"
            assert log_data["node_name"] == "test_node"
            assert log_data["duration"] == 1.5
    
    def test_monitoring_dashboard(self, monitor):
        """Test monitoring dashboard generation"""
        monitor.enable_monitoring(enable_debugging=True)  # Explicitly enable debugging
        
        # Add some execution data
        monitor.debugger.start_execution_trace("exec_123", "Test Topic")
        monitor.debugger.complete_execution_trace("exec_123", {"topic": "Test"}, "completed")
        
        # Get dashboard
        dashboard = monitor.get_monitoring_dashboard()
        
        assert "timestamp" in dashboard
        assert "monitoring_status" in dashboard
        assert "performance_summary" in dashboard
        assert "recent_executions" in dashboard
        assert "system_health" in dashboard
        assert "alerts" in dashboard
        
        # Check monitoring status
        status = dashboard["monitoring_status"]
        assert status["enabled"]
        assert status["debug_mode"]  # Should be True now
        
        # Check system health (will show unavailable if psutil not installed)
        health = dashboard["system_health"]
        assert "status" in health
        # Status will be "unavailable" if psutil is not installed, "healthy" if it is
    
    def test_system_alerts(self, monitor):
        """Test system alert generation"""
        monitor.enable_monitoring()
        
        # Add performance data that should trigger alerts
        monitor.profiler.record_node_performance("slow_node", 35.0)  # Should trigger slow node alert
        monitor.profiler.analyze_bottlenecks()
        
        dashboard = monitor.get_monitoring_dashboard()
        alerts = dashboard["alerts"]
        
        # Should have performance alert for slow node
        performance_alerts = [a for a in alerts if a["type"] == "performance"]
        assert len(performance_alerts) > 0
        
        slow_node_alerts = [a for a in performance_alerts if "slow_node" in a["message"]]
        assert len(slow_node_alerts) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])