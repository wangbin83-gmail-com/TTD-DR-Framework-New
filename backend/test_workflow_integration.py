#!/usr/bin/env python3
"""
Integration test for TTD-DR workflow execution.
This script demonstrates the complete workflow from start to finish.
"""

import logging
from .workflow.graph import create_ttdr_workflow
from .models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from .models.state_management import create_workflow_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a complete TTD-DR workflow integration test"""
    
    print("=" * 60)
    print("TTD-DR Workflow Integration Test")
    print("=" * 60)
    
    # Create research requirements
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        max_iterations=2,
        quality_threshold=0.7,
        max_sources=10
    )
    
    print(f"Research Requirements:")
    print(f"  Domain: {requirements.domain}")
    print(f"  Complexity: {requirements.complexity_level}")
    print(f"  Max Iterations: {requirements.max_iterations}")
    print(f"  Quality Threshold: {requirements.quality_threshold}")
    print()
    
    # Create initial state
    topic = "Artificial Intelligence in Healthcare"
    initial_state = create_workflow_state(topic, requirements)
    
    print(f"Research Topic: {topic}")
    print()
    
    # Create and compile workflow
    print("Creating TTD-DR workflow...")
    workflow = create_ttdr_workflow()
    compiled_workflow = workflow.compile()
    print("Workflow compiled successfully!")
    print()
    
    # Execute workflow
    print("Executing workflow...")
    print("-" * 40)
    
    try:
        final_state = compiled_workflow.invoke(initial_state)
        
        print("-" * 40)
        print("Workflow execution completed successfully!")
        print()
        
        # Display results
        print("EXECUTION RESULTS:")
        print("=" * 40)
        print(f"Topic: {final_state['topic']}")
        print(f"Iterations completed: {final_state['iteration_count']}")
        print(f"Information gaps identified: {len(final_state['information_gaps'])}")
        print(f"Information sources retrieved: {len(final_state['retrieved_info'])}")
        print(f"Evolution records: {len(final_state['evolution_history'])}")
        
        if final_state['quality_metrics']:
            print(f"Final quality score: {final_state['quality_metrics'].overall_score:.2f}")
            print(f"  - Completeness: {final_state['quality_metrics'].completeness:.2f}")
            print(f"  - Coherence: {final_state['quality_metrics'].coherence:.2f}")
            print(f"  - Accuracy: {final_state['quality_metrics'].accuracy:.2f}")
            print(f"  - Citation Quality: {final_state['quality_metrics'].citation_quality:.2f}")
        
        print(f"Errors encountered: {len(final_state['error_log'])}")
        print()
        
        # Display execution path
        if workflow.execution_history:
            latest_execution = workflow.execution_history[-1]
            print("EXECUTION PATH:")
            print("=" * 40)
            execution_path = latest_execution['execution_path']
            for i, node in enumerate(execution_path, 1):
                print(f"{i:2d}. {node}")
            print()
        
        # Display final report preview
        if final_state['final_report']:
            print("FINAL REPORT PREVIEW:")
            print("=" * 40)
            report_lines = final_state['final_report'].split('\n')
            preview_lines = report_lines[:15]  # Show first 15 lines
            for line in preview_lines:
                print(line)
            
            if len(report_lines) > 15:
                print(f"\n... ({len(report_lines) - 15} more lines)")
            print()
        
        # Display workflow statistics
        print("WORKFLOW STATISTICS:")
        print("=" * 40)
        print(f"Total nodes in workflow: {len(workflow.nodes)}")
        print(f"Total edges in workflow: {len(workflow.edges)}")
        print(f"Entry point: {workflow.entry_point}")
        print(f"End nodes: {', '.join(workflow.end_nodes)}")
        print()
        
        print("Integration test completed successfully! ✅")
        
    except Exception as e:
        print(f"Workflow execution failed: {str(e)}")
        print("Integration test failed! ❌")
        raise

if __name__ == "__main__":
    main()