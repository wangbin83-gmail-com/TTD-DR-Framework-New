#!/usr/bin/env python3
"""
Demo script showing how to use the TTD-DR workflow framework.
This demonstrates the basic usage patterns for the LangGraph-based workflow.
"""

from workflow.graph import create_ttdr_workflow
from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from models.state_management import create_workflow_state

def demo_basic_workflow():
    """Demonstrate basic workflow usage"""
    print("🚀 TTD-DR Workflow Demo")
    print("=" * 50)
    
    # 1. Define research requirements
    requirements = ResearchRequirements(
        domain=ResearchDomain.SCIENCE,
        complexity_level=ComplexityLevel.BASIC,
        max_iterations=1,
        quality_threshold=0.5
    )
    
    # 2. Create initial state
    topic = "Climate Change Impact on Ocean Ecosystems"
    initial_state = create_workflow_state(topic, requirements)
    
    # 3. Create and compile workflow
    workflow = create_ttdr_workflow()
    compiled_workflow = workflow.compile()
    
    # 4. Execute workflow
    print(f"📝 Researching: {topic}")
    final_state = compiled_workflow.invoke(initial_state)
    
    # 5. Display results
    print(f"✅ Research completed in {final_state['iteration_count']} iterations")
    print(f"📊 Quality score: {final_state['quality_metrics'].overall_score:.2f}")
    print(f"📚 Sources found: {len(final_state['retrieved_info'])}")
    
    return final_state

def demo_advanced_workflow():
    """Demonstrate advanced workflow with higher complexity"""
    print("\n🔬 Advanced TTD-DR Workflow Demo")
    print("=" * 50)
    
    # Advanced requirements
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.ADVANCED,
        max_iterations=3,
        quality_threshold=0.8,
        max_sources=15
    )
    
    topic = "Quantum Computing Applications in Cryptography"
    initial_state = create_workflow_state(topic, requirements)
    
    workflow = create_ttdr_workflow()
    compiled_workflow = workflow.compile()
    
    print(f"🔍 Deep research on: {topic}")
    final_state = compiled_workflow.invoke(initial_state)
    
    print(f"✅ Advanced research completed!")
    print(f"🔄 Iterations: {final_state['iteration_count']}")
    print(f"📈 Evolution records: {len(final_state['evolution_history'])}")
    
    return final_state

def demo_workflow_customization():
    """Demonstrate workflow customization options"""
    print("\n⚙️  Workflow Customization Demo")
    print("=" * 50)
    
    # Create workflow
    workflow = create_ttdr_workflow()
    
    print(f"📋 Workflow Structure:")
    print(f"   Nodes: {len(workflow.nodes)}")
    print(f"   Edges: {len(workflow.edges)}")
    print(f"   Entry: {workflow.entry_point}")
    print(f"   Ends: {', '.join(workflow.end_nodes)}")
    
    print(f"\n🔗 Node List:")
    for i, node_name in enumerate(workflow.nodes.keys(), 1):
        node = workflow.nodes[node_name]
        print(f"   {i}. {node_name}: {node.description}")
    
    return workflow

if __name__ == "__main__":
    # Run all demos
    demo_basic_workflow()
    demo_advanced_workflow()
    demo_workflow_customization()
    
    print("\n🎉 All demos completed successfully!")