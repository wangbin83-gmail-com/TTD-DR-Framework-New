import asyncio
import sys
import os
import traceback
sys.path.append('backend')

from models.core import TTDRState, ResearchRequirements, ResearchDomain, ComplexityLevel
from workflow.draft_generator import draft_generator_node

async def test_integration():
    print("Starting integration test...")
    
    # Temporarily disable API key to force fallback behavior
    os.environ['KIMI_K2_API_KEY'] = ''
    
    state = {
        'topic': 'Test AI Topic',
        'requirements': ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE
        ),
        'current_draft': None,
        'information_gaps': [],
        'retrieved_info': [],
        'iteration_count': 0,
        'quality_metrics': None,
        'evolution_history': [],
        'final_report': None,
        'error_log': []
    }
    
    print("State created, calling draft_generator_node (with fallback mode)...")
    
    try:
        # Set a timeout for the operation
        result = await asyncio.wait_for(draft_generator_node(state), timeout=30.0)
        print('✓ Draft generator node executed successfully')
        print(f'✓ Draft created: {result["current_draft"] is not None}')
        if result['current_draft']:
            print(f'✓ Sections: {len(result["current_draft"].structure.sections)}')
            print(f'✓ Topic: {result["current_draft"].topic}')
            print(f'✓ Domain: {result["current_draft"].structure.domain}')
        print('✓ Integration test passed')
        return True
    except asyncio.TimeoutError:
        print('✗ Integration test timed out (30 seconds)')
        return False
    except Exception as e:
        print(f'✗ Integration test failed: {e}')
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_integration())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Failed to run async test: {e}")
        traceback.print_exc()
        sys.exit(1)