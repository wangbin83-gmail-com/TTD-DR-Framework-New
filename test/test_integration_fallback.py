import asyncio
import sys
import os
import traceback
from unittest.mock import patch
sys.path.append('backend')

from models.core import TTDRState, ResearchRequirements, ResearchDomain, ComplexityLevel
from workflow.draft_generator import draft_generator_node, KimiK2DraftGenerator
from services.kimi_k2_client import KimiK2Error

async def test_integration_with_fallback():
    print("Starting integration test with forced fallback...")
    
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
    
    print("State created, mocking Kimi K2 to force fallback...")
    
    try:
        # Mock the Kimi K2 client to force fallback behavior
        with patch.object(KimiK2DraftGenerator, '_create_research_structure') as mock_structure:
            # Make the structure creation use fallback
            generator = KimiK2DraftGenerator()
            mock_structure.side_effect = lambda topic, req: generator._create_fallback_structure(topic, req)
            
            result = await draft_generator_node(state)
            
        print('✓ Draft generator node executed successfully')
        print(f'✓ Draft created: {result["current_draft"] is not None}')
        if result['current_draft']:
            print(f'✓ Sections: {len(result["current_draft"].structure.sections)}')
            print(f'✓ Topic: {result["current_draft"].topic}')
            print(f'✓ Domain: {result["current_draft"].structure.domain}')
            print(f'✓ Complexity: {result["current_draft"].structure.complexity_level}')
        print('✓ Integration test passed')
        return True
    except Exception as e:
        print(f'✗ Integration test failed: {e}')
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_integration_with_fallback())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Failed to run async test: {e}")
        traceback.print_exc()
        sys.exit(1)