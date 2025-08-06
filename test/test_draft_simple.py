import sys
sys.path.append('backend')

from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from workflow.draft_generator import KimiK2DraftGenerator

def test_fallback_structure():
    generator = KimiK2DraftGenerator()
    topic = "Test AI Topic"
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.INTERMEDIATE
    )
    
    try:
        structure = generator._create_fallback_structure(topic, requirements)
        print('✓ Fallback structure created successfully')
        print(f'✓ Sections: {len(structure.sections)}')
        print(f'✓ Domain: {structure.domain}')
        print(f'✓ Complexity: {structure.complexity_level}')
        print('✓ Test passed')
        return True
    except Exception as e:
        print(f'✗ Test failed: {e}')
        return False

if __name__ == "__main__":
    success = test_fallback_structure()
    sys.exit(0 if success else 1)