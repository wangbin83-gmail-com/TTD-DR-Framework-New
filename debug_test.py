import sys
import traceback

print("Starting debug test...")

try:
    print("1. Adding backend to path...")
    sys.path.append('backend')
    
    print("2. Importing models...")
    from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
    print("   ✓ Models imported successfully")
    
    print("3. Importing draft generator...")
    from workflow.draft_generator import KimiK2DraftGenerator
    print("   ✓ Draft generator imported successfully")
    
    print("4. Creating generator instance...")
    generator = KimiK2DraftGenerator()
    print("   ✓ Generator created successfully")
    
    print("5. Testing fallback structure...")
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.INTERMEDIATE
    )
    
    structure = generator._create_fallback_structure("Test Topic", requirements)
    print(f"   ✓ Fallback structure created with {len(structure.sections)} sections")
    
    print("6. All tests passed!")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()