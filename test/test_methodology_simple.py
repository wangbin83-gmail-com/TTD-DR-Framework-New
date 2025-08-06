"""
Simple test to verify methodology documentation methods exist and work
"""

import sys
sys.path.append('.')

# Test import
try:
    from backend.services.kimi_k2_report_synthesizer import KimiK2ReportSynthesizer
    print("✓ Successfully imported KimiK2ReportSynthesizer")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test instantiation
try:
    synthesizer = KimiK2ReportSynthesizer()
    print("✓ Successfully instantiated KimiK2ReportSynthesizer")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    exit(1)

# Test method existence
methods_to_check = [
    'generate_research_methodology_documentation',
    'generate_source_bibliography', 
    'generate_methodology_summary',
    '_extract_methodology_data',
    '_build_methodology_prompt',
    '_generate_technical_appendix'
]

print("\nChecking method existence:")
for method_name in methods_to_check:
    exists = hasattr(synthesizer, method_name)
    status = "✓" if exists else "❌"
    print(f"{status} {method_name}: {exists}")

# Check if methods are callable
print("\nChecking if methods are callable:")
for method_name in methods_to_check:
    if hasattr(synthesizer, method_name):
        method = getattr(synthesizer, method_name)
        is_callable = callable(method)
        status = "✓" if is_callable else "❌"
        print(f"{status} {method_name} is callable: {is_callable}")

# Test basic functionality with mock data
print("\nTesting basic functionality:")
try:
    # Test _extract_methodology_data with minimal state
    minimal_state = {
        "topic": "Test Topic",
        "iteration_count": 1,
        "retrieved_info": [],
        "information_gaps": [],
        "evolution_history": []
    }
    
    if hasattr(synthesizer, '_extract_methodology_data'):
        methodology_data = synthesizer._extract_methodology_data(minimal_state, None)
        print(f"✓ _extract_methodology_data returned: {type(methodology_data)}")
        print(f"  - Research topic: {methodology_data.get('research_topic', 'Not found')}")
        print(f"  - Iteration count: {methodology_data.get('iteration_count', 'Not found')}")
    else:
        print("❌ _extract_methodology_data method not found")

    # Test _generate_technical_appendix
    if hasattr(synthesizer, '_generate_technical_appendix'):
        test_data = {
            "research_topic": "Test Topic",
            "iteration_count": 1,
            "workflow_stages": [],
            "sources_used": [],
            "search_queries": [],
            "quality_metrics": None,
            "evolution_history": []
        }
        appendix = synthesizer._generate_technical_appendix(test_data)
        print(f"✓ _generate_technical_appendix returned: {len(appendix)} characters")
    else:
        print("❌ _generate_technical_appendix method not found")

except Exception as e:
    print(f"❌ Basic functionality test failed: {e}")

print("\n" + "="*60)
print("METHODOLOGY DOCUMENTATION IMPLEMENTATION TEST COMPLETE")
print("="*60)