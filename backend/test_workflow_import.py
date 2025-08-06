#!/usr/bin/env python3

import sys
import traceback

print("Testing workflow recovery imports...")

# Test individual imports
try:
    print("1. Testing models.core...")
    from models.core import TTDRState
    print("   ✓ TTDRState imported")
except Exception as e:
    print(f"   ✗ TTDRState failed: {e}")
    traceback.print_exc()

try:
    print("2. Testing services.error_handling...")
    from services.error_handling import ErrorHandlingFramework, RecoveryStrategy
    print("   ✓ error_handling imported")
except Exception as e:
    print(f"   ✗ error_handling failed: {e}")
    traceback.print_exc()

# Test the workflow recovery module step by step
try:
    print("3. Testing workflow_recovery module...")
    
    # Import the module
    import services.workflow_recovery as wr
    print(f"   ✓ Module imported: {wr}")
    
    # Check what's in the module
    attrs = [attr for attr in dir(wr) if not attr.startswith('_')]
    print(f"   Available attributes: {attrs}")
    
    # Try to access specific classes
    if hasattr(wr, 'WorkflowRecoveryManager'):
        print("   ✓ WorkflowRecoveryManager found")
        manager = wr.WorkflowRecoveryManager()
        print("   ✓ WorkflowRecoveryManager instantiated")
    else:
        print("   ✗ WorkflowRecoveryManager not found")
        
except Exception as e:
    print(f"   ✗ workflow_recovery failed: {e}")
    traceback.print_exc()

print("Import test complete.")