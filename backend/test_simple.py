#!/usr/bin/env python3

# Test if we can define a simple class in the workflow_recovery module
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

# Try to import and check what happens
try:
    print("Importing workflow_recovery...")
    import services.workflow_recovery as wr
    
    print("Module imported successfully")
    print("Module dict keys:", list(wr.__dict__.keys()))
    
    # Check if the classes are defined in the module's namespace
    if hasattr(wr, 'WorkflowRecoveryManager'):
        print("WorkflowRecoveryManager found!")
    else:
        print("WorkflowRecoveryManager NOT found")
        
    # Try to access the class directly from the module dict
    if 'WorkflowRecoveryManager' in wr.__dict__:
        print("WorkflowRecoveryManager in __dict__")
    else:
        print("WorkflowRecoveryManager NOT in __dict__")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()