import sys
import traceback

print("Testing imports step by step...")

try:
    print("1. Testing models.core import...")
    from models.core import TTDRState, Draft, QualityMetrics, InformationGap, RetrievedInfo
    print("   models.core imported successfully")
except Exception as e:
    print(f"   models.core import failed: {e}")
    traceback.print_exc()

try:
    print("2. Testing services.error_handling import...")
    from services.error_handling import (
        ErrorHandlingFramework, TTDRError, WorkflowError, ErrorSeverity, 
        ErrorCategory, RecoveryStrategy, ErrorContext
    )
    print("   services.error_handling imported successfully")
except Exception as e:
    print(f"   services.error_handling import failed: {e}")
    traceback.print_exc()

try:
    print("3. Testing workflow_recovery module execution...")
    import services.workflow_recovery
    print("   Module imported, checking contents...")
    print(f"   Module file: {services.workflow_recovery.__file__}")
    print(f"   Module attributes: {[attr for attr in dir(services.workflow_recovery) if not attr.startswith('__')]}")
except Exception as e:
    print(f"   workflow_recovery import failed: {e}")
    traceback.print_exc()

# Try to execute the file directly
try:
    print("4. Executing workflow_recovery.py directly...")
    exec(open('services/workflow_recovery.py').read())
    print("   Direct execution successful")
except Exception as e:
    print(f"   Direct execution failed: {e}")
    traceback.print_exc()