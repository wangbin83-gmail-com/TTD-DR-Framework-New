try:
    import services.workflow_recovery
    print("Import successful")
    print("Available attributes:", dir(services.workflow_recovery))
    
    # Try to import specific classes
    try:
        from services.workflow_recovery import WorkflowRecoveryManager
        print("WorkflowRecoveryManager imported successfully")
    except ImportError as e:
        print(f"Failed to import WorkflowRecoveryManager: {e}")
        
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()