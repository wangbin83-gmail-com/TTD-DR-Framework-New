"""
Simple verification script for Kimi K2 configuration
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def verify_kimi_k2_config():
    """Verify Kimi K2 configuration"""
    print("🔍 Kimi K2 Configuration Verification")
    print("=" * 50)
    
    try:
        # Check configuration
        from config.settings import settings
        
        print("✅ Configuration Details:")
        print(f"   Model: {settings.kimi_k2_model}")
        print(f"   API Base URL: {settings.kimi_k2_base_url}")
        print(f"   Max Tokens: {settings.kimi_k2_max_tokens}")
        print(f"   Temperature: {settings.kimi_k2_temperature}")
        print(f"   API Key: {'✅ Configured' if settings.kimi_k2_api_key else '❌ Missing'}")
        
        # Verify model name
        if settings.kimi_k2_model == "kimi-k2-0711-preview":
            print("✅ Using correct Kimi K2 model: kimi-k2-0711-preview")
        else:
            print(f"⚠️  Model may not be Kimi K2: {settings.kimi_k2_model}")
        
        # Check client import
        from services.kimi_k2_client import KimiK2Client
        client = KimiK2Client()
        print(f"✅ Kimi K2 Client initialized with model: {client.model}")
        
        # Check workflow components
        print("\n🔄 Workflow Components Using Kimi K2:")
        
        components = [
            "workflow.draft_generator.KimiK2DraftGenerator",
            "services.kimi_k2_gap_analyzer.KimiK2InformationGapAnalyzer", 
            "services.kimi_k2_quality_assessor.KimiK2QualityAssessor",
            "services.kimi_k2_report_synthesizer.KimiK2ReportSynthesizer"
        ]
        
        for component_path in components:
            try:
                module_path, class_name = component_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)
                print(f"   ✅ {class_name}")
            except ImportError:
                print(f"   ⚠️  {class_name} (not found)")
            except Exception as e:
                print(f"   ❌ {class_name} (error: {e})")
        
        print("\n🎯 Verification Result:")
        if settings.kimi_k2_model == "kimi-k2-0711-preview" and settings.kimi_k2_api_key:
            print("✅ CONFIRMED: System is configured to use Kimi K2 model")
            print("   Model: kimi-k2-0711-preview")
            print("   API: Moonshot AI (https://api.moonshot.cn/v1)")
            return True
        else:
            print("❌ Configuration issue detected")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_kimi_k2_config()
    
    if success:
        print("\n🎉 Kimi K2 configuration verified successfully!")
    else:
        print("\n⚠️  Please check Kimi K2 configuration.")
    
    exit(0 if success else 1)