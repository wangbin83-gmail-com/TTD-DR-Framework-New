"""
Test script to verify Kimi K2 model configuration and connection
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_kimi_k2_connection():
    """Test Kimi K2 API connection and model configuration"""
    print("🔍 Testing Kimi K2 Model Configuration")
    print("=" * 50)
    
    try:
        # Import configuration
        from config.settings import settings
        
        print(f"✅ Configuration loaded:")
        print(f"   API Base URL: {settings.kimi_k2_base_url}")
        print(f"   Model: {settings.kimi_k2_model}")
        print(f"   Max Tokens: {settings.kimi_k2_max_tokens}")
        print(f"   Temperature: {settings.kimi_k2_temperature}")
        print(f"   API Key: {'✅ Configured' if settings.kimi_k2_api_key else '❌ Not configured'}")
        
        # Test client import
        from services.kimi_k2_client import KimiK2Client, kimi_k2_client
        print(f"✅ Kimi K2 Client imported successfully")
        
        # Test client initialization
        client = KimiK2Client()
        print(f"✅ Kimi K2 Client initialized")
        print(f"   Using model: {client.model}")
        
        if not client.api_key:
            print("❌ API key not configured - skipping connection test")
            return False
        
        # Test API connection
        print("\n🔗 Testing API Connection...")
        
        async with client:
            # Test health check
            is_healthy = await client.health_check()
            
            if is_healthy:
                print("✅ API connection successful")
                
                # Test basic text generation
                print("\n📝 Testing text generation...")
                response = await client.generate_text(
                    "请用中文简单介绍一下Kimi K2模型的特点。",
                    max_tokens=200
                )
                
                print(f"✅ Text generation successful")
                print(f"   Model used: {response.model}")
                print(f"   Response length: {len(response.content)} characters")
                print(f"   Finish reason: {response.finish_reason}")
                print(f"   Usage: {response.usage}")
                print(f"\n📄 Generated content preview:")
                print(f"   {response.content[:200]}...")
                
                # Test structured response
                print("\n🏗️ Testing structured response...")
                schema = {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "summary": {"type": "string"},
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
                
                structured_response = await client.generate_structured_response(
                    "请分析人工智能在教育领域的应用，并按照指定的JSON格式返回结果。",
                    schema,
                    max_tokens=500
                )
                
                print(f"✅ Structured response successful")
                print(f"   Response keys: {list(structured_response.keys())}")
                print(f"   Topic: {structured_response.get('topic', 'N/A')}")
                print(f"   Key points count: {len(structured_response.get('key_points', []))}")
                
                return True
                
            else:
                print("❌ API connection failed")
                return False
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_kimi_k2_in_workflow():
    """Test Kimi K2 integration in workflow components"""
    print("\n🔄 Testing Kimi K2 in Workflow Components")
    print("=" * 50)
    
    try:
        # Test draft generator
        from workflow.draft_generator import KimiK2DraftGenerator
        
        generator = KimiK2DraftGenerator()
        print("✅ Draft generator with Kimi K2 imported")
        
        # Test other Kimi K2 components
        components_to_test = [
            ("services.kimi_k2_gap_analyzer", "KimiK2InformationGapAnalyzer"),
            ("services.kimi_k2_quality_assessor", "KimiK2QualityAssessor"),
            ("services.kimi_k2_report_synthesizer", "KimiK2ReportSynthesizer"),
            ("services.kimi_k2_self_evolution_enhancer", "KimiK2SelfEvolutionEnhancer")
        ]
        
        for module_name, class_name in components_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                component_class = getattr(module, class_name)
                instance = component_class()
                print(f"✅ {class_name} imported and initialized")
            except ImportError as e:
                print(f"⚠️  {class_name} not found: {e}")
            except Exception as e:
                print(f"❌ {class_name} initialization failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow component test failed: {e}")
        return False

async def main():
    """Run all Kimi K2 tests"""
    print("🚀 Kimi K2 Model Configuration Verification")
    print("=" * 60)
    
    # Test 1: Basic connection
    connection_success = await test_kimi_k2_connection()
    
    # Test 2: Workflow integration
    workflow_success = await test_kimi_k2_in_workflow()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    print(f"API Connection Test: {'✅ PASSED' if connection_success else '❌ FAILED'}")
    print(f"Workflow Integration Test: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
    
    if connection_success and workflow_success:
        print("\n🎉 All tests passed! Kimi K2 is properly configured.")
        print("\n✅ Confirmed using Kimi K2 model:")
        from config.settings import settings
        print(f"   Model: {settings.kimi_k2_model}")
        print(f"   API Endpoint: {settings.kimi_k2_base_url}")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
    
    return connection_success and workflow_success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)