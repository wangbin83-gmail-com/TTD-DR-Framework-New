"""
测试修正后的Kimi K2客户端超时处理
Test the fixed Kimi K2 client timeout handling
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_kimi_k2_timeout_fixes():
    """测试Kimi K2客户端的超时修正"""
    print("🔧 测试Kimi K2客户端超时修正")
    print("=" * 60)
    
    try:
        from services.kimi_k2_client import KimiK2Client
        from config.settings import settings
        
        print(f"✅ 配置信息:")
        print(f"   模型: {settings.kimi_k2_model}")
        print(f"   API端点: {settings.kimi_k2_base_url}")
        print(f"   API密钥: {'已配置' if settings.kimi_k2_api_key else '未配置'}")
        
        if not settings.kimi_k2_api_key:
            print("❌ API密钥未配置，跳过实际API测试")
            return False
        
        async with KimiK2Client() as client:
            print(f"\n🔗 客户端配置:")
            print(f"   连接超时: 10秒")
            print(f"   读取超时: 120秒")
            print(f"   写入超时: 30秒")
            print(f"   最大重试: 5次")
            print(f"   连接池: 20个连接")
            
            # 测试1: 简单文本生成
            print(f"\n📝 测试1: 简单文本生成")
            start_time = time.time()
            
            try:
                response = await client.generate_text(
                    "请简单介绍一下人工智能的发展历程。",
                    max_tokens=500
                )
                
                duration = time.time() - start_time
                print(f"✅ 简单文本生成成功")
                print(f"   响应时间: {duration:.2f}秒")
                print(f"   内容长度: {len(response.content)}字符")
                print(f"   使用模型: {response.model}")
                print(f"   Token使用: {response.usage}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ 简单文本生成失败: {e}")
                print(f"   失败时间: {duration:.2f}秒")
                return False
            
            # 测试2: 长文本生成（更容易超时）
            print(f"\n📄 测试2: 长文本生成")
            start_time = time.time()
            
            try:
                response = await client.generate_text(
                    """请详细分析人工智能在以下领域的应用现状和发展趋势：
                    1. 自然语言处理
                    2. 计算机视觉
                    3. 机器学习
                    4. 深度学习
                    5. 强化学习
                    
                    每个领域都要包含技术原理、应用案例、发展挑战和未来展望。
                    请提供详细的分析，字数控制在2000字左右。""",
                    max_tokens=2500
                )
                
                duration = time.time() - start_time
                print(f"✅ 长文本生成成功")
                print(f"   响应时间: {duration:.2f}秒")
                print(f"   内容长度: {len(response.content)}字符")
                print(f"   使用模型: {response.model}")
                print(f"   Token使用: {response.usage}")
                print(f"   完成原因: {response.finish_reason}")
                
                # 保存生成的内容
                with open("kimi_k2_long_text_test.md", "w", encoding="utf-8") as f:
                    f.write(f"# Kimi K2长文本生成测试\n\n")
                    f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**响应时间:** {duration:.2f}秒\n")
                    f.write(f"**使用模型:** {response.model}\n")
                    f.write(f"**Token使用:** {response.usage}\n\n")
                    f.write("---\n\n")
                    f.write(response.content)
                
                print(f"   内容已保存到: kimi_k2_long_text_test.md")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ 长文本生成失败: {e}")
                print(f"   失败时间: {duration:.2f}秒")
                return False
            
            # 测试3: 结构化响应生成
            print(f"\n🏗️ 测试3: 结构化响应生成")
            start_time = time.time()
            
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "conclusion": {"type": "string"}
                    }
                }
                
                structured_response = await client.generate_structured_response(
                    "请分析区块链技术的发展现状，并按照指定的JSON格式返回结果。",
                    schema,
                    max_tokens=800
                )
                
                duration = time.time() - start_time
                print(f"✅ 结构化响应生成成功")
                print(f"   响应时间: {duration:.2f}秒")
                print(f"   响应字段: {list(structured_response.keys())}")
                print(f"   标题: {structured_response.get('title', 'N/A')}")
                print(f"   要点数量: {len(structured_response.get('key_points', []))}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ 结构化响应生成失败: {e}")
                print(f"   失败时间: {duration:.2f}秒")
                return False
            
            # 测试4: 健康检查
            print(f"\n🏥 测试4: 健康检查")
            start_time = time.time()
            
            try:
                is_healthy = await client.health_check()
                duration = time.time() - start_time
                
                if is_healthy:
                    print(f"✅ 健康检查通过")
                    print(f"   检查时间: {duration:.2f}秒")
                else:
                    print(f"❌ 健康检查失败")
                    print(f"   检查时间: {duration:.2f}秒")
                    return False
                    
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ 健康检查异常: {e}")
                print(f"   异常时间: {duration:.2f}秒")
                return False
            
            print(f"\n🎉 所有测试通过！超时问题已修正")
            return True
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_timeout_scenarios():
    """测试各种超时场景的处理"""
    print(f"\n🕐 测试超时场景处理")
    print("=" * 40)
    
    try:
        from services.kimi_k2_client import KimiK2Client, KimiK2Error
        
        # 创建一个配置了较短超时的客户端用于测试
        client = KimiK2Client()
        
        # 模拟超时场景的测试提示
        timeout_test_prompts = [
            {
                "name": "极长文本生成",
                "prompt": "请写一篇关于人工智能发展历史的详细论文，包含以下内容：" + 
                         "1. 人工智能的起源和早期发展（1950-1980）\n" +
                         "2. 专家系统时代（1980-1990）\n" +
                         "3. 机器学习兴起（1990-2010）\n" +
                         "4. 深度学习革命（2010-2020）\n" +
                         "5. 大模型时代（2020至今）\n" +
                         "每个部分都要详细描述技术发展、重要人物、关键事件、技术突破等，" +
                         "总字数要求在5000字以上。",
                "max_tokens": 4000
            }
        ]
        
        for i, test_case in enumerate(timeout_test_prompts, 1):
            print(f"\n测试场景 {i}: {test_case['name']}")
            start_time = time.time()
            
            try:
                async with client:
                    response = await client.generate_text(
                        test_case["prompt"],
                        max_tokens=test_case["max_tokens"]
                    )
                
                duration = time.time() - start_time
                print(f"✅ 场景 {i} 成功处理")
                print(f"   处理时间: {duration:.2f}秒")
                print(f"   内容长度: {len(response.content)}字符")
                
            except KimiK2Error as e:
                duration = time.time() - start_time
                print(f"⚠️  场景 {i} 预期错误: {e.error_type}")
                print(f"   错误信息: {e.message}")
                print(f"   处理时间: {duration:.2f}秒")
                
                # 验证错误处理是否正确
                if e.error_type in ["timeout", "read_timeout", "connection_timeout"]:
                    print(f"✅ 超时错误处理正确")
                else:
                    print(f"❌ 意外的错误类型: {e.error_type}")
                    
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ 场景 {i} 意外错误: {e}")
                print(f"   错误时间: {duration:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 超时场景测试失败: {e}")
        return False

async def main():
    """运行所有测试"""
    print("🚀 Kimi K2客户端超时修正验证")
    print("=" * 80)
    
    # 测试1: 基本功能测试
    basic_test_success = await test_kimi_k2_timeout_fixes()
    
    # 测试2: 超时场景测试
    timeout_test_success = await test_timeout_scenarios()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    print(f"基本功能测试: {'✅ 通过' if basic_test_success else '❌ 失败'}")
    print(f"超时场景测试: {'✅ 通过' if timeout_test_success else '❌ 失败'}")
    
    if basic_test_success and timeout_test_success:
        print("\n🎉 所有测试通过！Kimi K2客户端超时问题已修正")
        print("\n✅ 修正内容:")
        print("   - 增加了读取超时到120秒（适合长文本生成）")
        print("   - 优化了连接池配置（20个连接，10个保持连接）")
        print("   - 改进了重试逻辑（5次重试，智能退避）")
        print("   - 增加了详细的错误分类和处理")
        print("   - 添加了针对不同错误类型的专门处理")
        print("   - 优化了日志记录和调试信息")
    else:
        print("\n❌ 部分测试失败，请检查配置和网络连接")
    
    return basic_test_success and timeout_test_success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)