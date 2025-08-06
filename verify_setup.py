#!/usr/bin/env python3
"""
验证 TTD-DR 框架 Draft Generator 设置和功能
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# 添加 backend 到路径
sys.path.append('backend')

def print_header(title):
    """打印测试标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """打印测试步骤"""
    print(f"\n{step}. {description}")

def print_success(message):
    """打印成功消息"""
    print(f"   [OK] {message}")

def print_error(message):
    """打印错误消息"""
    print(f"   [ERROR] {message}")

def test_imports():
    """测试所有必要的导入"""
    print_step(1, "测试模块导入")
    
    try:
        from models.core import (
            TTDRState, ResearchRequirements, ResearchDomain, 
            ComplexityLevel, Draft, ResearchStructure, Section
        )
        print_success("核心模型导入成功")
        
        from services.kimi_k2_client import KimiK2Client, KimiK2Error
        print_success("Kimi K2 客户端导入成功")
        
        from workflow.draft_generator import (
            KimiK2DraftGenerator, draft_generator_node, DraftGenerationError
        )
        print_success("Draft Generator 导入成功")
        
        return True
    except Exception as e:
        print_error(f"导入失败: {e}")
        traceback.print_exc()
        return False

def test_fallback_structure():
    """测试 fallback 结构生成"""
    print_step(2, "测试 Fallback 结构生成")
    
    try:
        from workflow.draft_generator import KimiK2DraftGenerator
        from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
        
        generator = KimiK2DraftGenerator()
        
        # 测试不同域的结构生成
        domains = [
            ResearchDomain.TECHNOLOGY,
            ResearchDomain.SCIENCE,
            ResearchDomain.BUSINESS,
            ResearchDomain.ACADEMIC,
            ResearchDomain.GENERAL
        ]
        
        for domain in domains:
            requirements = ResearchRequirements(
                domain=domain,
                complexity_level=ComplexityLevel.INTERMEDIATE
            )
            
            structure = generator._create_fallback_structure("测试主题", requirements)
            
            if len(structure.sections) > 0:
                print_success(f"{domain.value} 域结构生成成功 ({len(structure.sections)} 个章节)")
            else:
                print_error(f"{domain.value} 域结构生成失败")
                return False
        
        return True
    except Exception as e:
        print_error(f"Fallback 结构生成失败: {e}")
        traceback.print_exc()
        return False

def test_placeholder_content():
    """测试占位符内容生成"""
    print_step(3, "测试占位符内容生成")
    
    try:
        from workflow.draft_generator import KimiK2DraftGenerator
        from models.core import Section
        
        generator = KimiK2DraftGenerator()
        
        # 测试不同类型的章节
        sections = [
            Section(id="introduction", title="Introduction", estimated_length=500),
            Section(id="background", title="Background", estimated_length=800),
            Section(id="methodology", title="Methodology", estimated_length=600),
            Section(id="conclusion", title="Conclusion", estimated_length=400),
            Section(id="custom_section", title="Custom Section", estimated_length=300)
        ]
        
        for section in sections:
            content = generator._create_placeholder_content(section, "人工智能")
            
            if len(content) > 100:  # 确保内容足够长
                print_success(f"{section.title} 占位符内容生成成功 ({len(content)} 字符)")
            else:
                print_error(f"{section.title} 占位符内容太短")
                return False
        
        return True
    except Exception as e:
        print_error(f"占位符内容生成失败: {e}")
        traceback.print_exc()
        return False

async def test_draft_generation():
    """测试完整的 draft 生成"""
    print_step(4, "测试完整 Draft 生成 (使用 Fallback)")
    
    try:
        from workflow.draft_generator import draft_generator_node
        from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
        
        # 临时禁用 API key 以强制使用 fallback
        original_api_key = os.environ.get('KIMI_K2_API_KEY')
        os.environ['KIMI_K2_API_KEY'] = ''
        
        state = {
            'topic': '人工智能在医疗保健中的应用',
            'requirements': ResearchRequirements(
                domain=ResearchDomain.TECHNOLOGY,
                complexity_level=ComplexityLevel.ADVANCED,
                max_iterations=3,
                quality_threshold=0.8
            ),
            'current_draft': None,
            'information_gaps': [],
            'retrieved_info': [],
            'iteration_count': 0,
            'quality_metrics': None,
            'evolution_history': [],
            'final_report': None,
            'error_log': []
        }
        
        # 设置超时以防止长时间等待
        result = await asyncio.wait_for(draft_generator_node(state), timeout=30.0)
        
        # 恢复原始 API key
        if original_api_key:
            os.environ['KIMI_K2_API_KEY'] = original_api_key
        
        # 验证结果
        if result['current_draft'] is not None:
            draft = result['current_draft']
            print_success(f"Draft 生成成功")
            print_success(f"主题: {draft.topic}")
            print_success(f"章节数量: {len(draft.structure.sections)}")
            print_success(f"内容章节: {len(draft.content)}")
            print_success(f"域: {draft.structure.domain}")
            print_success(f"复杂度: {draft.structure.complexity_level}")
            print_success(f"预估长度: {draft.structure.estimated_length} 词")
            print_success(f"实际字数: {draft.metadata.word_count} 词")
            
            # 验证所有章节都有内容
            missing_content = []
            for section in draft.structure.sections:
                if section.id not in draft.content or len(draft.content[section.id]) < 50:
                    missing_content.append(section.id)
            
            if missing_content:
                print_error(f"以下章节缺少内容: {missing_content}")
                return False
            else:
                print_success("所有章节都有内容")
            
            return True
        else:
            print_error("Draft 生成失败 - 返回 None")
            return False
            
    except asyncio.TimeoutError:
        print_error("Draft 生成超时 (30秒)")
        return False
    except Exception as e:
        print_error(f"Draft 生成失败: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print_step(5, "测试错误处理")
    
    try:
        from workflow.draft_generator import KimiK2DraftGenerator, DraftGenerationError
        from services.kimi_k2_client import KimiK2Error
        
        # 测试 DraftGenerationError
        try:
            raise DraftGenerationError("测试错误")
        except DraftGenerationError as e:
            print_success("DraftGenerationError 正常工作")
        
        # 测试 KimiK2Error
        try:
            raise KimiK2Error("测试 API 错误", 500, "server")
        except KimiK2Error as e:
            if e.status_code == 500 and e.error_type == "server":
                print_success("KimiK2Error 正常工作")
            else:
                print_error("KimiK2Error 属性不正确")
                return False
        
        return True
    except Exception as e:
        print_error(f"错误处理测试失败: {e}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """运行所有测试"""
    print_header("TTD-DR Framework Draft Generator 验证")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("模块导入", test_imports),
        ("Fallback 结构生成", test_fallback_structure),
        ("占位符内容生成", test_placeholder_content),
        ("完整 Draft 生成", test_draft_generation),
        ("错误处理", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"{test_name} 测试出现异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print_header("测试结果总结")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: 通过")
            passed += 1
        else:
            print_error(f"{test_name}: 失败")
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print_success("所有测试通过！TTD-DR Draft Generator 设置正确。")
        return True
    else:
        print_error(f"有 {failed} 个测试失败。请检查上述错误信息。")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试运行失败: {e}")
        traceback.print_exc()
        sys.exit(1)