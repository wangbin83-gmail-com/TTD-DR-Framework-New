#!/usr/bin/env python3
"""
TTD-DR完整工作流演示 - 生成真正的最终报告
使用完整的8节点工作流生成专业研究报告
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.workflow.graph import create_ttdr_workflow
from backend.models.core import (
    ResearchRequirements, 
    ResearchDomain, 
    ComplexityLevel,
    TTDRState
)

async def generate_complete_report():
    """运行完整工作流生成最终报告"""
    
    topic = "Python异步编程最佳实践2025"
    
    print("=" * 80)
    print("TTD-DR完整工作流 - 生成最终研究报告")
    print("=" * 80)
    print(f"研究主题: {topic}")
    print()
    
    # 创建完整工作流
    workflow = create_ttdr_workflow()
    
    # 设置研究要求
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.ADVANCED,
        max_iterations=3,
        quality_threshold=0.8,
        target_audience="高级Python开发者",
        desired_length="详细报告",
        specific_requirements=[
            "包含2025年最新实践",
            "提供完整代码示例",
            "性能对比分析",
            "实际部署案例"
        ]
    )
    
    # 初始状态
    initial_state = {
        "topic": topic,
        "requirements": requirements.model_dump(),
        "current_draft": "",
        "information_gaps": [],
        "retrieved_info": [],
        "iteration_count": 0,
        "quality_score": 0.0,
        "error_log": []
    }
    
    print("【启动完整工作流】")
    print("1. draft_generator → 生成初始研究结构")
    print("2. gap_analyzer → 识别信息缺口")
    print("3. retrieval_engine → 搜索相关信息")
    print("4. information_integrator → 集成信息")
    print("5. quality_assessor → 评估质量")
    print("6. self_evolution_enhancer → 优化内容")
    print("7. report_synthesizer → 生成最终报告")
    print()
    
    try:
        # 运行完整工作流
        final_state = await workflow.ainvoke(initial_state)
        
        # 提取最终报告
        final_report = final_state.get("final_report", "")
        
        if final_report:
            print("【最终报告生成成功】")
            print("=" * 50)
            print(final_report[:1000] + "..." if len(final_report) > 1000 else final_report)
            print("=" * 50)
            
            # 保存完整报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"final_report_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {topic} - 研究报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## 研究摘要\n")
                f.write(f"- 信息源数量: {len(final_state.get('retrieved_info', []))}\n")
                f.write(f"- 迭代次数: {final_state.get('iteration_count', 0)}\n")
                f.write(f"- 质量评分: {final_state.get('quality_score', 0)}\n\n")
                f.write(final_report)
            
            print(f"\n✅ 完整报告已保存: {filename}")
            return final_report
        else:
            print("❌ 未生成最终报告")
            print("状态信息:", {
                "quality_score": final_state.get("quality_score"),
                "iteration_count": final_state.get("iteration_count"),
                "has_final_report": bool(final_report)
            })
            return None
            
    except Exception as e:
        print(f"❌ 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def simple_workflow_demo():
    """简化工作流演示 - 模拟完整报告生成"""
    
    print("\n" + "=" * 80)
    print("简化工作流 - 生成模拟最终报告")
    print("=" * 80)
    
    topic = "Python异步编程最佳实践2025"
    
    # 使用Google搜索收集信息
    from backend.services.google_search_client import GoogleSearchClient
    from backend.services.kimi_k2_client import KimiK2Client
    
    google_client = GoogleSearchClient()
    kimi_client = KimiK2Client()
    
    # 阶段1: 研究计划
    print("【阶段1】研究计划生成...")
    research_plan = {
        "executive_summary": "深入分析Python异步编程在2025年的最佳实践",
        "key_areas": [
            "asyncio核心机制",
            "性能优化技巧", 
            "错误处理模式",
            "实际应用案例",
            "新技术趋势"
        ]
    }
    print("✅ 研究计划完成")
    
    # 阶段2: 信息收集和合成
    print("\n【阶段2】信息收集和合成...")
    
    search_queries = [
        "Python asyncio best practices 2025",
        "Python async performance optimization",
        "Python concurrent programming patterns"
    ]
    
    collected_info = []
    for query in search_queries:
        response = await google_client.search(query, num_results=3)
        collected_info.extend([
            {
                "title": item.title,
                "url": item.link,
                "content": item.snippet
            }
            for item in response.items
        ])
    
    print(f"✅ 收集 {len(collected_info)} 个信息源")
    
    # 阶段3: 最终报告生成
    print("\n【阶段3】最终报告生成...")
    
    # 使用Kimi生成专业报告
    report_prompt = f"""
    基于以下关于"{topic}"的研究信息，生成一份专业的技术研究报告：
    
    收集到的信息源：
    {chr(10).join([f"[{i+1}] {info['title']}: {info['content'][:100]}..." for i, info in enumerate(collected_info[:5])])}
    
    请生成包含以下部分的专业报告：
    
    1. 执行摘要
    2. 技术背景和历史
    3. 2025年最新最佳实践
    4. 性能优化技巧
    5. 实际应用案例
    6. 常见错误和解决方案
    7. 未来发展趋势
    8. 结论和建议
    
    格式要求：
    - 使用专业的技术报告格式
    - 包含具体的代码示例
    - 提供实际可操作的指导
    - 字数约2000-3000字
    """
    
    try:
        final_report = await kimi_client.generate_text(report_prompt, max_tokens=2500)
        
        # 保存最终报告
        filename = f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {topic} - 完整研究报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**信息源数量**: {len(collected_info)}\n")
            f.write(f"**研究方法**: TTD-DR三阶段自适应工作流\n\n")
            f.write("---\n\n")
            f.write(final_report)
        
        print("【最终报告预览】")
        print("-" * 50)
        print(final_report[:800] + "..." if len(final_report) > 800 else final_report)
        print("-" * 50)
        
        print(f"\n✅ 完整报告已保存: {filename}")
        print(f"📊 报告长度: {len(final_report)} 字符")
        print(f"🔗 信息源: {len(collected_info)} 个")
        
        return final_report
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        return None

if __name__ == "__main__":
    # 运行简化演示
    asyncio.run(simple_workflow_demo())