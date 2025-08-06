#!/usr/bin/env python3
"""
TTD-DR三阶段自适应研究系统完整演示

演示完整的8节点工作流：
1. 研究计划生成 (draft_generator)
2. 缺口分析 (gap_analyzer) → 生成搜索问题
3. 信息检索 (retrieval_engine) → RAG系统
4. 信息集成 (information_integrator) → 合成答案
5. 质量评估 (quality_assessor)
6. 自进化增强 (self_evolution_enhancer)
7. 最终报告生成 (report_synthesizer)
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.workflow.graph import create_ttdr_workflow
from backend.models.core import (
    ResearchRequirements, 
    ResearchDomain, 
    ComplexityLevel,
    TTDRState
)

class ResearchSystemDemo:
    """三阶段自适应研究系统演示"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.workflow = create_ttdr_workflow()
        self.state_log = []
        
    def log_state(self, stage: str, data: Dict):
        """记录状态"""
        self.state_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "data": data
        })
    
    async def run_full_research(self) -> Dict[str, Any]:
        """运行完整的三阶段研究流程"""
        
        print("=" * 80)
        print("TTD-DR三阶段自适应研究系统演示")
        print("=" * 80)
        print(f"研究主题: {self.topic}")
        print()
        
        # 阶段1：研究计划生成
        print("【阶段1】研究计划生成...")
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.75,
            target_audience="技术开发人员",
            desired_length="中等长度",
            specific_requirements=[
                "包含最新技术趋势",
                "提供实际应用案例",
                "包含技术实现细节"
            ]
        )
        
        # 初始状态
        initial_state = TTDRState(
            topic=self.topic,
            requirements=requirements,
            current_draft="",
            information_gaps=[],
            retrieved_info=[],
            iteration_count=0,
            quality_score=0.0,
            error_log=[]
        )
        
        print("✅ 研究计划生成完成")
        self.log_state("stage1_planning", {
            "topic": self.topic,
            "requirements": requirements.model_dump()
        })
        
        # 运行完整工作流
        print("\n【阶段2】启动迭代搜索和合成流程...")
        final_state = await self.workflow.ainvoke(initial_state.model_dump())
        
        # 阶段3：最终报告生成
        print("\n【阶段3】最终报告综合...")
        
        return {
            "final_report": final_state.get("final_report", ""),
            "quality_score": final_state.get("quality_score", 0.0),
            "iterations": final_state.get("iteration_count", 0),
            "research_log": self.state_log,
            "metadata": {
                "total_sources": len(final_state.get("retrieved_info", [])),
                "gaps_addressed": len(final_state.get("information_gaps", [])),
                "evolution_applied": final_state.get("evolution_metrics", {})
            }
        }
    
    def save_demo_report(self, result: Dict[str, Any]):
        """保存演示报告"""
        filename = f"demo_research_{self.topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 演示报告已保存: {filename}")

async def run_stage_demo():
    """分阶段演示"""
    
    # 设置研究主题
    topic = "Python异步编程最佳实践2025"
    
    demo = ResearchSystemDemo(topic)
    result = await demo.run_full_research()
    
    # 保存演示结果
    demo.save_demo_report(result)
    
    return result

async def quick_demo():
    """快速演示"""
    print("快速三阶段演示")
    print("-" * 50)
    
    # 使用简单搜索演示阶段2
    from backend.services.google_search_client import GoogleSearchClient
    
    client = GoogleSearchClient()
    
    # 阶段1：研究计划
    print("【阶段1】研究计划: Python异步编程最佳实践")
    
    # 阶段2：搜索问题生成 + 信息检索 + 合成
    print("\n【阶段2】搜索和合成...")
    
    # 模拟搜索问题
    search_queries = [
        "Python asyncio best practices 2025",
        "Python async await patterns",
        "Python concurrent programming tutorial"
    ]
    
    all_results = []
    for query in search_queries:
        print(f"  搜索: {query}")
        response = await client.search(query, num_results=3)
        
        for item in response.items:
            all_results.append({
                "title": item.title,
                "url": item.link,
                "snippet": item.snippet
            })
    
    # 阶段3：报告综合
    print(f"\n【阶段3】综合报告...")
    print(f"  收集到 {len(all_results)} 个信息源")
    print("  生成最终报告...")
    
    return {
        "topic": "Python异步编程最佳实践2025",
        "sources_collected": len(all_results),
        "search_queries": search_queries,
        "sample_results": all_results[:3]
    }

if __name__ == "__main__":
    # 直接运行快速演示
    print("运行三阶段自适应研究系统演示...")
    result = asyncio.run(quick_demo())
    
    print("\n演示完成！")
    print(f"主题: {result.get('topic', 'N/A')}")
    print(f"信息源数量: {result.get('sources_collected', len(result.get('sample_results', [])))}")