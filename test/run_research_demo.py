#!/usr/bin/env python3
"""
三阶段自适应研究系统 - 实际运行演示
给定题目："Python异步编程最佳实践2025"
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient
from backend.services.kimi_k2_client import KimiK2Client

class ThreeStageResearchSystem:
    """三阶段自适应研究系统"""
    
    def __init__(self):
        self.google_client = GoogleSearchClient()
        self.kimi_client = KimiK2Client()
        self.research_log = []
    
    def log_stage(self, stage: str, data: dict):
        """记录阶段信息"""
        self.research_log.append({
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    
    async def stage1_research_planning(self, topic: str) -> dict:
        """阶段1：研究计划生成"""
        print("【阶段1】研究计划生成...")
        
        prompt = f"""
        作为研究专家，请为"{topic}"生成详细的研究计划：
        
        要求：
        1. 研究目标和范围
        2. 结构化大纲（包含主要章节）
        3. 每个章节的核心要点
        4. 需要重点调研的技术点
        5. 预期信息缺口
        
        请以JSON格式返回完整的研究计划。
        """
        
        try:
            response = await self.kimi_client.generate_text(prompt)
            plan = {
                "topic": topic,
                "research_plan": response,
                "structure": [
                    "1. 基础概念与原理",
                    "2. 当前技术现状分析", 
                    "3. 最佳实践指南",
                    "4. 实际应用案例",
                    "5. 未来发展趋势"
                ],
                "expected_gaps": [
                    "最新框架对比",
                    "性能基准测试",
                    "实际部署经验",
                    "错误处理模式"
                ]
            }
            
            self.log_stage("stage1_planning", plan)
            print("✅ 研究计划生成完成")
            return plan
            
        except Exception as e:
            print(f"阶段1失败: {e}")
            return {"error": str(e)}
    
    async def stage2_search_and_synthesize(self, plan: dict) -> dict:
        """阶段2：迭代搜索和合成"""
        print("\n【阶段2】迭代搜索和合成...")
        
        # 2a: 生成搜索问题
        search_queries = plan.get("expected_gaps", [])
        search_queries.extend([
            "Python asyncio best practices 2025",
            "Python async await performance optimization",
            "Python concurrent programming patterns"
        ])
        
        # 2b: RAG系统执行
        all_results = []
        
        for i, query in enumerate(search_queries[:5], 1):
            print(f"  搜索 {i}/{len(search_queries[:5])}: {query}")
            
            try:
                response = await self.google_client.search(query, num_results=3)
                
                for item in response.items:
                    all_results.append({
                        "title": item.title,
                        "url": item.link,
                        "snippet": item.snippet,
                        "query": query
                    })
                    
            except Exception as e:
                print(f"  搜索失败: {e}")
        
        # 合成信息
        synthesis = {
            "search_queries": search_queries,
            "sources_collected": len(all_results),
            "synthesized_content": await self._synthesize_information(all_results),
            "raw_sources": all_results
        }
        
        self.log_stage("stage2_search_synthesize", synthesis)
        print(f"✅ 收集 {len(all_results)} 个信息源，完成合成")
        return synthesis
    
    async def _synthesize_information(self, sources: list) -> str:
        """信息合成"""
        if not sources:
            return "无可用信息源"
        
        prompt = f"""
        基于以下信息源，为"Python异步编程最佳实践2025"生成综合分析：
        
        信息源：
        {chr(10).join([f"- {s['title']}: {s['snippet']}" for s in sources[:5]])}
        
        请生成：
        1. 主要发现总结
        2. 最佳实践建议
        3. 技术趋势分析
        """
        
        try:
            return await self.kimi_client.generate_text(prompt, max_tokens=1000)
        except Exception as e:
            return f"合成失败: {e}"
    
    async def stage3_final_report(self, plan: dict, synthesis: dict) -> dict:
        """阶段3：最终报告综合"""
        print("\n【阶段3】最终报告生成...")
        
        prompt = f"""
        基于研究计划和合成信息，生成"Python异步编程最佳实践2025"的完整研究报告：
        
        研究计划：{plan.get('structure', [])}
        合成信息：{synthesis.get('synthesized_content', '')[:500]}...
        
        请生成包含以下部分的专业报告：
        1. 执行摘要
        2. 详细研究报告
        3. 实际应用建议
        4. 参考资料列表
        """
        
        try:
            final_report = await self.kimi_client.generate_text(prompt, max_tokens=2000)
            
            report = {
                "final_report": final_report,
                "metadata": {
                    "total_sources": synthesis.get("sources_collected", 0),
                    "research_quality": "high",
                    "completion_time": datetime.now().isoformat()
                }
            }
            
            self.log_stage("stage3_final_report", report)
            print("✅ 最终报告生成完成")
            return report
            
        except Exception as e:
            return {"error": str(e)}
    
    async def run_adaptive_research(self, topic: str) -> dict:
        """运行完整的自适应研究流程"""
        
        print("=" * 70)
        print("TTD-DR三阶段自适应研究系统")
        print("=" * 70)
        print(f"研究主题: {topic}")
        print()
        
        # 阶段1: 研究计划
        plan = await self.stage1_research_planning(topic)
        
        # 阶段2: 搜索和合成
        synthesis = await self.stage2_search_and_synthesize(plan)
        
        # 阶段3: 最终报告
        report = await self.stage3_final_report(plan, synthesis)
        
        # 完整结果
        result = {
            "topic": topic,
            "stages": {
                "stage1": plan,
                "stage2": synthesis,
                "stage3": report
            },
            "research_log": self.research_log,
            "summary": {
                "total_sources": synthesis.get("sources_collected", 0),
                "research_time": datetime.now().isoformat(),
                "quality_assessment": "completed"
            }
        }
        
        # 保存结果
        filename = f"research_report_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("研究完成！")
        print(f"报告已保存: {filename}")
        print("=" * 70)
        
        return result

async def main():
    """主函数"""
    system = ThreeStageResearchSystem()
    topic = "Python异步编程最佳实践2025"
    
    result = await system.run_adaptive_research(topic)
    
    # 显示摘要
    print(f"\n研究摘要:")
    print(f"- 主题: {result['topic']}")
    print(f"- 信息源: {result['summary']['total_sources']}")
    print(f"- 完成时间: {result['summary']['research_time']}")

if __name__ == "__main__":
    asyncio.run(main())