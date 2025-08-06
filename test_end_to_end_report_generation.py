"""
端到端测试：给定题目生成完整研究报告
测试已实现的TTD-DR框架功能
"""

import asyncio
import sys
import os
from datetime import datetime
import uuid

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from models.core import (
        TTDRState, ResearchRequirements, ResearchDomain, ComplexityLevel,
        Draft, ResearchStructure, Section, DraftMetadata
    )
    from services.kimi_k2_coherence_manager import KimiK2CoherenceManager
    print("✅ 成功导入核心模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("尝试简化版本...")
    
    # 简化版本的数据结构
    class SimpleSection:
        def __init__(self, id, title, content="", estimated_length=0):
            self.id = id
            self.title = title
            self.content = content
            self.estimated_length = estimated_length
    
    class SimpleStructure:
        def __init__(self, sections):
            self.sections = sections
    
    class SimpleDraft:
        def __init__(self, id, topic, structure, content):
            self.id = id
            self.topic = topic
            self.structure = structure
            self.content = content
            self.quality_score = 0.6
            self.iteration = 0


def create_sample_draft(topic: str) -> Draft:
    """创建一个示例研究草稿"""
    
    # 创建研究结构
    sections = [
        Section(
            id="introduction",
            title="引言",
            content=f"本研究探讨{topic}的相关问题。这是一个重要的研究领域，需要深入分析。",
            estimated_length=500
        ),
        Section(
            id="background",
            title="背景",
            content=f"关于{topic}的背景信息还需要进一步补充。目前的研究现状如下。",
            estimated_length=800
        ),
        Section(
            id="methodology",
            title="研究方法",
            content="本研究采用文献综述的方法进行分析。",
            estimated_length=600
        ),
        Section(
            id="results",
            title="研究结果",
            content="通过分析发现了以下几个重要发现。",
            estimated_length=1000
        ),
        Section(
            id="conclusion",
            title="结论",
            content="基于以上分析，我们得出以下结论。",
            estimated_length=400
        )
    ]
    
    structure = ResearchStructure(
        sections=sections,
        relationships=[],
        estimated_length=3300,
        complexity_level=ComplexityLevel.INTERMEDIATE,
        domain=ResearchDomain.TECHNOLOGY
    )
    
    content = {
        section.id: section.content for section in sections
    }
    
    metadata = DraftMetadata(
        created_at=datetime.now(),
        updated_at=datetime.now(),
        author="TTD-DR Framework",
        version="1.0",
        word_count=sum(len(content) for content in content.values())
    )
    
    return Draft(
        id=str(uuid.uuid4()),
        topic=topic,
        structure=structure,
        content=content,
        metadata=metadata,
        quality_score=0.6,
        iteration=0
    )


async def generate_research_report(topic: str) -> str:
    """
    给定题目生成研究报告的完整流程
    
    Args:
        topic: 研究题目
        
    Returns:
        生成的研究报告文本
    """
    
    print(f"🚀 开始生成研究报告：{topic}")
    print("=" * 60)
    
    # 步骤1：创建初始草稿
    print("📝 步骤1：创建初始研究草稿...")
    draft = create_sample_draft(topic)
    print(f"   ✅ 创建了包含{len(draft.structure.sections)}个章节的初始草稿")
    print(f"   📊 初始质量分数：{draft.quality_score:.2f}")
    
    # 步骤2：分析信息缺口
    print("\n🔍 步骤2：分析信息缺口...")
    try:
        gap_analyzer = KimiK2InformationGapAnalyzer()
        gaps = await gap_analyzer.identify_gaps(draft)
        print(f"   ✅ 识别出{len(gaps)}个信息缺口")
        
        for i, gap in enumerate(gaps[:3], 1):  # 显示前3个
            print(f"   📋 缺口{i}: {gap.description} (优先级: {gap.priority})")
            
    except Exception as e:
        print(f"   ⚠️  信息缺口分析失败，使用fallback: {e}")
        gaps = []
    
    # 步骤3：生成搜索查询
    print("\n🔎 步骤3：生成搜索查询...")
    try:
        query_generator = KimiK2SearchQueryGenerator()
        all_queries = []
        
        for gap in gaps[:2]:  # 处理前2个缺口
            queries = await query_generator.generate_queries(gap)
            all_queries.extend(queries)
            print(f"   ✅ 为缺口'{gap.description[:30]}...'生成了{len(queries)}个查询")
            
        print(f"   📊 总共生成{len(all_queries)}个搜索查询")
        
    except Exception as e:
        print(f"   ⚠️  搜索查询生成失败，使用fallback: {e}")
        all_queries = []
    
    # 步骤4：检索信息（模拟）
    print("\n🌐 步骤4：检索外部信息...")
    try:
        retrieval_engine = DynamicRetrievalEngine()
        retrieved_info = []
        
        # 由于没有真实的Google Search API，我们模拟一些检索结果
        from models.core import RetrievedInfo, Source
        
        mock_sources = [
            Source(
                title=f"{topic}相关研究综述",
                url="https://example.com/research1",
                domain="example.com",
                credibility_score=0.8,
                last_accessed=datetime.now()
            ),
            Source(
                title=f"{topic}的最新发展趋势",
                url="https://example.com/research2", 
                domain="example.com",
                credibility_score=0.7,
                last_accessed=datetime.now()
            )
        ]
        
        for source in mock_sources:
            retrieved_info.append(RetrievedInfo(
                source=source,
                content=f"这是关于{topic}的重要研究发现。最新的研究表明，该领域正在快速发展，具有重要的理论和实践意义。",
                relevance_score=0.8,
                credibility_score=source.credibility_score,
                extraction_timestamp=datetime.now()
            ))
            
        print(f"   ✅ 检索到{len(retrieved_info)}条相关信息")
        
    except Exception as e:
        print(f"   ⚠️  信息检索失败: {e}")
        retrieved_info = []
    
    # 步骤5：信息整合
    print("\n🔗 步骤5：整合检索信息...")
    try:
        integrator = KimiK2InformationIntegrator()
        
        if retrieved_info:
            integrated_draft = await integrator.integrate_information(
                draft, retrieved_info, gaps
            )
            print(f"   ✅ 成功整合{len(retrieved_info)}条信息到草稿中")
            draft = integrated_draft
        else:
            print("   ℹ️  没有新信息需要整合")
            
    except Exception as e:
        print(f"   ⚠️  信息整合失败，使用原始草稿: {e}")
    
    # 步骤6：一致性维护和引用管理
    print("\n📚 步骤6：维护一致性和管理引用...")
    try:
        coherence_manager = KimiK2CoherenceManager()
        
        # 维护一致性
        coherent_draft, coherence_report = await coherence_manager.maintain_coherence(draft)
        print(f"   ✅ 一致性分析完成，总分：{coherence_report.overall_score:.2f}")
        print(f"   📋 发现{len(coherence_report.issues)}个一致性问题")
        
        # 管理引用
        if retrieved_info:
            final_draft, citations = await coherence_manager.manage_citations(
                coherent_draft, retrieved_info
            )
            print(f"   ✅ 添加了{len(citations)}个引用")
        else:
            final_draft = coherent_draft
            citations = []
            
        draft = final_draft
        
    except Exception as e:
        print(f"   ⚠️  一致性维护失败，使用当前草稿: {e}")
    
    # 步骤7：生成最终报告
    print("\n📄 步骤7：生成最终研究报告...")
    
    # 编译最终报告
    final_report = f"""# {draft.topic}

## 研究报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**报告版本**: {draft.metadata.version}
**质量评分**: {draft.quality_score:.2f}

---

"""
    
    # 添加各个章节内容
    for section in draft.structure.sections:
        content = draft.content.get(section.id, "")
        final_report += f"## {section.title}\n\n{content}\n\n"
    
    # 添加统计信息
    final_report += f"""---

## 报告统计

- **总字数**: {draft.metadata.word_count}
- **章节数**: {len(draft.structure.sections)}
- **处理的信息缺口**: {len(gaps)}
- **整合的外部信息**: {len(retrieved_info)}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*本报告由TTD-DR框架自动生成*
"""
    
    print("   ✅ 最终报告生成完成！")
    print(f"   📊 报告总长度：{len(final_report)}字符")
    
    return final_report


async def main():
    """主函数：演示完整的报告生成流程"""
    
    # 测试题目
    test_topics = [
        "人工智能在教育领域的应用",
        "区块链技术的发展现状与前景",
        "可持续发展与绿色能源"
    ]
    
    print("🎯 TTD-DR框架端到端报告生成测试")
    print("=" * 60)
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n🔥 测试 {i}/{len(test_topics)}: {topic}")
        print("-" * 40)
        
        try:
            # 生成报告
            report = await generate_research_report(topic)
            
            # 保存报告
            filename = f"generated_report_{i}_{topic.replace(' ', '_')}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n✅ 报告已保存到: {filename}")
            print(f"📄 报告预览（前200字符）:")
            print("-" * 30)
            print(report[:200] + "...")
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(test_topics):
            print(f"\n⏳ 等待3秒后处理下一个题目...")
            await asyncio.sleep(3)
    
    print(f"\n🎉 所有测试完成！")


if __name__ == "__main__":
    asyncio.run(main())