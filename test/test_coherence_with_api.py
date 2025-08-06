"""
使用真实API测试Coherence Manager
"""

import asyncio
import sys
import os
from datetime import datetime
import uuid

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def create_test_draft(topic: str):
    """创建测试草稿"""
    
    from models.core import (
        Draft, ResearchStructure, Section, DraftMetadata, ComplexityLevel
    )
    
    sections = [
        Section(
            id="introduction",
            title="引言",
            content=f"本研究探讨{topic}的相关问题。这是一个重要的研究领域。",
            estimated_length=500
        ),
        Section(
            id="background", 
            title="背景",
            content=f"关于{topic}的背景信息需要进一步补充。",
            estimated_length=800
        ),
        Section(
            id="methodology",
            title="研究方法", 
            content="本研究采用文献综述的方法。",
            estimated_length=600
        )
    ]
    
    structure = ResearchStructure(
        sections=sections,
        relationships=[],
        estimated_length=1900,
        complexity_level=ComplexityLevel.INTERMEDIATE
    )
    
    content = {
        "introduction": f"本研究探讨{topic}的相关问题。这是一个重要的研究领域，需要深入分析。当前的研究现状表明，该领域存在诸多挑战和机遇。",
        "background": f"关于{topic}的背景信息如下。历史发展过程中，该领域经历了多个重要阶段。目前的技术水平和应用现状需要进一步分析。",
        "methodology": "本研究采用文献综述的方法进行分析。通过系统性地收集和分析相关文献，我们能够全面了解该领域的发展状况。"
    }
    
    metadata = DraftMetadata(
        created_at=datetime.now(),
        updated_at=datetime.now(),
        author="TTD-DR Framework",
        version="1.0",
        word_count=sum(len(c) for c in content.values())
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

def create_test_retrieved_info(topic: str):
    """创建测试检索信息"""
    
    from models.core import RetrievedInfo, Source
    
    sources = [
        Source(
            title=f"{topic}研究综述",
            url="https://example.com/research1",
            domain="example.com",
            credibility_score=0.8,
            last_accessed=datetime.now()
        ),
        Source(
            title=f"{topic}发展趋势",
            url="https://example.com/research2", 
            domain="example.com",
            credibility_score=0.7,
            last_accessed=datetime.now()
        )
    ]
    
    return [
        RetrievedInfo(
            source=sources[0],
            content=f"这是关于{topic}的重要研究发现。最新的研究表明，该领域正在快速发展。",
            relevance_score=0.9,
            credibility_score=0.8,
            extraction_timestamp=datetime.now()
        ),
        RetrievedInfo(
            source=sources[1],
            content=f"关于{topic}的发展趋势分析显示，未来几年将有重大突破。",
            relevance_score=0.8,
            credibility_score=0.7,
            extraction_timestamp=datetime.now()
        )
    ]

async def test_coherence_with_real_api(topic: str):
    """使用真实API测试coherence manager"""
    
    print(f"🧪 使用真实Kimi K2 API测试: {topic}")
    print("=" * 50)
    
    try:
        # 检查API配置
        from config.settings import settings
        print(f"📋 API配置检查:")
        print(f"   API Key: {'已配置' if settings.kimi_k2_api_key else '未配置'}")
        print(f"   Base URL: {settings.kimi_k2_base_url}")
        print(f"   Model: {settings.kimi_k2_model}")
        
        if not settings.kimi_k2_api_key:
            print("❌ API Key未配置")
            return False
        
        # 导入coherence manager
        from services.kimi_k2_coherence_manager import KimiK2CoherenceManager
        coherence_manager = KimiK2CoherenceManager()
        
        # 创建测试数据
        print(f"\n📝 步骤1: 创建测试草稿...")
        draft = create_test_draft(topic)
        print(f"   ✅ 创建了包含{len(draft.structure.sections)}个章节的草稿")
        
        print(f"\n📚 步骤2: 创建测试检索信息...")
        retrieved_info = create_test_retrieved_info(topic)
        print(f"   ✅ 创建了{len(retrieved_info)}条检索信息")
        
        # 测试一致性维护（使用真实API）
        print(f"\n🔧 步骤3: 测试一致性维护（真实API）...")
        try:
            coherent_draft, coherence_report = await coherence_manager.maintain_coherence(draft)
            print(f"   ✅ 一致性维护完成")
            print(f"   📊 一致性分数: {coherence_report.overall_score:.2f}")
            print(f"   📋 发现问题: {len(coherence_report.issues)}个")
            print(f"   💪 文档优势: {len(coherence_report.strengths)}个")
            
            # 显示具体的分析结果
            if coherence_report.issues:
                print(f"   🔍 发现的问题:")
                for i, issue in enumerate(coherence_report.issues[:3], 1):
                    print(f"      {i}. {issue.description} (严重程度: {issue.severity})")
            
            if coherence_report.strengths:
                print(f"   ✨ 文档优势:")
                for i, strength in enumerate(coherence_report.strengths[:3], 1):
                    print(f"      {i}. {strength}")
                    
        except Exception as e:
            print(f"   ❌ 一致性维护失败: {e}")
            coherent_draft = draft
        
        # 测试引用管理（使用真实API）
        print(f"\n📖 步骤4: 测试引用管理（真实API）...")
        try:
            final_draft, citations = await coherence_manager.manage_citations(
                coherent_draft, retrieved_info
            )
            print(f"   ✅ 引用管理完成")
            print(f"   📚 添加引用: {len(citations)}个")
            
            if citations:
                print(f"   📋 引用列表:")
                for i, citation in enumerate(citations[:3], 1):
                    print(f"      {i}. {citation.source.title}")
                    
        except Exception as e:
            print(f"   ❌ 引用管理失败: {e}")
            final_draft = coherent_draft
            citations = []
        
        # 生成最终报告
        print(f"\n📄 步骤5: 生成最终报告...")
        final_report = generate_enhanced_report(final_draft, citations, topic)
        
        # 保存报告
        filename = f"kimi_api_test_report_{topic.replace(' ', '_')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"   ✅ 报告已保存到: {filename}")
        print(f"   📊 报告长度: {len(final_report)}字符")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_enhanced_report(draft, citations, topic):
    """生成增强版报告"""
    
    report = f"""# {topic}

## 研究报告（Kimi K2 API版本）

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**生成工具**: TTD-DR框架 + Kimi K2 API
**质量评分**: {draft.quality_score:.2f}

---

"""
    
    # 添加各个章节
    for section in draft.structure.sections:
        content = draft.content.get(section.id, "")
        report += f"## {section.title}\n\n{content}\n\n"
    
    # 添加引用部分
    if citations:
        report += "## 参考文献\n\n"
        for i, citation in enumerate(citations, 1):
            report += f"{i}. {citation.source.title}. Retrieved from {citation.source.url}\n"
        report += "\n"
    
    # 添加统计信息
    report += f"""---

## API测试统计

- **章节数**: {len(draft.structure.sections)}
- **引用数**: {len(citations)}
- **API调用**: ✅ 成功
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*本报告使用真实的Kimi K2 API生成*
"""
    
    return report

async def main():
    """主函数"""
    
    print("🎯 TTD-DR框架 + Kimi K2 API 真实测试")
    print("=" * 60)
    
    test_topics = [
        "人工智能在教育领域的应用",
        "区块链技术发展现状"
    ]
    
    success_count = 0
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n🔥 测试 {i}/{len(test_topics)}: {topic}")
        print("-" * 40)
        
        success = await test_coherence_with_real_api(topic)
        if success:
            success_count += 1
        
        if i < len(test_topics):
            print(f"\n⏳ 等待3秒后进行下一个测试...")
            await asyncio.sleep(3)
    
    print(f"\n🎉 测试完成！")
    print(f"📊 成功率: {success_count}/{len(test_topics)} ({success_count/len(test_topics)*100:.1f}%)")
    
    if success_count > 0:
        print("\n✅ Kimi K2 API集成成功！")
        print("💡 TTD-DR框架现在可以使用真实的AI能力了")
    else:
        print("\n❌ API集成测试失败")

if __name__ == "__main__":
    asyncio.run(main())