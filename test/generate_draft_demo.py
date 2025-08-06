#!/usr/bin/env python3
"""
Simple draft generation demo for TTD-DR Framework

Usage examples:
    python generate_draft_demo.py "人工智能在教育领域的应用"
    python generate_draft_demo.py "区块链技术发展趋势"
    python generate_draft_demo.py "气候变化对农业的影响"

Requirements:
    - KIMI_K2_API_KEY environment variable must be set
    - Backend dependencies from requirements.txt must be installed
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加 backend 到路径
sys.path.append('backend')

from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from workflow.draft_generator import KimiK2DraftGenerator

async def generate_draft_report(topic: str):
    """生成指定主题的草稿报告"""
    print(f"正在生成主题: {topic}")
    
    # 初始化草稿生成器（内部会自动初始化Kimi客户端）
    draft_generator = KimiK2DraftGenerator()
    
    # 创建研究需求
    requirements = ResearchRequirements(
        domain=ResearchDomain.TECHNOLOGY,
        complexity_level=ComplexityLevel.ADVANCED,
        max_iterations=3,
        quality_threshold=0.8
    )
    
    try:
        # 生成草稿
        print("开始生成草稿...")
        draft = await draft_generator.generate_initial_draft(topic, requirements)
        
        if draft and draft.structure.sections:
            print(f"\n✅ 草稿生成成功!")
            print(f"📊 包含 {len(draft.structure.sections)} 个章节")
            
            # 保存草稿到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"draft_{safe_topic.replace(' ', '_')}_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {topic}\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"质量评分: {draft.quality_score:.2f}\n")
                f.write(f"字数统计: {draft.metadata.word_count}\n\n")
                
                for section in draft.structure.sections:
                    f.write(f"## {section.title}\n\n")
                    section_content = draft.content.get(section.id, f"*[内容待补充: {section.title}]*")
                    f.write(f"{section_content}\n\n")
            
            print(f"💾 草稿已保存到: {filename}")
            
            # 打印摘要
            print("\n📋 草稿摘要:")
            total_chars = 0
            for i, section in enumerate(draft.structure.sections, 1):
                section_content = draft.content.get(section.id, "")
                section_chars = len(section_content)
                total_chars += section_chars
                print(f"{i}. {section.title} - {section_chars} 字符")
            print(f"总计: {total_chars} 字符")
            
            return draft
        else:
            print("❌ 草稿生成失败")
            return None
            
    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    if len(sys.argv) < 2:
        topic = "人工智能在教育领域的应用"
        print(f"未提供主题，使用默认主题: {topic}")
    else:
        topic = sys.argv[1]
    
    # 运行异步函数
    draft = asyncio.run(generate_draft_report(topic))
    
    if draft:
        print("\n🎉 草稿生成完成!")
    else:
        print("\n💥 草稿生成失败，请检查配置和API密钥")

if __name__ == "__main__":
    main()