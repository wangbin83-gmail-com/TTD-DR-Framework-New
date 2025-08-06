#!/usr/bin/env python3
"""
Simple draft generation script for TTD-DR Framework
快速生成研究草稿示例
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加 backend 到路径
sys.path.append('backend')

from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from workflow.draft_generator import KimiK2DraftGenerator

async def quick_draft(topic: str):
    """快速生成草稿"""
    print(f"🚀 生成主题: {topic}")
    
    try:
        generator = KimiK2DraftGenerator()
        
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=2,
            quality_threshold=0.7
        )
        
        print("📋 正在生成草稿...")
        draft = await generator.generate_initial_draft(topic, requirements)
        
        if draft:
            print("草稿生成成功!")
            print(f"章节数: {len(draft.structure.sections)}")
            print(f"质量评分: {draft.quality_score:.2f}")
            print(f"总字数: {draft.metadata.word_count}")
            
            # 显示摘要
            print("\n草稿结构:")
            for section in draft.structure.sections:
                content = draft.content.get(section.id, "")[:100] + "..." if len(draft.content.get(section.id, "")) > 100 else draft.content.get(section.id, "")
                print(f"   - {section.title}: {len(draft.content.get(section.id, ''))} 字符")
                
            return draft
        else:
            print("草稿生成失败")
            return None
            
    except Exception as e:
        print(f"错误: {e}")
        return None

def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "人工智能在教育中的应用"
    
    print("TTD-DR 草稿生成器")
    print("=" * 30)
    
    draft = asyncio.run(quick_draft(topic))
    
    if draft:
        print("\n🎉 草稿生成完成!")
        # 保存简单版本
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = f"draft_{topic[:10]}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {topic}\n\n")
            for section in draft.structure.sections:
                f.write(f"## {section.title}\n")
                f.write(f"{draft.content.get(section.id, '')}\n\n")
        
        print(f"💾 已保存: {filename}")
    else:
        print("🔄 尝试使用备用模式...")

if __name__ == "__main__":
    main()