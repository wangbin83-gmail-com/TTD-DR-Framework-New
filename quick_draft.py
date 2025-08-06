#!/usr/bin/env python3
"""
快速草稿生成脚本
使用方法: python quick_draft.py "研究主题"
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加 backend 到路径
sys.path.append('backend')

from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
from workflow.draft_generator import KimiK2DraftGenerator

async def generate_draft(topic: str):
    """生成草稿"""
    print(f"主题: {topic}")
    
    try:
        generator = KimiK2DraftGenerator()
        
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=2,
            quality_threshold=0.7
        )
        
        print("正在生成草稿...")
        draft = await generator.generate_initial_draft(topic, requirements)
        
        if draft:
            print(f"成功! 章节: {len(draft.structure.sections)}, 评分: {draft.quality_score:.2f}")
            
            # 创建文件名
            timestamp = datetime.now().strftime("%m%d_%H%M")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()[:20]
            filename = f"draft_{safe_topic}_{timestamp}.md"
            
            # 保存文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {topic}\n\n")
                for section in draft.structure.sections:
                    content = draft.content.get(section.id, "")
                    f.write(f"## {section.title}\n\n{content}\n\n")
            
            print(f"已保存: {filename}")
            return True
        else:
            print("生成失败")
            return False
            
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "人工智能在教育中的应用"
    print("TTD-DR 草稿生成器")
    print("-" * 20)
    
    success = asyncio.run(generate_draft(topic))
    
    if success:
        print("完成!")
    else:
        print("请检查API配置")

if __name__ == "__main__":
    main()