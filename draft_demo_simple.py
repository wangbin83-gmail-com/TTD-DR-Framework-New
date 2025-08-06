#!/usr/bin/env python3
"""
简单草稿生成演示 - 使用框架核心功能
"""

import sys
import os
from datetime import datetime

# 添加 backend 到路径
sys.path.append('backend')

def demonstrate_draft_generation():
    """演示草稿生成流程"""
    
    # 使用已有的测试功能
    try:
        from workflow.draft_generator import KimiK2DraftGenerator
        from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel
        
        # 示例主题
        topics = [
            "区块链技术在供应链金融中的应用",
            "人工智能在教育领域的创新应用", 
            "气候变化对全球经济的影响"
        ]
        
        topic = topics[0]  # 使用第一个主题作为演示
        
        print("=== TTD-DR 草稿生成演示 ===")
        print(f"主题: {topic}")
        print("-" * 40)
        
        # 创建需求配置
        requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            max_iterations=3,
            quality_threshold=0.8
        )
        
        print("1. 研究需求配置:")
        print(f"   - 领域: {requirements.domain}")
        print(f"   - 复杂度: {requirements.complexity_level}")
        print(f"   - 最大迭代: {requirements.max_iterations}")
        print(f"   - 质量阈值: {requirements.quality_threshold}")
        
        print("\n2. 预期生成的草稿结构:")
        print("   - 引言 (Introduction)")
        print("   - 背景研究 (Background)")
        print("   - 方法论 (Methodology)")
        print("   - 关键发现 (Key Findings)")
        print("   - 结论 (Conclusion)")
        
        print("\n3. 执行命令:")
        print("   python -c \"")
        print("   import asyncio")
        print("   from workflow.draft_generator import KimiK2DraftGenerator")
        print("   from models.core import ResearchRequirements, ResearchDomain, ComplexityLevel")
        print("")
        print("   async def run():")
        print("       generator = KimiK2DraftGenerator()")
        print("       requirements = ResearchRequirements(")
        print("           domain=ResearchDomain.TECHNOLOGY,")
        print("           complexity_level=ComplexityLevel.INTERMEDIATE,")
        print("           max_iterations=2,")
        print("           quality_threshold=0.7")
        print("       )")
        print("       draft = await generator.generate_initial_draft('" + topic + "', requirements)")
        print("       return draft")
        print("")
        print("   draft = asyncio.run(run())")
        print("   print('草稿生成完成!')")
        print("   \"")
        
        print("\n4. 输出文件:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"draft_demo_{timestamp}.md"
        print(f"   文件将保存为: {filename}")
        
        return True
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖并配置了API密钥")
        return False

if __name__ == "__main__":
    demonstrate_draft_generation()