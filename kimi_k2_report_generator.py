"""
使用Kimi K2模型生成完整研究报告
Complete research report generation using Kimi K2 model
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def generate_research_report_with_kimi_k2(topic: str, complexity: str = "intermediate"):
    """
    使用Kimi K2模型生成完整的研究报告
    
    Args:
        topic: 研究主题
        complexity: 复杂度级别
    """
    logger.info(f"开始使用Kimi K2生成研究报告: {topic}")
    
    try:
        # Import Kimi K2 client
        from services.kimi_k2_client import KimiK2Client
        from config.settings import settings
        
        # Verify Kimi K2 configuration
        logger.info(f"使用模型: {settings.kimi_k2_model}")
        logger.info(f"API端点: {settings.kimi_k2_base_url}")
        
        # Generate execution ID
        execution_id = f"kimi_k2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{topic.replace(' ', '_')[:20]}"
        start_time = time.time()
        
        async with KimiK2Client() as client:
            logger.info("Kimi K2客户端连接成功")
            
            # Step 1: 生成研究大纲
            logger.info("步骤1: 生成研究大纲...")
            outline_prompt = f"""
请为主题"{topic}"生成一个详细的研究报告大纲。

要求:
1. 包含完整的章节结构
2. 每个章节都有具体的子主题
3. 适合{complexity}级别的深度分析
4. 符合学术研究报告的标准格式
5. 包含引言、现状分析、发展趋势、挑战与机遇、案例研究、政策建议、结论等部分

请用中文回答，并提供详细的大纲结构。
"""
            
            outline_response = await client.generate_text(outline_prompt, max_tokens=1000)
            logger.info(f"大纲生成完成，长度: {len(outline_response.content)} 字符")
            
            # Step 2: 生成详细内容
            logger.info("步骤2: 生成详细研究内容...")
            content_prompt = f"""
基于以下大纲，请为主题"{topic}"撰写一份完整的研究报告。

大纲:
{outline_response.content}

要求:
1. 每个章节都要有详细的内容，不少于200字
2. 包含具体的数据、案例和分析
3. 语言专业、逻辑清晰
4. 适合{complexity}级别的专业深度
5. 包含具体的实例和应用场景
6. 提供实用的建议和展望
7. 总字数控制在3000-5000字之间

请生成完整的研究报告内容，使用标准的Markdown格式。
"""
            
            content_response = await client.generate_text(content_prompt, max_tokens=3000)
            logger.info(f"详细内容生成完成，长度: {len(content_response.content)} 字符")
            
            # Step 3: 质量评估和优化
            logger.info("步骤3: 质量评估和内容优化...")
            quality_prompt = f"""
请评估以下研究报告的质量，并提供改进建议:

报告内容:
{content_response.content[:2000]}...

评估维度:
1. 内容完整性 (0-1分)
2. 逻辑连贯性 (0-1分)  
3. 专业准确性 (0-1分)
4. 实用价值 (0-1分)
5. 语言表达 (0-1分)

请提供JSON格式的评估结果:
{{
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "practicality": 0.0-1.0,
    "expression": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "suggestions": ["改进建议1", "改进建议2"]
}}
"""
            
            quality_schema = {
                "type": "object",
                "properties": {
                    "completeness": {"type": "number"},
                    "coherence": {"type": "number"},
                    "accuracy": {"type": "number"},
                    "practicality": {"type": "number"},
                    "expression": {"type": "number"},
                    "overall_score": {"type": "number"},
                    "suggestions": {"type": "array", "items": {"type": "string"}}
                }
            }
            
            quality_assessment = await client.generate_structured_response(
                quality_prompt, quality_schema, max_tokens=500
            )
            logger.info(f"质量评估完成，总分: {quality_assessment.get('overall_score', 0):.3f}")
            
            # Step 4: 根据评估结果优化内容
            if quality_assessment.get('overall_score', 0) < 0.8:
                logger.info("步骤4: 根据评估结果优化内容...")
                optimization_prompt = f"""
请根据以下评估建议优化研究报告:

原始报告:
{content_response.content}

改进建议:
{', '.join(quality_assessment.get('suggestions', []))}

请生成优化后的完整报告，重点改进以下方面:
1. 提高内容的完整性和深度
2. 增强逻辑连贯性
3. 补充更多专业数据和案例
4. 提升实用价值
5. 优化语言表达

保持Markdown格式，确保报告的专业性和可读性。
"""
                
                optimized_response = await client.generate_text(optimization_prompt, max_tokens=3500)
                final_content = optimized_response.content
                logger.info("内容优化完成")
            else:
                final_content = content_response.content
                logger.info("内容质量良好，无需优化")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create final report with metadata
            final_report = f"""# 研究报告: {topic}

**由TTD-DR框架生成 (使用Kimi K2模型)**
**执行ID:** {execution_id}
**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**执行时间:** {execution_time:.2f} 秒
**使用模型:** {settings.kimi_k2_model}
**复杂度级别:** {complexity}
**质量评分:** {quality_assessment.get('overall_score', 0):.3f}

---

{final_content}

---

## 质量评估详情

- **内容完整性:** {quality_assessment.get('completeness', 0):.3f}
- **逻辑连贯性:** {quality_assessment.get('coherence', 0):.3f}
- **专业准确性:** {quality_assessment.get('accuracy', 0):.3f}
- **实用价值:** {quality_assessment.get('practicality', 0):.3f}
- **语言表达:** {quality_assessment.get('expression', 0):.3f}
- **总体评分:** {quality_assessment.get('overall_score', 0):.3f}

## 改进建议

{chr(10).join(f"- {suggestion}" for suggestion in quality_assessment.get('suggestions', []))}

---
*本报告由TTD-DR框架使用Kimi K2模型 ({settings.kimi_k2_model}) 生成*
"""
            
            # Save report to file
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_topic = safe_topic.replace(' ', '_')[:50]
            report_filename = f"kimi_k2_report_{execution_id}_{safe_topic}.md"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            logger.info(f"报告已保存到: {report_filename}")
            
            # Return results
            return {
                "success": True,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "report_file": report_filename,
                "quality_score": quality_assessment.get('overall_score', 0),
                "report_length": len(final_content),
                "model_used": settings.kimi_k2_model,
                "steps_completed": 4 if quality_assessment.get('overall_score', 0) < 0.8 else 3
            }
            
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time if 'start_time' in locals() else 0
        }

async def main():
    """运行两个研究报告生成任务"""
    print("🚀 使用Kimi K2模型生成研究报告")
    print("=" * 80)
    
    # 定义两个研究主题
    topics = [
        {
            "topic": "人工智能在医疗诊断中的应用现状与发展前景",
            "complexity": "advanced",
            "description": "AI在医疗诊断领域的深度应用分析"
        },
        {
            "topic": "5G技术对智慧城市建设的推动作用研究",
            "complexity": "intermediate", 
            "description": "5G技术在智慧城市中的应用和影响"
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, topic_config in enumerate(topics, 1):
        print(f"\n📊 报告 {i}/2: {topic_config['topic']}")
        print(f"描述: {topic_config['description']}")
        print(f"复杂度: {topic_config['complexity']}")
        print("-" * 60)
        
        result = await generate_research_report_with_kimi_k2(
            topic=topic_config["topic"],
            complexity=topic_config["complexity"]
        )
        
        results.append({
            "topic": topic_config["topic"],
            "description": topic_config["description"],
            "result": result
        })
        
        if result["success"]:
            print(f"✅ 报告 {i} 生成成功!")
            print(f"   执行时间: {result['execution_time']:.2f}秒")
            print(f"   质量评分: {result['quality_score']:.3f}")
            print(f"   使用模型: {result['model_used']}")
            print(f"   处理步骤: {result['steps_completed']}")
            print(f"   文件: {result['report_file']}")
        else:
            print(f"❌ 报告 {i} 生成失败: {result.get('error', '未知错误')}")
        
        # 短暂暂停避免API限制
        if i < len(topics):
            print("⏳ 准备下一个报告...")
            await asyncio.sleep(2)
    
    total_execution_time = time.time() - total_start_time
    
    # 最终总结
    print("\n" + "=" * 80)
    print("🎉 Kimi K2研究报告生成完成")
    print("=" * 80)
    
    successful_reports = sum(1 for r in results if r["result"]["success"])
    failed_reports = len(results) - successful_reports
    
    print(f"总报告数量: {len(results)}")
    print(f"成功报告: {successful_reports}")
    print(f"失败报告: {failed_reports}")
    print(f"总执行时间: {total_execution_time:.2f} 秒")
    
    if successful_reports > 0:
        avg_quality = sum(r["result"]["quality_score"] for r in results if r["result"]["success"]) / successful_reports
        print(f"平均质量评分: {avg_quality:.3f}")
    
    print("\n报告详情:")
    print("-" * 40)
    
    for i, result_data in enumerate(results, 1):
        result = result_data["result"]
        print(f"\n报告 {i}: {result_data['topic']}")
        print(f"描述: {result_data['description']}")
        
        if result["success"]:
            print(f"✅ 状态: 成功")
            print(f"   执行时间: {result['execution_time']:.2f}秒")
            print(f"   质量评分: {result['quality_score']:.3f}")
            print(f"   报告长度: {result['report_length']:,} 字符")
            print(f"   使用模型: {result['model_used']}")
            print(f"   文件: {result['report_file']}")
        else:
            print(f"❌ 状态: 失败")
            print(f"   错误: {result.get('error', '未知错误')}")
            print(f"   执行时间: {result.get('execution_time', 0):.2f}秒")
    
    print(f"\n🎯 使用Kimi K2模型生成完成! 请查看生成的报告文件。")
    
    return results

if __name__ == "__main__":
    # 运行完整演示
    results = asyncio.run(main())
    
    # 根据结果退出
    successful = sum(1 for r in results if r["result"]["success"])
    exit(0 if successful > 0 else 1)