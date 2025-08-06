"""
简化的Kimi K2研究报告生成演示
Simplified Kimi K2 research report generation demo
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_simple_report_with_kimi_k2(topic: str):
    """使用Kimi K2生成简化的研究报告"""
    logger.info(f"开始生成报告: {topic}")
    
    try:
        from services.kimi_k2_client import KimiK2Client
        from config.settings import settings
        
        execution_id = f"kimi_k2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()
        
        async with KimiK2Client() as client:
            logger.info(f"使用模型: {settings.kimi_k2_model}")
            
            # 生成研究报告
            prompt = f"""
请为主题"{topic}"撰写一份专业的研究报告。

要求:
1. 包含以下结构：摘要、引言、现状分析、发展趋势、挑战与机遇、案例分析、结论
2. 每个部分都要有具体内容，总字数2000-3000字
3. 语言专业、逻辑清晰
4. 包含具体的数据和实例
5. 使用Markdown格式

请直接生成完整的研究报告内容。
"""
            
            response = await client.generate_text(
                prompt, 
                max_tokens=2500,
                temperature=0.7
            )
            
            execution_time = time.time() - start_time
            
            # 创建完整报告
            final_report = f"""# 研究报告: {topic}

**由TTD-DR框架生成 (使用Kimi K2模型)**
**执行ID:** {execution_id}
**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**执行时间:** {execution_time:.2f} 秒
**使用模型:** {settings.kimi_k2_model}

---

{response.content}

---

## 生成信息

- **模型:** {response.model}
- **Token使用:** {response.usage}
- **完成原因:** {response.finish_reason}
- **内容长度:** {len(response.content)} 字符

---
*本报告由TTD-DR框架使用Kimi K2模型生成*
"""
            
            # 保存报告
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_topic = safe_topic.replace(' ', '_')[:30]
            filename = f"kimi_k2_report_{execution_id}_{safe_topic}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            logger.info(f"报告已保存: {filename}")
            
            return {
                "success": True,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "filename": filename,
                "content_length": len(response.content),
                "model": response.model,
                "usage": response.usage
            }
            
    except Exception as e:
        logger.error(f"生成失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time if 'start_time' in locals() else 0
        }

async def main():
    """运行两个报告生成任务"""
    print("🚀 Kimi K2研究报告生成演示")
    print("=" * 60)
    
    topics = [
        "人工智能在教育领域的应用与发展趋势",
        "区块链技术在金融科技中的创新应用"
    ]
    
    results = []
    
    for i, topic in enumerate(topics, 1):
        print(f"\n📊 生成报告 {i}/2: {topic}")
        print("-" * 40)
        
        result = await generate_simple_report_with_kimi_k2(topic)
        results.append({"topic": topic, "result": result})
        
        if result["success"]:
            print(f"✅ 成功生成报告")
            print(f"   执行时间: {result['execution_time']:.2f}秒")
            print(f"   内容长度: {result['content_length']:,} 字符")
            print(f"   使用模型: {result['model']}")
            print(f"   文件: {result['filename']}")
        else:
            print(f"❌ 生成失败: {result['error']}")
        
        if i < len(topics):
            print("⏳ 准备下一个报告...")
            await asyncio.sleep(3)
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 生成总结")
    print("=" * 60)
    
    successful = sum(1 for r in results if r["result"]["success"])
    print(f"成功生成: {successful}/{len(results)} 个报告")
    
    if successful > 0:
        avg_time = sum(r["result"]["execution_time"] for r in results if r["result"]["success"]) / successful
        print(f"平均生成时间: {avg_time:.2f}秒")
    
    for i, data in enumerate(results, 1):
        result = data["result"]
        print(f"\n报告 {i}: {data['topic']}")
        if result["success"]:
            print(f"  ✅ 成功 - {result['filename']}")
        else:
            print(f"  ❌ 失败 - {result['error']}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(main())
    successful = sum(1 for r in results if r["result"]["success"])
    exit(0 if successful > 0 else 1)