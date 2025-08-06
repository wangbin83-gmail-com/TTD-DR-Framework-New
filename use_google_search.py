#!/usr/bin/env python3
"""
Google搜索使用示例 - 可直接运行的简单脚本

这个脚本展示了如何使用TTD-DR框架的Google搜索功能
"""

import asyncio
import sys
import os

# 确保路径正确
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient

async def search_examples():
    """搜索示例"""
    client = GoogleSearchClient()
    
    # 搜索示例
    queries = [
        "Python FastAPI tutorial",
        "AI automation tools 2025",
        "machine learning frameworks"
    ]
    
    for query in queries:
        print(f"\n=== 搜索: {query} ===")
        try:
            response = await client.search(query, num_results=3)
            
            for i, item in enumerate(response.items, 1):
                print(f"{i}. {item.title}")
                print(f"   {item.link}")
                print(f"   {item.snippet}")
                print()
                
        except Exception as e:
            print(f"搜索失败: {e}")

def simple_search(query: str, num_results: int = 5) -> list:
    """简单的同步搜索函数
    
    参数:
        query: 搜索关键词
        num_results: 返回结果数量 (默认5个)
        
    返回:
        搜索结果列表，每项包含(title, link, snippet)
    """
    async def _search():
        client = GoogleSearchClient()
        response = await client.search(query, num_results=num_results)
        
        return [(item.title, item.link, item.snippet) 
                for item in response.items]
    
    return asyncio.run(_search())

# 使用示例
if __name__ == "__main__":
    print("TTD-DR Google搜索功能演示")
    print("=" * 50)
    
    # 运行示例搜索
    asyncio.run(search_examples())
    
    # 或者直接使用简单函数
    print("\n" + "=" * 50)
    print("简单使用方式:")
    print("results = simple_search('Python tutorial', 3)")
    
    # 示例
    results = simple_search("Python tutorial", 3)
    for title, link, snippet in results:
        print(f"- {title}: {link}")