#!/usr/bin/env python3
"""
Google搜索快速使用指南

使用方法:
1. 首先配置API密钥
2. 然后运行: python quick_google_search.py "你的搜索词"
"""

import asyncio
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient

async def search_google(query: str, max_results: int = 5):
    """执行Google搜索"""
    client = GoogleSearchClient()
    
    # 检查配置
    if not client.api_key:
        print("错误: 未设置GOOGLE_SEARCH_API_KEY")
        print("请在.env文件中配置:")
        print("GOOGLE_SEARCH_API_KEY=你的API密钥")
        return None
    
    if not client.search_engine_id:
        print("错误: 未设置GOOGLE_SEARCH_ENGINE_ID")
        print("请在.env文件中配置:")
        print("GOOGLE_SEARCH_ENGINE_ID=你的搜索引擎ID")
        return None
    
    try:
        print(f"正在搜索: {query}")
        response = await client.search(query, num_results=max_results)
        
        results = []
        for item in response.items:
            results.append({
                'title': item.title,
                'url': item.link,
                'description': item.snippet
            })
        
        return results
        
    except Exception as e:
        print(f"搜索错误: {e}")
        return None

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python quick_google_search.py \"搜索词\" [结果数量]")
        print("示例: python quick_google_search.py \"Python教程\" 5")
        return
    
    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # 运行搜索
    results = asyncio.run(search_google(query, max_results))
    
    if results:
        print(f"\n找到 {len(results)} 个结果:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   描述: {result['description']}")
            print()
    else:
        print("未找到结果或配置错误")

if __name__ == "__main__":
    main()