#!/usr/bin/env python3
"""
Google搜索功能演示
"""

import asyncio
import sys
import os

# 添加backend到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient

async def test_google_search():
    """测试Google搜索功能"""
    
    client = GoogleSearchClient()
    
    print("Google搜索测试")
    print("=" * 30)
    
    # 检查配置
    if not client.api_key or not client.search_engine_id:
        print("需要配置API密钥:")
        print("1. 创建 .env 文件")
        print("2. 添加 GOOGLE_SEARCH_API_KEY 和 GOOGLE_SEARCH_ENGINE_ID")
        return False
    
    print("API配置已就绪")
    print(f"API Key: {client.api_key[:10]}...")
    print(f"Engine ID: {client.search_engine_id[:10]}...")
    
    try:
        # 执行简单搜索
        print("\n正在搜索: Python tutorial")
        response = await client.search("Python tutorial", num_results=3)
        
        print(f"找到 {len(response.items)} 个结果")
        
        for i, item in enumerate(response.items, 1):
            print(f"\n{i}. {item.title}")
            print(f"   链接: {item.link}")
            print(f"   摘要: {item.snippet[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"搜索失败: {e}")
        return False

def simple_search_sync(query: str, num_results: int = 5):
    """同步搜索函数"""
    
    async def _async_search():
        client = GoogleSearchClient()
        if not client.api_key or not client.search_engine_id:
            return None
        
        try:
            response = await client.search(query, num_results=num_results)
            return [
                {
                    "title": item.title,
                    "link": item.link,
                    "snippet": item.snippet
                }
                for item in response.items
            ]
        except Exception as e:
            print(f"错误: {e}")
            return []
    
    return asyncio.run(_async_search())

if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(test_google_search())
    
    if success:
        print("\n测试完成！")
        print("可以使用 simple_search_sync() 函数进行搜索")
    else:
        print("\n测试失败")