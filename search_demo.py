#!/usr/bin/env python3
"""
Google搜索演示 - 简单易用的搜索工具
"""

import asyncio
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient

class SimpleSearcher:
    def __init__(self):
        self.client = GoogleSearchClient()
    
    async def search(self, query: str, num_results: int = 5):
        """执行搜索"""
        try:
            print(f"Searching: {query}")
            response = await self.client.search(query, num_results=num_results)
            
            print(f"Found {len(response.items)} results:")
            print("-" * 80)
            
            for i, item in enumerate(response.items, 1):
                print(f"{i}. {item.title}")
                print(f"   URL: {item.link}")
                print(f"   Description: {item.snippet}")
                print("-" * 80)
                
            return response.items
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []

def search_sync(query: str, num_results: int = 5):
    """同步搜索函数"""
    async def _async_search():
        searcher = SimpleSearcher()
        return await searcher.search(query, num_results)
    
    return asyncio.run(_async_search())

async def main():
    """主函数演示"""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Python tutorial"
    
    searcher = SimpleSearcher()
    await searcher.search(query, 5)

if __name__ == "__main__":
    asyncio.run(main())