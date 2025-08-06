#!/usr/bin/env python3
"""
Google搜索简单使用示例

使用方法:
    python google_search_simple.py "搜索关键词"
    python google_search_simple.py "Python教程" 5
"""

import asyncio
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient

async def search_google(query: str, num_results: int = 5):
    """执行Google搜索"""
    client = GoogleSearchClient()
    
    try:
        print(f"Searching: {query}")
        response = await client.search(query, num_results=num_results)
        
        print(f"Found {len(response.items)} results:")
        print("-" * 80)
        
        for i, item in enumerate(response.items, 1):
            title = item.title.encode('utf-8').decode('utf-8', errors='ignore')
            url = item.link
            desc = item.snippet.encode('utf-8').decode('utf-8', errors='ignore')
            
            print(f"{i}. {title}")
            print(f"   URL: {url}")
            print(f"   Description: {desc}")
            print("-" * 80)
            
        return response.items
            
    except Exception as e:
        print(f"Search error: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Python tutorial"
    
    num = 5 if len(sys.argv) <= 2 else int(sys.argv[2])
    
    asyncio.run(search_google(query, num))