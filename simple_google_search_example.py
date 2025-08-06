#!/usr/bin/env python3
"""
简单Google搜索使用示例
演示如何使用TTD-DR框架的Google搜索功能
"""

import asyncio
import sys
import os
from typing import List

# 添加backend到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient, GoogleSearchResult

class SimpleGoogleSearcher:
    """简单的Google搜索包装器"""
    
    def __init__(self):
        self.client = GoogleSearchClient()
    
    async def search(self, query: str, num_results: int = 5) -> List[dict]:
        """
        执行简单搜索并返回格式化结果
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量 (1-20)
        
        Returns:
            包含搜索结果的列表，每个结果包含title, link, snippet
        """
        try:
            # 确保配置了API密钥
            if not self.client.api_key or not self.client.search_engine_id:
                print("⚠️  警告: Google搜索API未配置")
                print("请设置环境变量:")
                print("GOOGLE_SEARCH_API_KEY=你的API密钥")
                print("GOOGLE_SEARCH_ENGINE_ID=你的搜索引擎ID")
                return []
            
            # 执行搜索
            response = await self.client.search(query, num_results=min(num_results, 10))
            
            # 格式化结果
            results = []
            for item in response.items:
                results.append({
                    "title": item.title,
                    "link": item.link,
                    "snippet": item.snippet,
                    "display_link": item.display_link
                })
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    async def search_multiple(self, query: str, max_results: int = 10) -> List[dict]:
        """搜索多个页面获取更多结果"""
        try:
            if not self.client.api_key or not self.client.search_engine_id:
                print("⚠️  警告: Google搜索API未配置")
                return []
            
            response = await self.client.search_multiple_pages(query, max_results=max_results)
            
            results = []
            for item in response.items:
                results.append({
                    "title": item.title,
                    "link": item.link,
                    "snippet": item.snippet,
                    "display_link": item.display_link
                })
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []

async def main():
    """主函数 - 演示如何使用"""
    
    # 创建搜索器实例
    searcher = SimpleGoogleSearcher()
    
    print("🔍 TTD-DR Google搜索示例")
    print("=" * 50)
    
    # 检查是否配置了API
    if not searcher.client.api_key or not searcher.client.search_engine_id:
        print("🔧 配置说明:")
        print("1. 创建 .env 文件在项目根目录")
        print("2. 添加以下内容:")
        print("   GOOGLE_SEARCH_API_KEY=你的API密钥")
        print("   GOOGLE_SEARCH_ENGINE_ID=你的搜索引擎ID")
        print("\n3. 获取API密钥:")
        print("   - 访问 https://console.developers.google.com/")
        print("   - 创建项目并启用Custom Search API")
        print("   - 创建API密钥")
        print("   - 访问 https://cse.google.com/cse/ 创建搜索引擎")
        print("   - 获取搜索引擎ID")
        return
    
    # 示例搜索
    queries = [
        "Python FastAPI tutorial",
        "AI research automation tools",
        "LangGraph workflow patterns"
    ]
    
    for query in queries:
        print(f"\n📋 搜索: {query}")
        print("-" * 30)
        
        results = await searcher.search(query, num_results=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   🔗 {result['link']}")
                print(f"   📝 {result['snippet'][:100]}...")
                print()
        else:
            print("没有找到结果")
        
        # 短暂延迟，避免频繁请求
        await asyncio.sleep(1)

def search_sync(query: str, num_results: int = 5) -> List[dict]:
    """
    同步版本的搜索函数，方便直接调用
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
    
    Returns:
        搜索结果列表
    """
    async def _async_search():
        searcher = SimpleGoogleSearcher()
        return await searcher.search(query, num_results)
    
    return asyncio.run(_async_search())

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
    
    # 或者使用同步版本
    # results = search_sync("Python tutorial", 3)
    # for result in results:
    #     print(f"{result['title']} - {result['link']}")