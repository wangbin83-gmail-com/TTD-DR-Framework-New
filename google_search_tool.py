#!/usr/bin/env python3
"""
Google搜索工具 - 简单易用的命令行搜索工具

使用方法:
    python google_search_tool.py "搜索关键词" [--num 结果数量]
    
示例:
    python google_search_tool.py "Python FastAPI教程" --num 5
    python google_search_tool.py "AI自动化工具"
"""

import asyncio
import argparse
import sys
import os
from typing import List, Dict, Optional

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.google_search_client import GoogleSearchClient

class GoogleSearchTool:
    """Google搜索命令行工具"""
    
    def __init__(self):
        self.client = GoogleSearchClient()
    
    def check_configuration(self) -> bool:
        """检查API配置"""
        if not self.client.api_key:
            print("❌ 错误: 未设置GOOGLE_SEARCH_API_KEY")
            print("解决方案:")
            print("1. 创建 .env 文件")
            print("2. 添加: GOOGLE_SEARCH_API_KEY=你的实际API密钥")
            return False
        
        if not self.client.search_engine_id:
            print("❌ 错误: 未设置GOOGLE_SEARCH_ENGINE_ID")
            print("解决方案:")
            print("1. 创建 .env 文件")
            print("2. 添加: GOOGLE_SEARCH_ENGINE_ID=你的实际搜索引擎ID")
            return False
        
        return True
    
    async def search(self, query: str, num_results: int = 5) -> Optional[List[Dict]]:
        """执行搜索"""
        if not self.check_configuration():
            return None
        
        try:
            print(f"🔍 正在搜索: {query}")
            response = await self.client.search(query, num_results=min(num_results, 10))
            
            if not response.items:
                print("⚠️  未找到相关结果")
                return []
            
            results = []
            for item in response.items:
                results.append({
                    'title': item.title,
                    'url': item.link,
                    'description': item.snippet,
                    'domain': item.display_link
                })
            
            print(f"✅ 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return None
    
    async def search_multiple_pages(self, query: str, max_results: int = 20) -> Optional[List[Dict]]:
        """多页搜索获取更多结果"""
        if not self.check_configuration():
            return None
        
        try:
            print(f"🔍 正在搜索 (多页): {query}")
            response = await self.client.search_multiple_pages(query, max_results=max_results)
            
            if not response.items:
                print("⚠️  未找到相关结果")
                return []
            
            results = []
            for item in response.items:
                results.append({
                    'title': item.title,
                    'url': item.link,
                    'description': item.snippet,
                    'domain': item.display_link
                })
            
            print(f"✅ 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return None
    
    def display_results(self, results: List[Dict]) -> None:
        """显示搜索结果"""
        if not results:
            return
        
        print("\n" + "=" * 80)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   🌐 {result['url']}")
            print(f"   📝 {result['description']}")
            print(f"   📍 {result['domain']}")
            print("-" * 80)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Google搜索工具')
    parser.add_argument('query', help='搜索关键词')
    parser.add_argument('--num', type=int, default=5, 
                       help='结果数量 (1-20, 默认: 5)')
    parser.add_argument('--multi', action='store_true',
                       help='使用多页搜索获取更多结果')
    
    args = parser.parse_args()
    
    if args.num < 1 or args.num > 20:
        print("错误: 结果数量必须在1-20之间")
        return
    
    tool = GoogleSearchTool()
    
    if args.multi:
        results = await tool.search_multiple_pages(args.query, args.num)
    else:
        results = await tool.search(args.query, args.num)
    
    if results:
        tool.display_results(results)

if __name__ == "__main__":
    asyncio.run(main())