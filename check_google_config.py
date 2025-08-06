#!/usr/bin/env python3
"""
Google搜索配置检查工具
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.config.settings import settings
from backend.services.google_search_client import GoogleSearchClient

def check_config():
    """检查配置"""
    print("Google搜索配置检查")
    print("=" * 50)
    
    # 检查.env文件
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env文件存在")
    else:
        print("❌ .env文件不存在")
        print("请创建.env文件并添加必要的配置")
        return False
    
    # 检查API密钥
    api_key = settings.google_search_api_key
    engine_id = settings.google_search_engine_id
    
    print(f"Google API Key: {'✅' if api_key and 'your_' not in str(api_key) else '❌'}")
    print(f"Google Engine ID: {'✅' if engine_id and 'your_' not in str(engine_id) else '❌'}")
    
    if api_key and 'your_' not in str(api_key):
        print(f"   API Key: {api_key[:10]}...")
    
    if engine_id and 'your_' not in str(engine_id):
        print(f"   Engine ID: {engine_id[:10]}...")
    
    # 检查客户端
    try:
        client = GoogleSearchClient()
        print("✅ GoogleSearchClient 创建成功")
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")
        return False
    
    return bool(api_key and engine_id and 'your_' not in str(api_key) and 'your_' not in str(engine_id))

def print_setup_guide():
    """打印配置指南"""
    print("\n配置指南:")
    print("1. 访问 https://console.developers.google.com/")
    print("2. 创建项目并启用 Custom Search API")
    print("3. 创建 API 密钥")
    print("4. 访问 https://cse.google.com/cse/ 创建搜索引擎")
    print("5. 在 .env 文件中添加:")
    print("   GOOGLE_SEARCH_API_KEY=你的实际API密钥")
    print("   GOOGLE_SEARCH_ENGINE_ID=你的实际搜索引擎ID")

if __name__ == "__main__":
    is_configured = check_config()
    
    if not is_configured:
        print_setup_guide()
    else:
        print("\n✅ 配置完成！可以使用Google搜索功能了。")