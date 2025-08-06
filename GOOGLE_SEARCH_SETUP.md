# Google搜索API配置和使用指南

## 概述

TTD-DR框架使用Google Custom Search API进行网络搜索。本指南将帮助你配置和使用该功能。

## 1. 获取Google API凭据

### 步骤1: 创建Google Cloud项目
1. 访问 [Google Cloud Console](https://console.developers.google.com/)
2. 点击"选择项目" → "新建项目"
3. 输入项目名称，点击"创建"

### 步骤2: 启用Custom Search API
1. 在Google Cloud Console中，进入"API和服务" → "库"
2. 搜索"Custom Search API"
3. 点击该API，然后点击"启用"

### 步骤3: 创建API密钥
1. 进入"API和服务" → "凭据"
2. 点击"创建凭据" → "API密钥"
3. 复制生成的API密钥

AIzaSyCpXD_tALUAdcALE-65DHZrEIAXWr4jlEc

### 步骤4: 创建自定义搜索引擎
1. 访问 [Google Custom Search Engine](https://cse.google.com/cse/)
2. 点击"创建自定义搜索引擎"
3. 配置搜索引擎:
   - 网站搜索: `*` (搜索整个网络)
   - 语言: 选择需要的语言
   - 地区: 选择需要的地区
4. 点击"创建"
5. 复制"搜索引擎ID"

a75a90078815f472e

## 2. 配置项目

### 创建.env文件
在项目根目录创建 `.env` 文件，添加以下内容:

```bash
# Google搜索API配置
GOOGLE_SEARCH_API_KEY=你的实际API密钥
GOOGLE_SEARCH_ENGINE_ID=你的实际搜索引擎ID

# Kimi K2 API配置 (可选)
KIMI_K2_API_KEY=你的kimi_api_key
```

### 验证配置
运行以下命令验证配置:

```bash
python -c "
from backend.config.settings import settings
print('Google API Key:', bool(settings.google_search_api_key))
print('Google Engine ID:', bool(settings.google_search_engine_id))
"
```

## 3. 使用示例

### 基础使用

```python
import asyncio
import sys
import os

# 添加路径
sys.path.insert(0, 'backend')

from backend.services.google_search_client import GoogleSearchClient

async def simple_search():
    client = GoogleSearchClient()
    
    # 搜索Python教程
    response = await client.search("Python tutorial", num_results=5)
    
    for i, item in enumerate(response.items, 1):
        print(f"{i}. {item.title}")
        print(f"   {item.link}")
        print(f"   {item.snippet}")
        print()

# 运行搜索
asyncio.run(simple_search())
```

### 高级搜索

```python
async def advanced_search():
    client = GoogleSearchClient()
    
    # 搜索最近一周的AI相关内容
    response = await client.search(
        "AI automation tools",
        num_results=10,
        date_restrict="w1",  # 最近一周
        language="lang_en"
    )
    
    return response.items
```

### 多页搜索

```python
async def multi_page_search():
    client = GoogleSearchClient()
    
    # 获取更多结果（最多20个）
    response = await client.search_multiple_pages(
        "machine learning frameworks",
        max_results=20
    )
    
    return response.items
```

## 4. 现成使用脚本

### 快速搜索脚本
```bash
# 使用命令行搜索
python quick_google_search.py "Python tutorial" 5
```

### 批量搜索脚本
```bash
# 批量搜索多个关键词
python batch_search.py
```

## 5. 错误处理

常见错误及解决方案:

### API密钥无效
```
错误: API key not valid
解决: 检查.env文件中的GOOGLE_SEARCH_API_KEY是否正确
```

### 超出配额
```
错误: Daily quota exceeded
解决: 
- 免费配额为每天100次查询
- 考虑升级到付费计划
- 实现本地缓存减少API调用
```

### 搜索引擎ID错误
```
错误: Invalid CSE ID
解决: 检查.env文件中的GOOGLE_SEARCH_ENGINE_ID
```

## 6. 性能优化

### 缓存搜索结果
```python
import json
import os

class CachedGoogleSearch:
    def __init__(self, cache_file="search_cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    async def search_with_cache(self, query):
        if query in self.cache:
            return self.cache[query]
        
        # 执行实际搜索
        client = GoogleSearchClient()
        response = await client.search(query)
        
        # 缓存结果
        self.cache[query] = response.items
        self.save_cache()
        
        return response.items
```

## 7. 监控和调试

### 日志配置
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 测试连接
async def test_connection():
    client = GoogleSearchClient()
    try:
        response = await client.health_check()
        print(f"API连接状态: {response}")
    except Exception as e:
        print(f"连接失败: {e}")
```

## 8. 集成到TTD-DR工作流

Google搜索被集成在以下工作流节点中:
- `retrieval_engine_node`: 基于信息缺口进行搜索
- `dynamic_retrieval_engine.py`: 动态检索引擎

### 示例工作流
```python
from backend.workflow.graph import create_ttdr_workflow
from backend.models.core import ResearchRequirements

workflow = create_ttdr_workflow()
requirements = ResearchRequirements(
    topic="AI in healthcare",
    search_queries=["AI healthcare applications", "medical AI tools"]
)

# 执行工作流
result = await workflow.ainvoke({"requirements": requirements})
```

## 9. 使用限制

- 免费配额: 每天100次查询
- 每次查询最多返回10个结果
- 多页搜索最多支持20个结果
- 需要有效的网络连接

## 10. 故障排除

如果仍然遇到问题:
1. 检查API密钥是否正确
2. 确认Custom Search API已启用
3. 验证搜索引擎ID是否正确
4. 检查网络连接
5. 查看Google Cloud Console中的API使用情况