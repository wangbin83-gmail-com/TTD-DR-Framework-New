# TTD-DR Framework

## 项目概述

TTD-DR Framework 是一个基于AI驱动的技术研究框架，专注于跨学科研究的深度分析和报告生成。该系统结合了Kimi K2大语言模型和Google搜索API，提供智能化的研究能力。

## 核心功能

- **智能研究助手**: 基于Kimi K2的AI驱动研究分析
- **跨学科整合**: 支持多个研究领域的深度分析
- **自动化报告**: 生成结构化的研究报告
- **实时搜索**: 集成Google搜索API获取最新信息
- **可视化界面**: 现代化的Web界面，支持实时交互

## 技术架构

### 后端 (Python/FastAPI)
- **框架**: FastAPI + Uvicorn
- **AI模型**: Kimi K2 (Moonshot AI)
- **搜索**: Google Custom Search API
- **数据处理**: Pydantic + 异步处理
- **部署**: Python 3.8+

### 前端 (React/TypeScript)
- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **UI组件**: Tailwind CSS + Headless UI
- **状态管理**: React Hooks
- **API通信**: Axios

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- Google Cloud API密钥
- Kimi K2 API密钥

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd TTD-DR-Framework
   ```

2. **后端设置**
   ```bash
   cd backend
   pip install -r requirements.txt
   cp .env.example .env
   # 编辑.env文件，填入API密钥
   python main.py
   ```

3. **前端设置**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### 环境变量配置

复制 `.env.example` 为 `.env` 并配置以下参数：

```bash
# Kimi K2 API Configuration
KIMI_K2_API_KEY=your_kimi_k2_api_key_here
KIMI_K2_BASE_URL=https://api.moonshot.cn/v1

# Google Search API Configuration
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
```

## API文档

### 核心端点

- `GET /` - 系统健康检查
- `POST /api/research/generate` - 生成研究报告
- `POST /api/research/analyze` - 分析研究主题
- `GET /api/research/status/{task_id}` - 查询任务状态

### WebSocket接口
- `ws://localhost:8000/ws` - 实时状态更新

## 项目结构

```
TTD-DR-Framework/
├── backend/                 # FastAPI后端
│   ├── api/                # API路由
│   ├── services/           # 业务逻辑
│   ├── models/             # 数据模型
│   ├── workflow/           # 工作流引擎
│   └── main.py             # 应用入口
├── frontend/               # React前端
│   ├── src/
│   │   ├── components/     # UI组件
│   │   ├── services/       # API服务
│   │   └── types/          # TypeScript类型
│   └── package.json
├── docs/                   # 项目文档
└── tests/                  # 测试文件
```

## 开发指南

### 运行测试

**后端测试**
```bash
cd backend
pytest tests/
```

**前端测试**
```bash
cd frontend
npm test
```

### 代码规范

- **Python**: PEP 8标准，使用black格式化
- **TypeScript**: ESLint + Prettier
- **提交规范**: 遵循Conventional Commits

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 支持

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

## 更新日志

### v1.0.0 (2024-08)
- 初始版本发布
- 集成Kimi K2 AI模型
- 支持Google搜索API
- 基础研究报告生成功能
- 现代化Web界面