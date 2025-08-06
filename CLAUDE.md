# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TTD-DR Framework** - Test-Time Diffusion Deep Researcher: An automated research report generation system that uses AI models and web search to create comprehensive research reports through iterative refinement.

## Architecture

### Core Components

#### Backend (Python/FastAPI)
- **Entry Point**: `backend/main.py` - FastAPI server with CORS for React frontend
- **Core Models**: `backend/models/core.py` - Pydantic models for research structure, gaps, drafts, and workflow state
- **Workflow Engine**: `backend/workflow/graph.py` - Custom LangGraph-like workflow orchestration
- **Dynamic Retrieval**: `backend/services/dynamic_retrieval_engine.py` - Google Search integration with credibility scoring
- **AI Integration**: Multiple Kimi K2 client services for draft generation, gap analysis, and content enhancement

#### Frontend (React/Vite)
- **Framework**: React 18 + TypeScript + Vite
- **Entry Point**: `frontend/src/App.tsx` - Basic React setup with API integration ready
- **Proxy**: Vite dev server configured to proxy `/api` to backend at `localhost:8000`

### Workflow Architecture

The system implements an 8-node iterative workflow:

```
draft_generator → gap_analyzer → retrieval_engine → information_integrator → quality_assessor
                      ↑                                                            ↓
                      └─────────────────── (if quality < threshold) ──────────────┘
                                                    ↓
                                          (if quality ≥ threshold)
                                                    ↓
                                        self_evolution_enhancer → report_synthesizer
```

## Key Directories

```
backend/
├── main.py                 # FastAPI server entry
├── models/                 # Pydantic data models
│   ├── core.py            # Core domain models
│   ├── research_structure.py
│   └── state_management.py
├── services/              # AI and search service integrations
│   ├── kimi_k2_client.py  # Kimi K2 API client
│   ├── google_search_client.py
│   └── dynamic_retrieval_engine.py
├── workflow/              # LangGraph workflow implementation
│   ├── graph.py          # Main workflow orchestration
│   ├── draft_generator.py
│   └── demo.py           # Workflow demonstration
└── tests/                # Backend test suite

frontend/
├── src/
│   ├── App.tsx           # React main component
│   └── services/api.ts   # API client setup
└── vite.config.ts        # Vite configuration with proxy
```

## Commands

### Backend Development
```bash
# Install dependencies
cd backend && pip install -r requirements.txt

# Run backend server
cd backend && python main.py
# or
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run backend tests
cd backend && python -m pytest tests/ -v

# Run specific test categories
python -m pytest backend/tests/test_workflow_structure.py -v
python -m pytest backend/tests/test_dynamic_retrieval_engine.py -v

# Run integration tests
python backend/test_workflow_integration.py
python backend/test_integration.py
```

### Frontend Development
```bash
# Install dependencies
cd frontend && npm install

# Start development server
cd frontend && npm run dev
# Serves on http://localhost:5173 with API proxy to backend

# Build for production
cd frontend && npm run build

# Run linting
cd frontend && npm run lint
```

### System Testing & Verification
```bash
# Verify complete framework setup
python verify_setup.py

# Test end-to-end report generation
python test_end_to_end_report_generation.py

# Test individual components
python test_simple_kimi_call.py
python test_google_search_integration.py
python test_gap_analyzer_integration.py
python test_information_integration_workflow.py
```

### Environment Configuration

**Required Environment Variables** (create `.env` file):
```bash
# Kimi K2 API Configuration
KIMI_K2_API_KEY=your_kimi_api_key_here
KIMI_K2_BASE_URL=https://api.kimi.moonshot.cn/v1

# Google Search API Configuration
GOOGLE_API_KEY=your_google_search_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Optional: Backend configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

## Key Services Integration

### Kimi K2 AI Model
- **Client**: `services/kimi_k2_client.py`
- **Capabilities**: Text generation, gap analysis, content enhancement, coherence management
- **Models**: Supports various Kimi models (k1, k1.5, k1.6)

### Google Search API
- **Client**: `services/google_search_client.py`
- **Integration**: Dynamic retrieval engine with credibility scoring
- **Features**: Source filtering, relevance scoring, content extraction

### Content Quality Pipeline
- **Filter**: `services/content_filter.py` - Multi-layer content quality assessment
- **Scoring**: Domain authority, recency, source credibility, content quality

## Development Workflow

### Testing Strategy
1. **Unit Tests**: Individual service and model testing
2. **Integration Tests**: Workflow node and system integration
3. **End-to-End Tests**: Complete report generation pipeline
4. **API Tests**: Backend endpoint testing

### Debugging
- **State Inspection**: Use `workflow_states/` directory for state persistence
- **Logging**: Comprehensive logging in all services and workflow nodes
- **Error Handling**: Graceful fallbacks with fallback content generation

### Adding New Features
1. **Workflow Nodes**: Add to `workflow/graph.py` following existing patterns
2. **Services**: Implement in `services/` directory with proper error handling
3. **Models**: Extend `models/core.py` with new Pydantic models
4. **Tests**: Add corresponding tests in `tests/` directory

## Performance Characteristics

- **Backend**: FastAPI async handling, efficient state management
- **Frontend**: Vite build optimization, React 18 concurrent features
- **Workflow**: Configurable iteration limits (default 3-5 iterations)
- **API Limits**: Respects Kimi K2 and Google Search API rate limits

## Common Development Tasks

### Create New Research Topic
```python
from backend.workflow.graph import create_ttdr_workflow
from backend.models.core import ResearchRequirements, ResearchDomain

workflow = create_ttdr_workflow()
requirements = ResearchRequirements(
    domain=ResearchDomain.TECHNOLOGY,
    complexity_level=ComplexityLevel.ADVANCED,
    max_iterations=5,
    quality_threshold=0.85
)
# Execute workflow...
```

### Debug Workflow Execution
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check workflow execution path
python backend/workflow/demo.py
```