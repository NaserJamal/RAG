# RAG Training - Progressive Learning Path

Learn Retrieval-Augmented Generation (RAG) through hands-on, production-ready examples.

## What You'll Build

Retrieval-Augmented Generation (RAG) is transforming how AI systems access and use information. Instead of relying solely on knowledge baked into model weights, RAG systems retrieve relevant information from external sources and use it to generate accurate, up-to-date responses.

This repository teaches RAG fundamentals through progressively complex levels. You'll start with basic vector search, progress through hybrid retrieval strategies, master chunking techniques, build a complete pipeline, and finally implement an intelligent agentic RAG system. Each level includes:

- **Working code** you can run immediately
- **Cost-conscious implementations** with optimization strategies
- **Detailed explanations** of why and when to use each technique
- **Modular architecture** using shared components

By the end, you'll understand the core building blocks of production RAG systems and have modular, adaptable code you can use in your own projects.

## Prerequisites

- **Python 3.8+**
- **Docker and Docker Compose** (for Qdrant vector database)
- **API keys**:
  - Embedding API (OpenAI, Groq, or any OpenAI-compatible endpoint)
  - LLM API for Level 05 (OpenRouter, OpenAI, or compatible)
- **Basic understanding of Python** and command line

No prior experience with RAG, embeddings, or vector search required – we'll teach you everything.

## Quick Start

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd RAG

# 2. Create virtual environment
# macOS/Linux:
python -m venv venv
source venv/bin/activate

# Windows PowerShell:
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant vector database
docker-compose up -d

# 5. Configure API credentials
cp .env.example .env
# Edit .env and add your API keys

# 6. Run Level 01
cd levels/01-basic-vector-search
python main.py
```

## Installation

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for API calls
- Docker and Docker Compose

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd RAG
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Start Qdrant vector database**:
```bash
docker-compose up -d
```

## Levels Overview

### 🎯 Level 01: Basic Vector Search
Learn the fundamentals of semantic search using embeddings and cosine similarity.

**What you'll learn**: Text embeddings, vector similarity, basic retrieval pipeline

[Start Level 01 →](levels/01-basic-vector-search/README.md)

---

### 🔀 Level 02: Semantic vs Exact Match
Compare semantic search with keyword matching to understand when each approach excels.

**What you'll learn**: BM25 keyword search, exact matching, semantic understanding, hybrid retrieval

[Start Level 02 →](levels/02-semantic-vs-exact/README.md)

---

### ✂️ Level 03: Chunking Strategies
Master the art of splitting documents for optimal retrieval quality.

**What you'll learn**: Fixed-size chunking, recursive splitting, semantic chunking, overlap strategies

[Start Level 03 →](levels/03-chunking-strategies/README.md)

---

### 🔄 Level 04: Full Pipeline
Build a complete end-to-end RAG system with PDF processing, caching, and multi-document search.

**What you'll learn**: PDF extraction, smart caching, production pipeline, multi-document retrieval

[Start Level 04 →](levels/04-full-pipeline/README.md)

---

### 🤖 Level 05: Agentic RAG
Implement an intelligent AI agent that decides when and how to search documents.

**What you'll learn**: Tool calling, autonomous reasoning, agentic patterns, streaming responses

[Start Level 05 →](levels/05-agentic-rag/README.md)

---

## Configuration

### Setting Up Your API Keys

1. **Copy the example environment file**:
```bash
cp .env.example .env
```

2. **Edit `.env` and add your API credentials**:
```bash
# Embedding API (Levels 01-05)
EMBEDDING_API_KEY=your-api-key-here
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# LLM API (Level 05 only)
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3-coder-30b-a3b-instruct

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Shared Components

This repository uses a **shared component architecture** to avoid code duplication:

- **`levels/shared/`** contains reusable components used across all levels
- Each level builds on these shared components while introducing new concepts
- Promotes clean, maintainable code and consistent patterns

## What Makes This Different?

- ✅ **Production-ready code**, not toy examples
  - Error handling, type hints, docstrings
  - Modular architecture you can adapt
  - Clean code following best practices

- ✅ **Cost-conscious** with optimization strategies
  - Detailed cost breakdowns for every level
  - Tips for reducing API expenses
  - Caching and efficiency recommendations

- ✅ **Progressive complexity** - one concept per level
  - Start simple, add sophistication gradually
  - Each level builds on previous knowledge
  - Clear explanations of when and why

- ✅ **Progressive complexity** - from basics to production
  - Start with fundamentals and build to complete systems
  - Each level introduces exactly one new concept
  - Clear learning path with detailed documentation

## Repository Structure

```
RAG/
├── README.md                        # You are here
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Qdrant vector database
├── .env.example                     # API configuration template
└── levels/                          # Progressive learning levels
    │
    ├── shared/                      # Shared components
    │   ├── config.py                # Centralized configuration
    │   ├── embedder.py              # Embedding generation
    │   ├── vector_store.py          # Qdrant integration
    │   ├── document_loader.py       # Document utilities
    │   ├── similarity.py            # Similarity metrics
    │   └── output_manager.py        # Results formatting
    │
    ├── 01-basic-vector-search/      # Fundamentals
    │   ├── main.py
    │   └── output/
    │
    ├── 02-semantic-vs-exact/        # Hybrid retrieval
    │   ├── main.py
    │   ├── citizens/                # Sample documents
    │   └── utils/
    │
    ├── 03-chunking-strategies/      # Document splitting
    │   ├── main.py
    │   └── utils/
    │
    ├── 04-full-pipeline/            # Complete RAG system
    │   ├── main.py
    │   ├── documents/               # PDF inputs
    │   └── utils/
    │
    └── 05-agentic-rag/              # Intelligent agent
        ├── main.py
        ├── core/                    # Agent framework
        ├── tools/                   # Tool implementations
        └── interface/               # UI components
```

## Learning Path

**Recommended progression**:

1. **Level 01** → Understand vector search fundamentals
2. **Level 02** → Learn when to use hybrid retrieval
3. **Level 03** → Master chunking strategies
4. **Level 04** → Build a complete production pipeline
5. **Level 05** → Implement intelligent agentic RAG

Each level takes 15-30 minutes to complete and builds on previous concepts.

## Common Issues

### Missing dependencies
```bash
pip install -r requirements.txt
```

### API authentication errors
Check your `.env` file has valid API keys for both embedding and LLM endpoints.

### Qdrant connection issues
```bash
docker-compose up -d  # Ensure Qdrant is running
```

### Rate limits
Wait briefly or reduce batch sizes. Consider using caching (implemented in Level 04).

## Next Steps

After completing this training:

### Production Considerations
- **Vector Databases**: Pinecone, Weaviate, or Qdrant Cloud for scale
- **Frameworks**: LangChain or LlamaIndex for rapid development
- **Monitoring**: Track retrieval quality, latency, and costs
- **Evaluation**: Implement metrics (MRR, NDCG, answer quality)
- **Security**: Access control, data privacy, PII handling

### Learning Resources
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [OpenAI: Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone: RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

**Ready to start?** → [Level 01: Basic Vector Search](levels/01-basic-vector-search/README.md)
