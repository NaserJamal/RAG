# Level 04: Agentic RAG

An AI agent with tool calling capabilities that can intelligently search and retrieve information from a document collection to answer user questions.

## Overview

This level demonstrates **agentic RAG** - a system where an AI agent autonomously decides when to search for information, formulates effective queries, and uses retrieved context to provide accurate answers. Unlike traditional RAG systems that always retrieve, the agent makes intelligent decisions about when and how to use its search tool.

## Features

- **Intelligent Tool Use**: Agent decides when to search based on user query
- **Semantic Search**: Leverages Qdrant vector database for semantic document retrieval
- **Clean Architecture**: Separation of concerns between agent logic, tools, and interface
- **Interactive Sessions**: Multi-turn conversations with context
- **Streaming Responses**: Real-time LLM output
- **Robust Error Handling**: Graceful degradation when searches fail

## Project Structure

```
04-agentic-rag/
├── main.py                          # Entry point
│
├── core/                            # Core framework
│   ├── agent/                       # Agent orchestration
│   │   ├── loop.py                 # Main agent loop
│   │   └── executor.py             # Tool execution
│   │
│   └── tool_system/                 # Tool infrastructure
│       └── registry.py             # Tool registration system
│
├── tools/                           # Tool implementations
│   └── qdrant_search.py            # Semantic document search
│
├── interface/                       # User interface
│   ├── ui/
│   │   └── display.py              # Terminal output
│   │
│   └── conversation/
│       └── manager.py              # Conversation flow
│
├── requirements.txt
└── README.md
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Qdrant** (if not already running):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## Configuration

Edit the root `.env` file (at the repository root) and add the LLM API configuration:

```bash
# LLM API Configuration (Level 04: Agentic RAG)
# OpenAI-compatible endpoint for the AI agent
LLM_API_KEY=sk-or-v1-YOUR-API-KEY
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3-coder-30b-a3b-instruct

# Embedding API (already configured in earlier levels)
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant Vector Database (already configured in earlier levels)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**Note**: This level uses the root `.env` file shared across all levels, so if you've completed earlier levels, you only need to add the `LLM_*` variables.

## Usage

Navigate to the level directory and run:

```bash
cd levels/04-agentic-rag
python main.py
```

The agent will:
1. Initialize the document collection (first run only)
2. Start an interactive session
3. Use the search tool when needed to answer your questions

### Example Session

```
────────────────────────────────────────────────────────────

You: What is semantic chunking?

💭 Agent thinking...

🔧 Tool Call: search_documents
   Arguments: {
     "query": "semantic chunking",
     "top_k": 3
   }
   ✓ Result: {
     "query": "semantic chunking",
     "result_count": 3,
     "results": [
       {
         "document_id": "chunking-strategies/semantic.txt",
         "relevance_score": 0.89,
         "content": "Semantic chunking divides text based on meaning..."
       }
     ]
   }

💭 Agent thinking...

🤖 Assistant:
Semantic chunking is an advanced technique that divides text based on
semantic meaning rather than fixed sizes. It groups related content
together by analyzing the semantic similarity between sentences...

────────────────────────────────────────────────────────────
```

## How It Works

### Agent Flow

1. **User Query** → Agent receives the question
2. **Planning** → LLM decides if search is needed
3. **Tool Execution** → If needed, searches document collection
4. **Context Integration** → LLM receives search results
5. **Response** → Agent formulates answer using retrieved context

### Search Tool

The `search_documents` tool:
- Generates query embeddings using OpenAI API
- Performs semantic search in Qdrant
- Returns relevant documents with scores
- Handles initialization and error cases gracefully

### Key Differences from Traditional RAG

| Traditional RAG | Agentic RAG |
|----------------|-------------|
| Always retrieves for every query | Decides when retrieval is needed |
| Single retrieval pass | Can make multiple searches |
| Fixed query | Can reformulate queries |
| No tool selection | Chooses appropriate tools |

## Adding New Tools

Create a new file in `tools/` and use the decorator pattern:

```python
from core.tool_system import registry

@registry.register(
    name="your_tool_name",
    description="What your tool does",
    parameters={
        "type": "object",
        "required": ["param1"],
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of parameter"
            }
        }
    }
)
def your_tool_name(param1: str) -> dict:
    """Your tool implementation."""
    return {"result": f"Processed {param1}"}
```

Import it in `tools/__init__.py` and it's automatically available!

## Available Tool

- **`search_documents`**: Semantic search over the document collection
  - Parameters: `query` (string), `top_k` (integer, 1-10)
  - Returns: Relevant documents with content and relevance scores

## Technologies Used

- **OpenRouter**: LLM API for agent reasoning
- **OpenAI API**: Text embeddings for semantic search
- **Qdrant**: Vector database for document storage and retrieval
- **Rich**: Terminal UI formatting
- **Tool Calling**: Native function calling support

## Learning Outcomes

This level demonstrates:
- **Agentic patterns**: Autonomous decision-making in RAG
- **Tool calling**: LLM function calling for tool selection
- **Semantic search**: Vector-based document retrieval
- **Clean architecture**: Maintainable, extensible codebase
- **Error handling**: Robust production-ready patterns
- **Streaming**: Real-time response display

## Next Steps

- Add more tools (web search, calculator, etc.)
- Implement query reformulation for poor results
- Add retrieval quality evaluation
- Enable multi-step reasoning chains
- Implement answer verification

---

← [Back to Main README](../../README.md)
