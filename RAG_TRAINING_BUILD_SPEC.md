# RAG Training Repository - Build Specification

## Overview

You are tasked with building a progressive learning repository for teaching Retrieval-Augmented Generation (RAG) concepts. This repository should follow the proven organizational patterns from a successful OCR training repository while focusing on RAG fundamentals.

## Reference Repository

The organizational structure and patterns should be based on this OCR training repository:
https://github.com/naserjamal/simple-ocr (or the local path if available)

Key aspects to study and replicate:
- Progressive level structure (01-XX naming)
- Consistent README.md patterns
- Modular code architecture for complex levels
- Shared resources at root level
- Self-contained, runnable examples at each level

## Repository Structure

```
simple-rag/
├── README.md                        # Master landing page
├── requirements.txt                 # Centralized dependencies
├── .env.example                     # API configuration template
├── .gitignore                       # Standard Python + custom patterns
├── documents/                       # Shared training documents
│   ├── company-kb/                  # Sample company knowledge base docs
│   ├── technical-docs/              # Technical documentation samples
│   ├── news-articles/               # Current events articles
│   └── mixed-content/               # Various document types
└── levels/
    ├── 01-basic-vector-search/
    ├── 02-hybrid-search/
    ├── 03-chunking-strategies/
    └── 04-agentic-rag/              # Placeholder for future implementation
```

## Level Breakdown

### Level 01: Basic Vector Search
**Time**: 15 minutes | **Cost**: $0.001-0.01

**What You'll Learn**:
- Text embedding fundamentals with OpenAI embeddings
- Vector similarity search using cosine similarity
- Basic retrieval pipeline (embed → store → search → retrieve)
- When simple vector search works and when it fails

**Code Complexity**: Single `main.py` file (~80-100 lines)

**Key Concepts**:
- Text embeddings (text-embedding-3-small)
- Cosine similarity calculation
- Simple in-memory vector store (numpy arrays)
- Top-k retrieval

**Required Functionality**:
1. Load sample documents from `../../documents/`
2. Generate embeddings using OpenAI API
3. Store embeddings in numpy array with document mapping
4. Accept user query
5. Embed query and compute similarity scores
6. Return top-k most relevant documents
7. Output results to `output/results.json`

**Output Files**:
- `output/results.json`: Retrieved documents with similarity scores
- `output/query_log.txt`: Query and retrieved chunks

**Dependencies**:
- `openai>=1.0.0`
- `numpy>=1.24.0`
- `python-dotenv>=1.0.0`

---

### Level 02: Hybrid Search
**Time**: 25 minutes | **Cost**: $0.001-0.02

**What You'll Learn**:
- Combining semantic search (embeddings) with keyword search (BM25)
- Reciprocal Rank Fusion (RRF) for merging results
- When to use hybrid vs pure vector search
- Trade-offs between semantic understanding and exact matching

**Code Complexity**: Modular structure with `utils/` package

**Structure**:
```
02-hybrid-search/
├── README.md
├── main.py                    # Entry point (30-40 lines)
├── output/
└── utils/
    ├── __init__.py
    ├── config.py              # Configuration constants
    ├── embedder.py            # Embedding logic
    ├── bm25_retriever.py      # Keyword search
    ├── vector_retriever.py    # Semantic search
    └── hybrid_retriever.py    # RRF fusion
```

**Key Concepts**:
- BM25 algorithm for keyword matching
- Reciprocal Rank Fusion (RRF)
- Score normalization
- Configurable weighting between semantic and keyword

**Required Functionality**:
1. Implement BM25 indexing and retrieval
2. Implement vector similarity search (from Level 01)
3. Apply RRF to merge results from both retrievers
4. Configurable alpha parameter for semantic vs keyword weighting
5. Comparison output showing: vector-only results, BM25-only results, hybrid results

**Output Files**:
- `output/vector_results.json`
- `output/bm25_results.json`
- `output/hybrid_results.json`
- `output/comparison.txt`: Side-by-side comparison

**Dependencies**:
- Previous dependencies
- `rank-bm25>=0.2.2`

---

### Level 03: Chunking Strategies
**Time**: 30 minutes | **Cost**: $0.005-0.03

**What You'll Learn**:
- Different text chunking approaches (fixed, recursive, semantic)
- Impact of chunk size on retrieval quality
- Overlap strategies for context preservation
- When to use which chunking strategy

**Code Complexity**: Modular with multiple chunking implementations

**Structure**:
```
03-chunking-strategies/
├── README.md
├── main.py                         # Interactive chunker comparison
├── output/
└── utils/
    ├── __init__.py
    ├── config.py
    ├── embedder.py                 # Reusable from Level 02
    ├── fixed_chunker.py            # Fixed-size chunks
    ├── recursive_chunker.py        # Recursive character splitting
    ├── semantic_chunker.py         # Semantic boundary detection
    └── chunk_evaluator.py          # Compare chunking strategies
```

**Key Concepts**:
- Fixed-size chunking with overlap
- Recursive character text splitting (by paragraph, sentence, etc.)
- Semantic chunking (using embedding similarity to detect topic boundaries)
- Chunk metadata (position, parent document, neighbors)

**Required Functionality**:
1. Implement three chunking strategies:
   - Fixed: Configurable size (e.g., 512 tokens) with overlap (e.g., 50 tokens)
   - Recursive: Split by \n\n, then \n, then sentences, then characters
   - Semantic: Detect topic shifts using embedding similarity between sentences
2. Interactive mode: User selects strategy and document
3. Visualization of chunks (show boundaries, sizes, overlap regions)
4. Comparative evaluation: Run the same query against all three strategies
5. Metrics: Retrieval quality, chunk count, average chunk size

**Output Files**:
- `output/fixed_chunks.json`
- `output/recursive_chunks.json`
- `output/semantic_chunks.json`
- `output/comparison_metrics.json`
- `output/chunk_visualization.txt`: ASCII visualization of chunks

**Dependencies**:
- Previous dependencies
- `tiktoken>=0.5.0` (for token counting)
- `nltk>=3.8.0` (for sentence splitting)

---

### Level 04: Agentic RAG
**Time**: TBD | **Cost**: TBD

**Status**: PLACEHOLDER

This level will be implemented separately with a custom agentic RAG implementation. For now, create the directory structure with a placeholder README.md explaining this is coming soon.

**Placeholder Structure**:
```
04-agentic-rag/
├── README.md                  # "Coming Soon" placeholder
└── .gitkeep
```

**Placeholder README Content**:
```markdown
# Level 04: Agentic RAG

## Status: Coming Soon

This level will cover advanced agentic RAG patterns including:
- Self-correcting retrieval
- Tool use and function calling
- Multi-step reasoning
- Query planning and decomposition
- Iterative refinement

Check back soon for the full implementation!
```

---

## Code Quality Requirements

### 1. Clean Code Principles
- **Simplicity First**: Start with simplest solution, add complexity only when needed
- **Single Responsibility**: Each function/class does ONE thing well
- **Clear Naming**: Variable and function names should be self-documenting
- **No Magic Numbers**: Use named constants in `config.py`

### 2. Modularity Requirements

**Simple Levels (01)**: Single file is acceptable if under 150 lines

**Complex Levels (02-03)**: Must use modular structure:
```python
# main.py - Clean entry point
from utils.embedder import Embedder
from utils.retriever import Retriever

def main():
    # 20-40 lines of orchestration logic
    embedder = Embedder()
    retriever = Retriever()
    # ...
```

**utils/ Package Pattern**:
- `config.py`: All constants, API keys loading, configuration
- `embedder.py`: Embedding generation logic
- `*_retriever.py`: Retrieval implementations
- `*_chunker.py`: Chunking implementations
- Each module should be independently testable

### 3. Error Handling
```python
# Good: Graceful error handling with helpful messages
try:
    embeddings = embedder.embed(texts)
except openai.AuthenticationError:
    print("❌ OpenAI API key is invalid. Check your .env file.")
    sys.exit(1)
except openai.RateLimitError:
    print("⚠️ Rate limit reached. Waiting 60 seconds...")
    time.sleep(60)
    # Retry logic
```

### 4. Type Hints (Levels 02+)
```python
from typing import List, Dict, Tuple
import numpy as np

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    # Implementation
```

### 5. Docstrings (All Functions)
```python
def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: List of ranked document IDs from different retrievers
        k: Constant for RRF formula (default: 60)

    Returns:
        List of (doc_id, score) tuples sorted by fused score
    """
```

### 6. Configuration Management
```python
# utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Retrieval Configuration
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

### 7. Resource Cleanup
```python
# Use context managers where appropriate
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Documentation Requirements

### Master README.md Structure
```markdown
# Simple RAG - Progressive RAG Training

Learn Retrieval-Augmented Generation (RAG) through hands-on, production-ready examples.

## What You'll Build

[2-3 paragraph overview of RAG and why it matters]

## Prerequisites

- Python 3.8+
- OpenAI API key (or OpenRouter for cheaper alternatives)
- Basic understanding of Python and APIs

## Quick Start

[Exact steps to get started - copy/paste friendly]

## Levels Overview

### 🎯 Level 01: Basic Vector Search
[1-2 sentence description]
- **Time**: 15 minutes | **Cost**: ~$0.005

### 🔀 Level 02: Hybrid Search
[1-2 sentence description]
- **Time**: 25 minutes | **Cost**: ~$0.01

### ✂️ Level 03: Chunking Strategies
[1-2 sentence description]
- **Time**: 30 minutes | **Cost**: ~$0.02

### 🤖 Level 04: Agentic RAG
[Coming Soon]

## Installation

[Step-by-step installation instructions]

## Configuration

[How to set up .env file]

## What Makes This Different?

- ✅ Production-ready code, not toy examples
- ✅ Cost-conscious with optimization strategies
- ✅ Progressive complexity - one concept per level
- ✅ Modular architecture you can adapt
- ✅ Real-world sample documents

## Repository Structure

[Tree view of the repository]

## Contributing

[How to contribute or provide feedback]

## License

MIT License
```

### Level README.md Structure (Consistent Across All Levels)
```markdown
# Level XX: [Title]

## What You'll Learn

- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]
- [Bullet point 4]

## Concept Explanation

[2-3 paragraphs explaining WHY this level exists and the problem it solves]

## How It Works

[Architecture overview with ASCII diagram if helpful]

### Code Breakdown

[Key code snippets with inline explanations]

## Prerequisites

- Completed: [Previous levels if any]
- API Key: [Required services]
- Python packages: [Installed via requirements.txt]

## Running the Example

```bash
# Step 1: Navigate to level directory
cd levels/XX-level-name

# Step 2: Run the example
python main.py

# Step 3: View the output
cat output/results.json
```

## Configuration & Tuning

[Adjustable parameters and their effects]

```python
# In utils/config.py
TOP_K = 5              # Number of results to retrieve
CHUNK_SIZE = 512       # Tokens per chunk
```

## Common Issues & Solutions

### Issue 1: [Problem]
**Solution**: [Fix]

### Issue 2: [Problem]
**Solution**: [Fix]

## Key Takeaways

- ✅ [Takeaway 1]
- ✅ [Takeaway 2]
- ✅ [Takeaway 3]
- ✅ [Takeaway 4]
- ✅ [Takeaway 5]

## Real-World Applications

- [Use case 1]
- [Use case 2]
- [Use case 3]

## Going Further

[Ideas for extending this level's concepts]

## What's Next?

➡️ Continue to [Level XX: Name](../XX-name/README.md)

---

← [Back to Main README](../../README.md)
```

---

## Sample Documents Requirements

Create a diverse set of sample documents in `documents/` directory:

### documents/company-kb/ (5-7 documents)
- Company policies (HR, vacation, expenses)
- Product documentation
- Internal wiki pages
- FAQ documents

**Characteristics**:
- 500-2000 words each
- Structured with headers
- Mix of paragraphs and bullet points
- Some overlapping information (to test retrieval precision)

### documents/technical-docs/ (5-7 documents)
- API documentation
- Code tutorials
- Technical specifications
- Installation guides

**Characteristics**:
- Code snippets included
- Step-by-step instructions
- Technical jargon
- Version-specific information

### documents/news-articles/ (5-7 documents)
- Current events articles
- Opinion pieces
- News summaries
- Topic-related articles (e.g., all about AI)

**Characteristics**:
- 300-1000 words each
- Narrative style
- Dates and named entities
- Similar topics but different perspectives

### documents/mixed-content/ (3-5 documents)
- Mixed-format documents
- Long documents (3000+ words)
- Short documents (100-200 words)
- Dense technical content

**Purpose**: Test edge cases and chunking strategies

---

## Dependencies & Configuration

### requirements.txt
```txt
openai>=1.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
rank-bm25>=0.2.2
tiktoken>=0.5.0
nltk>=3.8.0
```

### .env.example
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Use OpenRouter for cheaper alternatives
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# OPENAI_API_KEY=your_openrouter_key_here

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small
```

### .gitignore
```gitignore
# Environment
.env
venv/
env/
*.pyc
__pycache__/

# Outputs
levels/*/output/*
!levels/*/output/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Python
*.egg-info/
dist/
build/
```

---

## Cost Transparency

Every level README must include cost estimates:

### Cost Calculation Template
```markdown
## Cost Breakdown

### Embedding Costs
- Model: text-embedding-3-small
- Rate: $0.00002 per 1K tokens
- Average document size: 500 tokens
- 20 documents: ~$0.0002

### LLM Costs (if applicable)
- Model: gpt-4o-mini
- Rate: $0.150 per 1M input tokens
- Average query: 100 tokens
- 10 queries: ~$0.00015

### Total Estimated Cost
- One-time setup: ~$0.0002
- Per query: ~$0.00015
- **10 queries: ~$0.002 (less than a cent!)**

### Optimization Tips
- [Tip 1]
- [Tip 2]
```

---

## Development Workflow

### Step 1: Directory Structure
Create all directories and placeholder files first

### Step 2: Sample Documents
Populate `documents/` with realistic sample content

### Step 3: Level 01 Implementation
Build simplest level first, get it fully working

### Step 4: Level 01 Documentation
Write complete README before moving on

### Step 5: Test & Validate
Run Level 01 end-to-end, verify outputs

### Step 6: Levels 02-03
Repeat steps 3-5 for each level

### Step 7: Master README
Write comprehensive landing page

### Step 8: Final Polish
- Consistent formatting
- All links working
- All costs calculated
- Troubleshooting sections complete

---

## Success Criteria

A level is complete when:

✅ Code runs without errors on fresh environment
✅ All output files are generated correctly
✅ README has all required sections
✅ Cost estimates are accurate
✅ Common issues are documented with solutions
✅ Code follows modularity requirements
✅ Type hints and docstrings present (Levels 02+)
✅ Configuration is externalized to config.py
✅ Error handling is graceful and helpful

---

## Additional Guidelines

### Console Output
Provide helpful progress indicators:
```python
print("📄 Loading documents...")
print(f"✅ Loaded {len(documents)} documents")
print("🔢 Generating embeddings...")
print(f"✅ Generated {len(embeddings)} embeddings")
print("🔍 Searching for relevant documents...")
print(f"✅ Found {len(results)} results")
```

### Visual Feedback
For comparison tasks, use tables:
```
| Strategy   | Chunks | Avg Size | Top Result Score |
|------------|--------|----------|------------------|
| Fixed      | 45     | 512      | 0.87            |
| Recursive  | 38     | 612      | 0.91            |
| Semantic   | 28     | 743      | 0.89            |
```

### JSON Output Format
Consistent structure:
```json
{
  "query": "What is the vacation policy?",
  "timestamp": "2025-11-03T10:30:00Z",
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "document": "company-kb/vacation-policy.txt",
      "chunk_id": "chunk_3",
      "text": "..."
    }
  ],
  "metadata": {
    "retrieval_method": "hybrid",
    "total_documents": 20,
    "retrieval_time_ms": 145
  }
}
```

---

## Reference Patterns from OCR Repository

Study these aspects from the reference OCR repository:

1. **README Structure**: Look at `levels/02-hybrid-vlm-ocr/README.md` for excellent documentation patterns
2. **Modular Architecture**: Examine `levels/08-markdown-reconstruction/utils/` for package structure
3. **Configuration**: See how `utils/config.py` is used in complex levels
4. **Error Handling**: Review how API errors are handled gracefully
5. **Cost Transparency**: Notice how every level shows explicit cost breakdowns
6. **Visual Output**: Check `levels/04-element-detection/` for visualization examples
7. **Progressive Complexity**: Observe the 6-line `main.py` in Level 01 vs modular Level 08

---

## Final Notes

- **Don't overthink**: Start simple, add complexity only when needed
- **Test as you go**: Each level should be runnable before moving to the next
- **Documentation first**: Write README before code to clarify thinking
- **Real examples**: Use realistic documents and queries
- **Cost conscious**: Show costs and optimization strategies
- **Production ready**: This should be code developers can actually use

Good luck building! This should be a repository that developers can learn from AND use in production.
