# Shared Components for RAG Training

This directory contains shared, reusable components used across all RAG training levels. The components are designed to be modular, robust, and easy to use.

## Architecture Overview

```
shared/
├── __init__.py           # Package exports
├── config.py             # Centralized configuration
├── embedder.py           # OpenAI embedding generation
├── vector_store.py       # Vector storage with Qdrant
├── document_loader.py    # Document loading utilities
├── similarity.py         # Similarity metrics
└── output_manager.py     # Results formatting and saving
```

## Components

### Config (`config.py`)

Centralized configuration management for all training levels.

**Features:**
- Environment variable management
- Path resolution for levels, documents, and outputs
- Qdrant connection settings
- OpenAI API configuration

**Usage:**
```python
from shared import Config

# Validate configuration
Config.validate()

# Access configuration
api_key = Config.EMBEDDING_API_KEY
base_url = Config.EMBEDDING_BASE_URL
qdrant_url = Config.get_qdrant_url()
output_path = Config.get_output_path("01-basic-vector-search")
```

**Configuration Variables:**
- `EMBEDDING_API_KEY` - Required API key for embeddings
- `EMBEDDING_BASE_URL` - Required base URL for OpenAI-compatible API
- `EMBEDDING_MODEL` - Default: "text-embedding-3-small"
- `QDRANT_HOST` - Default: "localhost"
- `QDRANT_PORT` - Default: 6333
- `QDRANT_API_KEY` - Optional for cloud deployments

---

### Embedder (`embedder.py`)

OpenAI API-based embedding generation with batch support.

**Features:**
- Batch text embedding
- Single query embedding
- Automatic error handling
- Configurable model selection

**Usage:**
```python
from shared import Embedder

embedder = Embedder()

# Batch embeddings
texts = ["document 1", "document 2", "document 3"]
embeddings = embedder.embed(texts)  # Returns numpy array (n_texts, embedding_dim)

# Single query embedding
query = "What is the policy?"
query_embedding = embedder.embed_query(query)  # Returns 1D numpy array

# Get embedding dimension
dim = embedder.get_embedding_dimension()  # Returns 1536
```

---

### VectorStore (`vector_store.py`)

Abstract base class and Qdrant implementation for vector storage and retrieval.

**Features:**
- Abstract interface for vector stores
- Qdrant client integration
- Collection management
- Metadata support
- Batch operations
- Cosine similarity search

**Usage:**
```python
from shared import QdrantVectorStore
import numpy as np

vector_store = QdrantVectorStore()

# Create collection
vector_store.create_collection(
    collection_name="my_collection",
    vector_dim=1536
)

# Add vectors
vectors = np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]])
ids = ["doc1", "doc2"]
metadata = [{"content": "text1"}, {"content": "text2"}]

vector_store.add_vectors(
    collection_name="my_collection",
    vectors=vectors,
    ids=ids,
    metadata=metadata
)

# Search
query_vector = np.array([0.15, 0.25, ...])
results = vector_store.search(
    collection_name="my_collection",
    query_vector=query_vector,
    top_k=5
)
# Returns: [(doc_id, score), ...]
```

---

### Document Loader (`document_loader.py`)

Utilities for loading text documents from the documents directory.

**Features:**
- Recursive directory traversal
- Text file loading
- Consistent document structure
- Error handling

**Usage:**
```python
from shared import load_documents
from pathlib import Path

documents_path = Path("documents")
documents = load_documents(documents_path)

# Returns list of dicts:
# [
#   {
#     "id": "category/filename.txt",
#     "path": "/full/path/to/file.txt",
#     "content": "file contents..."
#   },
#   ...
# ]
```

---

### Similarity (`similarity.py`)

Efficient similarity computation utilities.

**Features:**
- Cosine similarity
- Euclidean distance
- Dot product similarity
- Batch operations
- Numpy-optimized

**Usage:**
```python
from shared import cosine_similarity
import numpy as np

query = np.array([0.1, 0.2, 0.3])
documents = np.array([
    [0.15, 0.25, 0.35],
    [0.05, 0.15, 0.25],
    [0.2, 0.3, 0.4]
])

similarities = cosine_similarity(query, documents)
# Returns: array([0.999..., 0.987..., 0.999...])
```

---

### Output Manager (`output_manager.py`)

Consistent output formatting and saving across all levels.

**Features:**
- JSON result saving
- Text file generation
- Human-readable formatting
- Search result formatting
- Comparison formatting
- Statistics formatting

**Usage:**
```python
from shared import OutputManager
from pathlib import Path

output_manager = OutputManager(Path("output"))

# Save JSON results
results = {
    "query": "test query",
    "results": [{"rank": 1, "score": 0.95}]
}
output_manager.save_results("search_results", results)

# Save text
content = "This is my output text"
output_manager.save_text("output.txt", content)

# Format search results
formatted = output_manager.format_search_results(
    query="test query",
    results=[{"rank": 1, "score": 0.95, "document_id": "doc1"}],
    title="My Search Results"
)

# Save formatted text
output_manager.save_text("formatted_results.txt", formatted)
```

---

## Vector Store: Qdrant

All levels use **Qdrant** as the vector database solution.

### Why Qdrant?

- **Performance**: Fast similarity search with HNSW indexing
- **Scalability**: Handles large-scale vector collections
- **Metadata**: Rich filtering and metadata support
- **Docker**: Easy deployment with docker-compose
- **Python Client**: Clean, well-documented Python API

### Setup

1. Start Qdrant with Docker:
```bash
docker-compose up -d
```

2. Verify Qdrant is running:
```bash
curl http://localhost:6333/health
```

3. Access Qdrant UI:
```
http://localhost:6333/dashboard
```

### Qdrant Collections

Each level creates its own collection:
- Level 01: `level_01_basic_search`
- Level 02: `level_02_hybrid_search`
- Level 03: Uses chunking without direct vector storage

---

## Environment Configuration

Create a `.env` file in the project root:

```bash
# Required - Embedding API (OpenAI-compatible)
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=your_embedding_base_url_here
EMBEDDING_MODEL=text-embedding-3-small

# Optional - Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=  # Only for Qdrant Cloud
QDRANT_USE_HTTPS=false
```

---

## Design Principles

### 1. Modularity
Each component has a single, well-defined responsibility and can be used independently.

### 2. Consistency
All levels use the same interfaces and patterns for common operations.

### 3. Extensibility
Abstract base classes (like `VectorStore`) allow for easy addition of new implementations.

### 4. Robustness
- Input validation
- Error handling
- Type hints
- Comprehensive docstrings

### 5. Clean Code
- PEP 8 compliant
- Clear naming conventions
- Minimal dependencies
- Well-documented

---

## Usage Patterns

### Basic Pattern (Level 01)
```python
from shared import Config, Embedder, QdrantVectorStore, load_documents

# Setup
Config.validate()
embedder = Embedder()
vector_store = QdrantVectorStore()

# Load and index
documents = load_documents(Config.get_documents_path("01-basic-vector-search"))
embeddings = embedder.embed([doc["content"] for doc in documents])
vector_store.create_collection("my_collection", embedder.get_embedding_dimension())
vector_store.add_vectors("my_collection", embeddings, [doc["id"] for doc in documents])

# Search
query_embedding = embedder.embed_query("my query")
results = vector_store.search("my_collection", query_embedding, top_k=5)
```

### With Output Management
```python
from shared import OutputManager

output_manager = OutputManager(Config.get_output_path("01-basic-vector-search"))

# Save results
output_manager.save_results("search_results", {
    "query": "my query",
    "results": results
})

# Format and save readable output
formatted = output_manager.format_search_results("my query", results)
output_manager.save_text("results.txt", formatted)
```

---

## Integration with Levels

Each level imports shared components while maintaining level-specific logic:

### Level 01: Basic Vector Search
- Uses: All shared components
- Unique: Simple end-to-end demo

### Level 02: Hybrid Search
- Uses: Embedder, Config, OutputManager
- Unique: BM25Retriever, HybridRetriever, RRF fusion

### Level 03: Chunking Strategies
- Uses: Embedder, Config, OutputManager, load_documents
- Unique: FixedChunker, RecursiveChunker, SemanticChunker, ChunkEvaluator

### Level 04: Agentic RAG (Future)
- Will use: All shared components
- Unique: Agent logic, tool use, multi-step reasoning

---

## Development Guidelines

### Adding New Shared Components

1. Create module in `levels/shared/`
2. Implement with clear interface and docstrings
3. Add comprehensive error handling
4. Export from `__init__.py`
5. Update this README
6. Add usage examples

### Modifying Existing Components

1. Ensure backward compatibility
2. Update docstrings
3. Update README if interface changes
4. Test across all levels

### Testing

Before committing changes:
1. Start Qdrant: `docker-compose up -d`
2. Test Level 01: `cd levels/01-basic-vector-search && python main.py`
3. Test Level 02: `cd levels/02-hybrid-search && python main.py`
4. Test Level 03: `cd levels/03-chunking-strategies && python main.py`

---

## API Reference

### Config Class

| Method | Description | Returns |
|--------|-------------|---------|
| `validate()` | Validate required config | None |
| `get_qdrant_url()` | Get Qdrant connection URL | str |
| `get_level_path(level_name)` | Get level directory path | Path |
| `get_documents_path(level_name)` | Get documents path for level | Path |
| `get_output_path(level_name)` | Get/create output path | Path |

### Embedder Class

| Method | Description | Returns |
|--------|-------------|---------|
| `embed(texts)` | Generate embeddings for texts | np.ndarray |
| `embed_query(query)` | Generate embedding for query | np.ndarray |
| `get_embedding_dimension()` | Get embedding dimension | int |

### QdrantVectorStore Class

| Method | Description | Returns |
|--------|-------------|---------|
| `create_collection(name, dim)` | Create vector collection | None |
| `add_vectors(name, vectors, ids, metadata)` | Add vectors to collection | None |
| `search(name, query, top_k, filter)` | Search for similar vectors | List[Tuple] |
| `delete_collection(name)` | Delete collection | None |
| `collection_exists(name)` | Check if collection exists | bool |
| `get_vector(name, id)` | Get vector by ID | np.ndarray |
| `get_metadata(name, id)` | Get metadata by ID | Dict |

### OutputManager Class

| Method | Description | Returns |
|--------|-------------|---------|
| `save_results(filename, results)` | Save results to JSON | Path |
| `save_text(filename, content)` | Save text file | Path |
| `format_search_results(query, results, title)` | Format search results | str |
| `format_comparison(query, comparisons)` | Format method comparison | str |
| `format_statistics(stats, title)` | Format statistics | str |
| `save_multiple_outputs(outputs, prefix)` | Save multiple outputs | List[Path] |

---

## Troubleshooting

### Qdrant Connection Issues

```python
# Check if Qdrant is running
curl http://localhost:6333/health

# Restart Qdrant
docker-compose restart qdrant
```

### Import Errors

Make sure the shared module is in the path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import Config, Embedder
```

### Embedding API Errors

- Check API key: `echo $EMBEDDING_API_KEY`
- Check base URL: `echo $EMBEDDING_BASE_URL`
- Verify `.env` file exists
- Check API quota/billing

---

## Future Enhancements

- [ ] Add support for other embedding providers (Cohere, HuggingFace)
- [ ] Implement caching for embeddings
- [ ] Add async/parallel processing for large datasets
- [ ] Support for other vector stores (Pinecone, Weaviate, ChromaDB)
- [ ] Monitoring and logging utilities
- [ ] Batch processing utilities
- [ ] Query preprocessing utilities

---

## Contributing

When adding features to shared components:

1. Follow existing patterns and conventions
2. Add comprehensive docstrings
3. Include usage examples
4. Update this README
5. Test across all levels
6. Keep components focused and modular

---

## License

This is part of the RAG training materials.
