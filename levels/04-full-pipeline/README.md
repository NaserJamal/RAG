# Level 04: Full Pipeline RAG

## Overview

This level demonstrates a complete, production-ready RAG (Retrieval-Augmented Generation) pipeline from start to finish. You'll see how all components work together in a real-world scenario:

```
PDF Documents → Text Extraction → Chunking → Embedding → Semantic Search
```

## What You'll Learn

- **End-to-end workflow**: See how each component connects to form a complete system
- **Smart caching**: Embeddings are cached to avoid redundant API calls on subsequent runs
- **Document processing**: Extract and organize text from PDF documents
- **Chunking strategy**: Split documents into optimal-sized pieces for retrieval
- **Multi-document search**: Search across multiple documents simultaneously

## Pipeline Architecture

```
┌─────────────────┐
│  PDF Documents  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Extraction │  (PyMuPDF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  (Sliding window: 500 chars, 100 overlap)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  (OpenAI-compatible API + Caching)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Search │  (Cosine similarity)
└─────────────────┘
```

## Directory Structure

```
04-full-pipeline/
├── documents/              # Place your PDF files here
│   └── your_document.pdf
├── output/                 # Processed documents and results
│   ├── .cache/             # Embedding cache (persistent)
│   ├── your_document/      # Per-document output
│   │   ├── raw_text.txt    # Extracted text
│   │   ├── chunk_001.txt   # Individual chunks
│   │   ├── chunk_002.txt
│   │   ├── ...
│   │   ├── chunks_metadata.json
│   │   └── embeddings.json # Generated embeddings
│   └── search_history.jsonl # Search queries and results
├── utils/                  # Pipeline components
│   ├── pdf_extractor.py    # PDF text extraction
│   ├── chunker.py          # Text chunking logic
│   ├── embedder.py         # Embedding generation with caching
│   ├── search.py           # Semantic search engine
│   └── display.py          # Terminal UI utilities
├── main.py                 # Main pipeline orchestrator
└── README.md               # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install pymupdf numpy openai python-dotenv qdrant-client
   ```

2. **Configure environment**:
   Ensure your `.env` file has the required API keys:
   ```
   EMBEDDING_API_KEY=your_key_here
   EMBEDDING_BASE_URL=https://api.openai.com/v1
   EMBEDDING_MODEL=text-embedding-3-small
   ```

3. **Add PDF documents**:
   Place one or more PDF files in the `documents/` directory:
   ```bash
   cp /path/to/your/document.pdf levels/04-full-pipeline/documents/
   ```

## Running the Pipeline

```bash
cd levels/04-full-pipeline
python main.py
```

### First Run

On the first run, the pipeline will:

1. **Extract text** from all PDFs in `documents/`
2. **Create chunks** and save them as individual `.txt` files
3. **Generate embeddings** for each chunk (this may take time depending on document size)
4. **Build search index** in memory
5. **Start interactive search** mode

### Subsequent Runs

The pipeline intelligently caches embeddings, so subsequent runs will:

- ✓ Skip text extraction if output already exists
- ✓ **Load cached embeddings** instead of regenerating them
- ✓ Start search mode almost instantly

This makes it efficient to restart the program without re-processing everything!

## Using the Search Interface

Once the pipeline is running, you can:

1. **Enter search queries** to find relevant chunks across all documents
2. **Type 'stats'** to see pipeline statistics
3. **Press Enter** on an empty line to exit

Example queries:
```
Enter your query: What is machine learning?
Enter your query: Explain neural networks
Enter your query: stats
Enter your query: [Press Enter to exit]
```

## Understanding the Output

### Document Processing

For each PDF, you'll find:

```
output/
└── document_name/
    ├── raw_text.txt              # Full extracted text
    ├── chunk_001.txt             # First chunk
    ├── chunk_002.txt             # Second chunk
    ├── ...
    ├── chunks_metadata.json      # Chunking metadata
    └── embeddings.json           # All embeddings + metadata
```

### Search Results

Each search shows:
- **Rank**: Position in results (1-5)
- **Score**: Similarity score (0-1, higher is better)
- **Document**: Which document the chunk came from
- **Chunk ID**: Which chunk number
- **Preview**: First 200 characters of the chunk

Results are also saved to `output/search_history.jsonl` for later analysis.

## Key Features

### 1. Smart Caching

The `EmbeddingManager` caches embeddings using content hashes:
- Each chunk's text is hashed (SHA256)
- Embeddings are stored in `output/.cache/embeddings_cache.json`
- Identical text chunks reuse cached embeddings
- Drastically reduces API calls and cost

### 2. Normalized Naming

PDF filenames are normalized for filesystem compatibility:
- `My Document (2024).pdf` → `my_document_2024/`
- Spaces and special characters are replaced with underscores
- Consistent, predictable output structure

### 3. Sliding Window Chunking

Chunks overlap to preserve context across boundaries:
- **Chunk size**: 500 characters
- **Overlap**: 100 characters
- Ensures important information isn't split awkwardly

### 4. Multi-Document Search

Search across all indexed documents simultaneously:
- Results ranked by semantic similarity
- Document attribution in results
- Efficient cosine similarity computation using NumPy

## Pipeline Components

### PDFExtractor (`utils/pdf_extractor.py`)

Extracts text from PDFs using PyMuPDF:
- Page-by-page extraction
- Automatic text concatenation
- Error handling for corrupted PDFs

### TextChunker (`utils/chunker.py`)

Splits text into manageable chunks:
- Configurable chunk size and overlap
- Saves chunks to individual files
- Generates metadata for tracking

### EmbeddingManager (`utils/embedder.py`)

Handles embedding generation with caching:
- SHA256 content hashing
- Persistent cache on disk
- Integration with shared `Embedder` class
- Progress reporting

### SemanticSearchEngine (`utils/search.py`)

Performs vector similarity search:
- Cosine similarity computation
- NumPy-optimized operations
- Top-k retrieval
- Multi-document support

## Customization

You can adjust the pipeline parameters in `main.py`:

```python
# Change chunk size and overlap
self.chunker = TextChunker(chunk_size=500, overlap=100)

# Adjust number of search results
results = self.search_engine.search(query_embedding, top_k=5)
```

## Comparison with Previous Levels

| Feature | Level 01 | Level 02 | Level 03 | Level 04 |
|---------|----------|----------|----------|----------|
| Document Format | Text files | Text files | Text files | **PDFs** |
| Processing | Pre-chunked | Pre-chunked | Pre-chunked | **Full pipeline** |
| Chunking | None | None | Multiple strategies | **Integrated** |
| Caching | Basic | Basic | Basic | **Smart hash-based** |
| Search | Vector only | Hybrid | Vector | **Multi-doc vector** |
| Real-world Ready | ❌ | ❌ | ⚠️ | **✅** |

## Troubleshooting

### No documents found
```
✗ No PDF documents found in documents/
```
**Solution**: Add PDF files to the `documents/` directory

### API key error
```
✗ Configuration error: EMBEDDING_API_KEY not found
```
**Solution**: Check your `.env` file has valid credentials

### Embedding generation fails
```
✗ Error during search: ...
```
**Solution**: Verify your embedding API is accessible and your quota isn't exceeded

### Cache issues
If you want to regenerate embeddings:
```bash
rm -rf output/.cache
rm output/*/embeddings.json
```

## Next Steps

After mastering this level, proceed to:

**Level 05: Agentic RAG** - Add LLM-based query routing and response generation on top of this retrieval pipeline

## Questions?

This level demonstrates a production-ready RAG retrieval system. You now understand:
- ✓ How to process real-world documents (PDFs)
- ✓ The importance of caching for cost and performance
- ✓ How all RAG components integrate together
- ✓ Building efficient multi-document search systems

The foundation you've built here is essential for more advanced RAG applications!
