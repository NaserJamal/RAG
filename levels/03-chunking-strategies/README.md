# Level 03: Chunking Strategies

## What You'll Learn

- Different text chunking approaches (fixed, recursive, semantic)
- Impact of chunk size on retrieval quality
- Overlap strategies for context preservation
- When to use which chunking strategy

## Concept Explanation

So far, we've treated documents as monolithic blocks of text. But what happens when documents are thousands of words long? Embedding an entire 10,000-word document into a single vector loses granularity – specific details get averaged out in the embedding space.

**Chunking** solves this by splitting documents into smaller, focused segments. Each chunk is embedded separately, allowing for more precise retrieval. Instead of retrieving an entire document, you retrieve the most relevant paragraphs or sections.

However, chunking introduces new questions:
- How large should chunks be?
- Should chunks overlap?
- Where should we split text?
- At fixed intervals? Paragraph boundaries? Semantic topic shifts?

This level implements three fundamentally different chunking strategies and compares their effectiveness.

## How It Works

### Three Chunking Strategies

#### 1. Fixed-Size Chunking
Split text into chunks of fixed token length with overlap.

**Pros**:
- Simple and predictable
- Consistent chunk sizes
- Fast to compute

**Cons**:
- Ignores document structure
- May split sentences or thoughts mid-way
- No semantic awareness

**Best for**: Uniform documents, when simplicity matters

#### 2. Recursive Character Splitting
Try splitting at document boundaries (paragraphs, sentences, words) in order.

**Pros**:
- Respects document structure
- Preserves sentences and paragraphs
- More readable chunks

**Cons**:
- Variable chunk sizes
- Still no semantic understanding
- May group unrelated paragraphs

**Best for**: Well-formatted documents with clear structure

#### 3. Semantic Chunking
Detect topic boundaries using embedding similarity between sentences.

**Pros**:
- Keeps related content together
- Splits at natural topic boundaries
- More coherent chunks

**Cons**:
- Computationally expensive (requires embedding all sentences)
- Variable chunk sizes
- Requires API calls

**Best for**: Documents with multiple topics, when quality matters most

### Architecture Overview

```
Document → Chunker → Chunks → Embeddings → Search
                                    ↓
                            Evaluation Metrics
```

### Code Breakdown

**Fixed Chunker** (`utils/fixed_chunker.py`):
```python
def chunk(self, text: str, doc_id: str) -> List[Dict]:
    """Split text into fixed-size chunks with overlap."""
    tokens = self.tokenizer.encode(text)
    start = 0

    while start < len(tokens):
        end = min(start + self.chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = self.tokenizer.decode(chunk_tokens)

        # Move start with overlap
        start = start + self.chunk_size - self.chunk_overlap
```

**Semantic Chunker** (`utils/semantic_chunker.py`):
```python
def chunk(self, text: str, doc_id: str) -> List[Dict]:
    """Split at semantic boundaries."""
    sentences = sent_tokenize(text)
    sentence_embeddings = self.embedder.embed(sentences)

    # Find where similarity drops (topic shift)
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(
            sentence_embeddings[i],
            sentence_embeddings[i + 1]
        )

        # Low similarity = topic boundary
        if sim < self.threshold:
            split_here()
```

## Prerequisites

- Completed: Levels 01-02 (for understanding retrieval concepts)
- API Key: OpenAI API key configured in `.env`
- Python packages: Installed via `requirements.txt` (includes `tiktoken` and `nltk`)

## Running the Example

```bash
# Step 1: Ensure dependencies are installed
pip install -r ../../requirements.txt

# Step 2: Navigate to level directory
cd levels/03-chunking-strategies

# Step 3: Run the example
python main.py

# Step 4: View the comparison
cat output/comparison_metrics.json
cat output/chunk_visualization.txt
```

The script will:
1. Load a sample document (the comprehensive ML systems design doc)
2. Chunk it using all three strategies
3. Evaluate each strategy with a test query
4. Generate comparison metrics and visualizations

## Configuration & Tuning

In `utils/config.py`:

```python
CHUNK_SIZE = 512            # Target chunk size in tokens
CHUNK_OVERLAP = 50          # Overlap between chunks
SEMANTIC_THRESHOLD = 0.5    # Similarity threshold for semantic splits
```

### Tuning CHUNK_SIZE

- **Small chunks (256-512 tokens)**:
  - More precise retrieval
  - Better for specific queries
  - More chunks = more storage/processing

- **Medium chunks (512-1024 tokens)**:
  - Balanced approach (recommended)
  - Good context without overwhelming

- **Large chunks (1024-2048 tokens)**:
  - More context per chunk
  - Risk of mixing multiple topics
  - Fewer chunks to process

### Tuning CHUNK_OVERLAP

- **No overlap (0 tokens)**:
  - Simplest approach
  - Risk of splitting important context

- **Small overlap (25-50 tokens)**:
  - Preserves sentence boundaries
  - Minimal redundancy

- **Large overlap (100-200 tokens)**:
  - Better context preservation
  - More redundancy and storage

### Tuning SEMANTIC_THRESHOLD

- **Low threshold (0.3-0.5)**:
  - More splits (smaller, focused chunks)
  - Sensitive to topic changes

- **High threshold (0.6-0.8)**:
  - Fewer splits (larger, broader chunks)
  - Only splits on major topic shifts

## Cost Breakdown

### Embedding Costs

**Fixed Chunking**:
- Chunks only (no sentence embeddings needed)
- 1 long document → ~20 chunks
- Cost: $0.0001 for chunking + embedding

**Recursive Chunking**:
- Chunks only
- Similar cost to fixed chunking

**Semantic Chunking**:
- Requires embedding ALL sentences (expensive!)
- 1 long document → 100+ sentences → 100+ embeddings
- Cost: $0.001-0.005 per document (10-50x more than fixed)

### Total Estimated Cost

- **Fixed/Recursive**: ~$0.0005 for 10 documents
- **Semantic**: ~$0.01 for 10 documents
- **Query evaluation**: ~$0.0001 per query

### Optimization Tips

- Use semantic chunking selectively for high-value documents
- Cache sentence embeddings if processing multiple times
- Consider fixed/recursive for cost-sensitive applications
- For production, pre-compute and store chunks offline

## Common Issues & Solutions

### Issue 1: "punkt not found" error from NLTK
**Solution**: The code automatically downloads it, but if it fails:
```python
import nltk
nltk.download('punkt')
```

### Issue 2: Semantic chunking is very slow
**Solution**: This is expected – it embeds every sentence. For large documents:
- Use fixed or recursive chunking instead
- Pre-compute and cache semantic chunks
- Limit semantic chunking to critical documents only

### Issue 3: Chunks are too small or too large
**Solution**: Adjust `CHUNK_SIZE` in config:
```python
# For smaller chunks
CHUNK_SIZE = 256

# For larger chunks
CHUNK_SIZE = 1024
```

### Issue 4: Semantic chunker creates too many/too few chunks
**Solution**: Adjust `SEMANTIC_THRESHOLD`:
```python
# More sensitive (more splits)
SEMANTIC_THRESHOLD = 0.4

# Less sensitive (fewer splits)
SEMANTIC_THRESHOLD = 0.7
```

### Issue 5: Fixed chunks split sentences awkwardly
**Solution**: Use recursive chunking instead, or increase chunk size.

## Key Takeaways

- ✅ Chunking is essential for long documents to maintain retrieval precision
- ✅ Fixed chunking is fast and predictable but ignores document structure
- ✅ Recursive chunking respects paragraphs and sentences for more readable chunks
- ✅ Semantic chunking groups related content but is computationally expensive
- ✅ Overlap preserves context across chunk boundaries
- ✅ Optimal chunk size depends on your use case (typically 512-1024 tokens)

## Real-World Applications

- **Technical Documentation**: Recursive chunking respects section boundaries
- **Research Papers**: Semantic chunking keeps related findings together
- **Legal Documents**: Fixed chunking ensures consistent processing
- **Customer Support**: Small chunks (256-512) for precise answers
- **Knowledge Base**: Medium chunks (512-1024) balance context and precision

## Going Further

Ideas for extending this level:

1. **Hierarchical Chunking**: Create parent-child relationships between chunks
2. **Context Windows**: Include prev/next chunk context in retrieval
3. **Metadata Enrichment**: Add section headers, page numbers to chunks
4. **Hybrid Chunking**: Combine strategies (semantic within fixed-size limit)
5. **Query-Aware Chunking**: Adjust chunk size based on expected query types

Example: Hierarchical chunking with context
```python
chunk = {
    "text": "...",
    "parent_section": "System Components",
    "previous_chunk": chunk_id_minus_1,
    "next_chunk": chunk_id_plus_1
}
```

## What's Next?

You now have a complete RAG pipeline: embedding, retrieval (semantic vs exact matching), and chunking. These are the core building blocks of production RAG systems.

**Congratulations!** You've completed the core RAG training levels. You now understand:
- ✅ Vector search and semantic similarity (Level 01)
- ✅ When to use semantic vs exact matching (Level 02)
- ✅ How to chunk documents effectively (Level 03)

**Continue Your Learning:**

➡️ [Level 04: Agentic RAG](../04-agentic-rag/README.md) - Explore advanced patterns with self-correction, tool use, and multi-step reasoning.

➡️ For now, experiment with the code:
- Try all three chunking strategies on your own documents
- Test different chunk sizes and overlap settings
- Combine what you learned in Levels 01-03 for your use case

---

← [Back to Level 02: Semantic vs Exact Match](../02-semantic-vs-exact/README.md) | [Back to Main README](../../README.md)
