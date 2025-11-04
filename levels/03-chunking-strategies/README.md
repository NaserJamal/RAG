# Level 03: Chunking Strategies

## What You'll Learn

- Different text chunking approaches (fixed, semantic, contextual)
- **Contextual Retrieval**: Using LLMs to enrich chunks with explanatory context
- Impact of chunk size and enrichment strategies on retrieval quality
- Overlap strategies for context preservation
- Cost-quality tradeoffs between chunking strategies
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

#### 2. Semantic Chunking
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

#### 3. Contextual Retrieval (LLM-Based Enrichment)
**Not a chunking method itself** - rather an enrichment technique that enhances any base chunking strategy. Uses an LLM to generate chunk-specific explanatory context, then prepends it to each chunk.

In this implementation: Fixed-size chunks → LLM enrichment → Enhanced chunks

**Pros**:
- Best retrieval accuracy (49-67% reduction in failed retrievals)
- Adds missing document context to each chunk
- Works with any base chunking strategy (fixed, semantic, etc.)
- Makes chunks self-contained and more meaningful

**Cons**:
- Most expensive (requires LLM calls for each chunk)
- Slower processing time
- Adds additional tokens to each chunk (increases storage and embedding costs)

**Best for**: High-value documents where retrieval accuracy is critical (e.g., legal, medical, research)

**How it works**:
```
Original chunk: "Deploy new model in shadow mode..."
Generated context: "This chunk covers deployment strategies including shadow deployment, canary deployment..."
Final enriched chunk: "[Context]\n\n[Original chunk]"
```

### Architecture Overview

**Standard Chunking Flow**:
```
Document → Chunker → Chunks → Embeddings → Search
```

**Contextual Retrieval Flow**:
```
Document → Base Chunker → Chunks → LLM Enrichment → Enhanced Chunks → Embeddings → Search
                                          ↓
                            (Adds explanatory context to each chunk)
```

Both flows end with evaluation metrics to compare retrieval accuracy.

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

**Contextual Chunker** (`utils/contextual_chunker.py`):
```python
def chunk(self, text: str, doc_id: str) -> List[Dict]:
    """Split text and enrich each chunk with LLM-generated context."""
    # Step 1: Create base chunks (fixed-size)
    base_chunks = self._create_base_chunks(text, doc_id)

    # Step 2: Enrich each chunk with context
    enriched_chunks = []
    for chunk in base_chunks:
        # Generate context using LLM
        context = self._generate_context(text, chunk["text"])

        # Prepend context to chunk
        enriched_text = f"{context}\n\n{chunk['text']}"
        enriched_chunks.append(enriched_text)

    return enriched_chunks
```

## Prerequisites

- Completed: Levels 01-02 (for understanding retrieval concepts)
- API Keys:
  - OpenAI API key for embeddings
  - LLM API key for contextual enrichment (OpenAI, Anthropic, or compatible)
- Python packages: Installed via `requirements.txt` (includes `tiktoken` and `nltk`)
- Note: Contextual retrieval requires LLM access and incurs additional costs

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
3. For contextual retrieval: Generate LLM-based context for each chunk
4. Evaluate each strategy with a test query
5. Generate comparison metrics and visualizations showing retrieval accuracy

## Configuration & Tuning

In `utils/config.py`:

```python
CHUNK_SIZE = 512            # Target chunk size in tokens
CHUNK_OVERLAP = 50          # Overlap between chunks
SEMANTIC_THRESHOLD = 0.5    # Similarity threshold for semantic splits
CONTEXT_INSTRUCTIONS_TEMPLATE = "..."  # Template for LLM context generation
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

### Tuning CONTEXT_INSTRUCTIONS_TEMPLATE

The LLM prompt controls how contextual enrichment works:

- **Short context**: 1-2 sentences summarizing the chunk's role
- **Detailed context**: Include document title, section, and topic
- **Query-focused**: Emphasize information retrieval aspects

Example template:
```python
"""Given the entire document:
<document>
{document}
</document>

Generate a concise context (50-100 words) for this chunk:
<chunk>
{chunk}
</chunk>

Context should explain what this chunk discusses and where it fits in the document."""
```

## Cost Breakdown

### Embedding Costs

**Fixed Chunking**:
- Chunks only (no sentence embeddings needed)
- 1 long document → ~20 chunks
- Cost: $0.0001 for chunking + embedding

**Semantic Chunking**:
- Requires embedding ALL sentences (expensive!)
- 1 long document → 100+ sentences → 100+ embeddings
- Cost: $0.001-0.005 per document (10-50x more than fixed)

**Contextual Retrieval**:
- Most expensive: LLM call for EACH chunk
- 1 long document → ~20 chunks → 20 LLM calls
- Each call processes full document + generates context
- Cost: $0.01-0.05 per document (10-50x more than semantic)
- Example: 20 chunks × (10k tokens input + 100 tokens output) ≈ $0.02-0.04

### Total Estimated Cost

- **Fixed**: ~$0.0005 for 10 documents
- **Semantic**: ~$0.01 for 10 documents
- **Contextual**: ~$0.30 for 10 documents (most expensive but best accuracy)
- **Query evaluation**: ~$0.0001 per query

### Optimization Tips

- **Use contextual retrieval selectively** for high-value, mission-critical documents
- **Implement prompt caching** to reduce repeated document processing (can reduce costs by 50-90%)
- Use semantic chunking for moderate quality needs
- Consider fixed chunking for cost-sensitive applications
- Pre-compute and cache all chunks offline in production
- Batch process documents to amortize API overhead

## Common Issues & Solutions

### Issue 1: "punkt not found" error from NLTK
**Solution**: The code automatically downloads it, but if it fails:
```python
import nltk
nltk.download('punkt')
```

### Issue 2: Semantic chunking is very slow
**Solution**: This is expected – it embeds every sentence. For large documents:
- Use fixed chunking instead
- Pre-compute and cache semantic chunks
- Limit semantic chunking to critical documents only

### Issue 3: Contextual retrieval is expensive and slow
**Solution**: This is expected – it makes LLM calls for each chunk. To optimize:
- Implement prompt caching (reduces cost by 50-90%)
- Process documents in batch offline
- Use only for high-value documents
- Consider fixed or semantic chunking for cost-sensitive use cases
- Cache generated contexts for reuse

### Issue 4: Chunks are too small or too large
**Solution**: Adjust `CHUNK_SIZE` in config:
```python
# For smaller chunks
CHUNK_SIZE = 256

# For larger chunks
CHUNK_SIZE = 1024
```

### Issue 5: Semantic chunker creates too many/too few chunks
**Solution**: Adjust `SEMANTIC_THRESHOLD`:
```python
# More sensitive (more splits)
SEMANTIC_THRESHOLD = 0.4

# Less sensitive (fewer splits)
SEMANTIC_THRESHOLD = 0.7
```

### Issue 6: Context generation is not helpful
**Solution**: Adjust the `CONTEXT_INSTRUCTIONS_TEMPLATE` to:
- Be more specific about what context to include
- Request shorter/longer contexts
- Focus on particular aspects (topic, purpose, relationships)
- Include examples of good contexts

### Issue 7: LLM API errors during context generation
**Solution**: Check your LLM configuration:
- Verify API key is set correctly
- Ensure base URL is correct for your provider
- Check model name is valid
- Monitor rate limits and implement backoff/retry logic

## Key Takeaways

- ✅ Chunking is essential for long documents to maintain retrieval precision
- ✅ Fixed chunking is fast and predictable but ignores document structure
- ✅ Semantic chunking groups related content but requires embedding all sentences
- ✅ **Contextual retrieval provides best accuracy** by enriching chunks with LLM-generated context
- ✅ Contextual retrieval reduces failed retrievals by 49-67% but is most expensive
- ✅ Overlap preserves context across chunk boundaries
- ✅ Optimal chunk size depends on your use case (typically 512-1024 tokens)
- ✅ Cost-quality tradeoff: Fixed < Semantic < Contextual (by both cost and quality)

## Real-World Applications

- **Legal Documents**: Contextual retrieval for critical contracts and regulatory docs
- **Medical Records**: Contextual enrichment ensures accurate clinical information retrieval
- **Research Papers**: Semantic chunking keeps related findings together
- **Technical Documentation**: Fixed chunking for fast, cost-effective processing
- **Customer Support**: Fixed chunking with small chunks (256-512) for precise answers
- **Enterprise Knowledge Base**: Contextual retrieval for mission-critical information
- **Financial Reports**: Contextual enrichment for accurate regulatory compliance

### Which Strategy to Choose?

| Use Case | Recommended Strategy | Reasoning |
|----------|---------------------|-----------|
| High-stakes retrieval (legal, medical) | **Contextual** | Best accuracy, worth the cost |
| Large-scale knowledge base | **Semantic** | Good balance of quality and cost |
| Cost-sensitive applications | **Fixed** | Fastest and cheapest |
| Real-time processing | **Fixed** | No LLM/embedding overhead |
| Mixed document types | **Contextual** | Handles variety best |

## Going Further

Ideas for extending this level:

1. **Hierarchical Chunking**: Create parent-child relationships between chunks
2. **Context Windows**: Include prev/next chunk context in retrieval
3. **Metadata Enrichment**: Add section headers, page numbers to chunks
4. **Hybrid Chunking**: Combine strategies (e.g., contextual enrichment on semantic chunks)
5. **Query-Aware Chunking**: Adjust chunk size based on expected query types
6. **Prompt Caching for Contextual**: Implement caching to reduce LLM costs by 50-90%
7. **Batch Processing**: Process multiple documents in parallel for faster contextual enrichment
8. **Context Quality Metrics**: Measure how helpful generated contexts are

Example: Hierarchical chunking with contextual enrichment
```python
chunk = {
    "text": "...",
    "context": "This section discusses...",  # LLM-generated
    "parent_section": "System Components",
    "previous_chunk": chunk_id_minus_1,
    "next_chunk": chunk_id_plus_1
}
```

Example: Hybrid approach (best of both worlds)
```python
# Step 1: Use semantic chunking to find natural boundaries
semantic_chunks = semantic_chunker.chunk(document)

# Step 2: Enrich each semantic chunk with contextual information
contextual_chunks = contextual_chunker.enrich(semantic_chunks, document)
```

## What's Next?

You now have a complete RAG pipeline with three different chunking strategies: fixed (fast), semantic (balanced), and contextual (best accuracy). These are the core building blocks of production RAG systems.

**Congratulations!** You've completed the core RAG training levels. You now understand:
- ✅ Vector search and semantic similarity (Level 01)
- ✅ When to use semantic vs exact matching (Level 02)
- ✅ How to chunk documents effectively with three different strategies (Level 03)
- ✅ How contextual retrieval dramatically improves accuracy using LLM-generated context

**Continue Your Learning:**

➡️ [Level 04: Agentic RAG](../04-agentic-rag/README.md) - Explore advanced patterns with self-correction, tool use, and multi-step reasoning.

➡️ For now, experiment with the code:
- Compare all three chunking strategies on your own documents
- Test contextual retrieval on high-value documents
- Experiment with different context generation prompts
- Measure the accuracy improvement from contextual enrichment
- Optimize costs using prompt caching and selective application

---

👈 Back to [Level 02: Semantic vs Exact Match](../02-semantic-vs-exact/README.md) | 👉 Continue to [Level 04: Full Pipeline](../04-full-pipeline/README.md)
