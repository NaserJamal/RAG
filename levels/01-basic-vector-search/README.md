# Level 01: Basic Vector Search

## What You'll Learn

- Text embedding fundamentals with OpenAI embeddings
- Vector similarity search using cosine similarity
- Basic retrieval pipeline (embed → store → search → retrieve)
- When simple vector search works and when it fails

## Concept Explanation

Vector search is the foundation of modern Retrieval-Augmented Generation (RAG) systems. Instead of matching keywords, vector search understands the *meaning* of text by converting it into high-dimensional numerical vectors (embeddings). Similar concepts cluster together in this vector space, enabling semantic search.

For example, searching for "time off" would match documents about "vacation policy" even though they don't share the same words – because they're semantically similar.

This level implements the simplest possible vector search system: load documents, generate embeddings, compute similarity scores, and return top matches. While basic, this pattern forms the core of much more sophisticated RAG systems.

## How It Works

### Architecture Overview

```
Documents → OpenAI Embeddings → Vector Store (NumPy) → Cosine Similarity → Top-K Results
```

### Pipeline Steps

1. **Load Documents**: Read all text files from the `documents/` directory
2. **Generate Embeddings**: Use OpenAI's `text-embedding-3-small` model to convert text to 1536-dimensional vectors
3. **Store Vectors**: Keep embeddings in a NumPy array (in-memory)
4. **Process Query**: Convert user query to embedding using same model
5. **Compute Similarity**: Calculate cosine similarity between query and all document embeddings
6. **Return Top-K**: Sort by similarity and return the K most relevant documents

### Code Breakdown

**Loading Documents**:
```python
def load_documents() -> List[Dict[str, str]]:
    """Load all text documents from the documents directory."""
    documents = []
    for doc_dir in DOCUMENTS_PATH.iterdir():
        if not doc_dir.is_dir():
            continue
        for doc_file in doc_dir.glob("*.txt"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "id": f"{doc_dir.name}/{doc_file.name}",
                    "path": str(doc_file),
                    "content": content
                })
    return documents
```

**Generating Embeddings**:
```python
def generate_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Generate embeddings using OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)
```

**Cosine Similarity**:
```python
def cosine_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and document embeddings."""
    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.dot(doc_norms, query_norm)
    return similarities
```

## Prerequisites

- Python 3.8+
- OpenAI API key (or OpenRouter for cheaper alternatives)
- Packages installed via `requirements.txt` in the root directory

## Running the Example

```bash
# Step 1: Ensure you're in the root directory and have installed dependencies
pip install -r requirements.txt

# Step 2: Set up your API credentials
cp .env.example .env
# Edit .env and add your EMBEDDING_API_KEY and EMBEDDING_BASE_URL

# Step 3: Navigate to level directory
cd levels/01-basic-vector-search

# Step 4: Run the example
python main.py

# Step 5: View the output
cat output/results.json
cat output/query_log.txt
```

## Configuration & Tuning

The code uses sensible defaults, but you can adjust parameters at the top of `main.py`:

```python
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
TOP_K = 5                                    # Number of results to return
```

**Adjusting TOP_K**:
- Lower values (3-5): More precise, less context
- Higher values (10-20): More context, may include less relevant documents

**Embedding Model Options**:
- `text-embedding-3-small` (1536 dims): Faster, cheaper, good quality
- `text-embedding-3-large` (3072 dims): Better quality, more expensive
- `text-embedding-ada-002` (1536 dims): Legacy model, similar to 3-small

## Cost Breakdown

### Embedding Costs
- Model: text-embedding-3-small
- Rate: $0.00002 per 1K tokens
- Average document: ~500 tokens
- 10 documents: ~$0.0001
- 10 queries: ~$0.0001

### Total Estimated Cost
- One-time document embedding: ~$0.0001
- Per query: ~$0.00001
- **100 queries: ~$0.001 (one-tenth of a cent!)**

### Optimization Tips
- Cache document embeddings instead of regenerating each run
- Use smaller embedding models for simple use cases
- Batch multiple queries together to reduce API overhead
- Consider OpenRouter for even cheaper rates

## Common Issues & Solutions

### Issue 1: "Embedding API key not found"
**Solution**: Ensure your `.env` file exists in the root directory and contains:
```bash
EMBEDDING_API_KEY=your_actual_key_here
EMBEDDING_BASE_URL=your_actual_base_url_here
```

### Issue 2: Rate limit errors
**Solution**: OpenAI has rate limits on API calls. If you hit limits:
- Wait 60 seconds and retry
- Reduce the number of documents
- Upgrade your OpenAI API plan

### Issue 3: Results seem irrelevant
**Solution**: Vector search works best when:
- Documents are high quality and well-written
- Query is clear and specific
- Documents actually contain relevant information

Try different queries or examine the similarity scores. Scores below 0.5 often indicate weak relevance.

### Issue 4: Slow performance with many documents
**Solution**: This simple implementation loads everything into memory. For larger datasets:
- Proceed to Level 02 for more efficient implementations
- Consider using vector databases (Pinecone, Weaviate, Qdrant)
- Implement batch processing for embeddings

## Key Takeaways

- ✅ Vector embeddings capture semantic meaning beyond keyword matching
- ✅ Cosine similarity is the standard metric for comparing embeddings
- ✅ Simple in-memory vector stores work well for small to medium datasets (< 10K documents)
- ✅ Quality of embeddings directly impacts retrieval quality
- ✅ Vector search is fast: O(n) similarity computation, O(n log k) for top-K selection

## Real-World Applications

- **Customer Support**: Find relevant help articles based on user questions
- **Internal Knowledge Base**: Search company documentation semantically
- **Content Recommendation**: Find similar articles or products
- **Research**: Find related papers or documents in academic databases

## Going Further

Ideas for extending this level:
1. **Caching**: Save embeddings to disk to avoid regenerating on each run
2. **Interactive Mode**: Add a loop to accept multiple user queries
3. **Filtering**: Add metadata filtering (e.g., only search within specific categories)
4. **Explain Results**: Show which parts of documents matched the query
5. **Benchmark**: Compare different embedding models on your documents

Try modifying the code to implement one of these features!

## What's Next?

Vector search is powerful but has limitations. It struggles with:
- Exact keyword matching (e.g., product codes, IDs, serial numbers)
- Rare or specialized terminology
- Queries that need precise exact matching

➡️ Continue to [Level 02: Semantic vs Exact Match](../02-semantic-vs-exact/README.md) to learn when to use semantic search vs exact keyword matching.

---

← [Back to Main README](../../README.md)
