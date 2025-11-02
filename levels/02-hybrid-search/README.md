# Level 02: Hybrid Search

## What You'll Learn

- Combining semantic search (embeddings) with keyword search (BM25)
- Reciprocal Rank Fusion (RRF) for merging results
- When to use hybrid vs pure vector search
- Trade-offs between semantic understanding and exact matching

## Concept Explanation

While vector search excels at understanding semantic meaning, it can miss exact matches for specific terms, product codes, or technical jargon. Conversely, keyword search (BM25) is great at exact matching but doesn't understand synonyms or context.

Hybrid search combines both approaches, giving you the best of both worlds: semantic understanding AND exact keyword matching.

For example:
- Query: "How do I submit travel expenses?"
- Vector search might find "expense reimbursement policy"
- BM25 might find documents with exact terms "submit" and "expenses"
- Hybrid search intelligently combines both to give you the most relevant results

This level implements Reciprocal Rank Fusion (RRF), a simple but effective method for combining ranked lists from different retrieval systems.

## How It Works

### Architecture Overview

```
                    ┌─────────────┐
                    │   Query     │
                    └──────┬──────┘
                           │
                  ┌────────┴────────┐
                  │                 │
         ┌────────▼────────┐  ┌────▼──────────┐
         │ Vector Search   │  │  BM25 Search  │
         │  (Semantic)     │  │  (Keyword)    │
         └────────┬────────┘  └────┬──────────┘
                  │                 │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ RRF Fusion      │
                  │ (Merge Results) │
                  └────────┬────────┘
                           │
                    ┌──────▼──────┐
                    │   Results   │
                    └─────────────┘
```

### Reciprocal Rank Fusion (RRF)

RRF merges ranked lists by assigning scores based on rank position:

```
Score(document) = Σ (1 / (k + rank))
```

Where:
- `k` is a constant (typically 60)
- `rank` is the position in each result list
- Higher-ranked documents get higher scores

The beauty of RRF is its simplicity and effectiveness without needing score normalization.

### Code Breakdown

**BM25 Retriever** (`utils/bm25_retriever.py`):
```python
class BM25Retriever:
    def index(self):
        """Build BM25 index from documents."""
        tokenized_docs = [doc["content"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search using BM25 keyword matching."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:k]
        # Return (doc_id, score) tuples
```

**Reciprocal Rank Fusion** (`utils/hybrid_retriever.py`):
```python
def _reciprocal_rank_fusion(self, vector_results, bm25_results, alpha=0.5):
    """Merge results using RRF."""
    fused_scores = defaultdict(float)

    # Vector search contribution (weighted by alpha)
    for rank, (doc_id, score) in enumerate(vector_results, start=1):
        rrf_score = alpha / (RRF_K + rank)
        fused_scores[doc_id] += rrf_score

    # BM25 contribution (weighted by 1-alpha)
    for rank, (doc_id, score) in enumerate(bm25_results, start=1):
        rrf_score = (1 - alpha) / (RRF_K + rank)
        fused_scores[doc_id] += rrf_score

    return dict(fused_scores)
```

## Prerequisites

- Completed: Level 01 (for understanding vector search concepts)
- API Key: OpenAI API key configured in `.env`
- Python packages: Installed via `requirements.txt`

## Running the Example

```bash
# Step 1: Ensure dependencies are installed
pip install -r ../../requirements.txt

# Step 2: Navigate to level directory
cd levels/02-hybrid-search

# Step 3: Run the example
python main.py

# Step 4: View the comparison
cat output/comparison.txt
```

## Configuration & Tuning

In `utils/config.py`, you can adjust:

```python
TOP_K = 5              # Number of results to retrieve
ALPHA = 0.5            # Weight: 0.0 = pure keyword, 1.0 = pure semantic
RRF_K = 60             # RRF constant (higher = more weight to top results)
```

### Adjusting ALPHA

The `alpha` parameter controls the balance between semantic and keyword search:

- **ALPHA = 0.0**: Pure keyword search (BM25 only)
  - Best for: Exact term matching, technical queries with specific jargon
  - Example: "CloudStore API key authentication"

- **ALPHA = 0.5**: Balanced hybrid (default)
  - Best for: General queries requiring both semantic and exact matching
  - Example: "How do I submit travel expenses?"

- **ALPHA = 1.0**: Pure semantic search (Vector only)
  - Best for: Conceptual queries, when synonyms and paraphrasing matter
  - Example: "What's the time-off policy?"

### Adjusting RRF_K

- **Lower K (30-40)**: More weight to top-ranked results
- **Higher K (80-100)**: More balanced weighting across ranks
- **Default K = 60**: Good general-purpose value

## Cost Breakdown

### Embedding Costs
- Model: text-embedding-3-small
- Rate: $0.00002 per 1K tokens
- 10 documents: ~$0.0001
- 10 queries: ~$0.0001

### BM25 Costs
- Free (runs locally, no API calls)

### Total Estimated Cost
- One-time setup: ~$0.0001
- Per query: ~$0.00001
- **100 queries: ~$0.001 (one-tenth of a cent!)**

### Optimization Tips
- BM25 adds no API cost, making hybrid search very cost-effective
- Cache embeddings to avoid regenerating on each run
- BM25 indexing is fast and runs locally

## Common Issues & Solutions

### Issue 1: BM25 returns unexpected results
**Solution**: BM25 uses simple tokenization (splitting on whitespace). For better results:
- Ensure documents are clean and well-formatted
- Consider lowercasing and removing punctuation
- For production, use more sophisticated tokenization (nltk, spaCy)

### Issue 2: Hybrid results favor one method too heavily
**Solution**: Adjust the `ALPHA` parameter:
```python
# In main.py, modify the search call:
hybrid_results = hybrid_retriever.search(query, k=TOP_K, alpha=0.7)  # More semantic
# or
hybrid_results = hybrid_retriever.search(query, k=TOP_K, alpha=0.3)  # More keyword
```

### Issue 3: Different methods return completely different results
**Solution**: This is normal! That's why hybrid search is powerful:
- Vector search finds semantically similar documents
- BM25 finds exact keyword matches
- RRF fusion combines insights from both
- Compare the outputs in `comparison.txt` to understand each method's strengths

### Issue 4: Scores are not normalized between methods
**Solution**: RRF doesn't require score normalization! It uses rank position, not raw scores. This is why RRF is robust and easy to implement.

## Key Takeaways

- ✅ Hybrid search combines semantic understanding with exact matching
- ✅ BM25 is a classical IR algorithm that works remarkably well for keyword search
- ✅ Reciprocal Rank Fusion is simple yet effective for merging results
- ✅ The alpha parameter lets you control semantic vs keyword weighting
- ✅ Different retrieval methods capture different aspects of relevance

## Real-World Applications

- **E-commerce Search**: Combine product name matching (keyword) with conceptual similarity (semantic)
- **Legal Document Search**: Find exact legal terms (keyword) and similar cases (semantic)
- **Code Search**: Match exact function names (keyword) and similar functionality (semantic)
- **Customer Support**: Find exact error codes (keyword) and similar issues (semantic)

## Going Further

Ideas for extending this level:

1. **Experiment with Alpha**: Try different alpha values for different query types
2. **Add Filtering**: Filter results by document type or metadata before fusion
3. **Custom Tokenization**: Implement better tokenization using nltk or spaCy
4. **Multi-Query Fusion**: Generate multiple query variations and fuse all results
5. **Score Visualization**: Plot score distributions from each method

Example: Auto-adjust alpha based on query characteristics:
```python
def smart_alpha(query: str) -> float:
    """Choose alpha based on query type."""
    # Queries with technical terms favor keyword search
    if any(term in query.lower() for term in ['api', 'error', 'code']):
        return 0.3  # More keyword
    # Conceptual queries favor semantic search
    elif any(word in query.lower() for word in ['how', 'why', 'what', 'explain']):
        return 0.7  # More semantic
    # Default balanced
    return 0.5
```

## What's Next?

Hybrid search significantly improves retrieval quality, but we've been treating documents as monolithic chunks. What if your documents are thousands of words long? You'll need an effective chunking strategy.

➡️ Continue to [Level 03: Chunking Strategies](../03-chunking-strategies/README.md) to learn how to split documents into optimal chunks for retrieval.

---

← [Back to Main README](../../README.md)
