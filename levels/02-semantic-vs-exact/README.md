# Level 02: Semantic vs Exact Match

## What You'll Learn

- Understanding semantic search (embeddings) vs keyword search (BM25)
- When vector search excels (semantic understanding)
- When BM25 excels (exact matching, IDs, codes)
- Comparing retrieval approaches with real-world examples

## Concept Explanation

Not all queries are created equal. Understanding when to use semantic search vs exact matching is crucial for building effective RAG systems.

**Vector Search (Semantic)** excels at:
- Understanding meaning and context
- Finding semantically similar content
- Handling paraphrasing and synonyms
- Example: "How do I get reimbursed for travel?" → finds "expense reimbursement policy"

**BM25 Search (Keyword/Exact Match)** excels at:
- Finding exact matches for IDs, codes, serial numbers
- Technical jargon and specific terminology
- Unique identifiers (Emirates IDs, product codes, etc.)
- Example: "784-1992-7856432-1" → finds exact citizen record

This level demonstrates both approaches side-by-side using two types of queries:
1. **Semantic query**: "How do I submit travel expenses?" (vector search wins)
2. **Exact ID query**: "784-1992-7856432-1" (BM25 wins)

## How It Works

### Comparison Architecture

```
                    ┌─────────────┐
                    │   Query     │
                    └──────┬──────┘
                           │
                  ┌────────┴────────┐
                  │                 │
         ┌────────▼────────┐  ┌────▼──────────┐
         │ Vector Search   │  │  BM25 Search  │
         │  (Semantic)     │  │  (Exact)      │
         │                 │  │               │
         │ • Embeddings    │  │ • Tokenization│
         │ • Cosine Sim    │  │ • Term Freq   │
         └────────┬────────┘  └────┬──────────┘
                  │                 │
                  │                 │
           ┌──────▼──────┐   ┌──────▼──────┐
           │  Results 1  │   │  Results 2  │
           └─────────────┘   └─────────────┘
                  │                 │
                  └────────┬────────┘
                           │
                    ┌──────▼──────┐
                    │  Comparison │
                    └─────────────┘
```

### Sample Documents

This level includes two types of documents:

1. **Generic Documents** (company policies, technical docs):
   - Best for semantic queries
   - Natural language content
   - Contextual information

2. **Citizen Records** (UAE Emirates IDs):
   - Best for exact ID matching
   - Structured data
   - Unique identifiers

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

**Dual Query Comparison** (`main.py`):
```python
queries = [
    {
        "query": "How do I submit travel expenses?",
        "description": "Semantic query - Vector search excels here",
        "k": TOP_K
    },
    {
        "query": "784-1992-7856432-1",
        "description": "Exact ID match - BM25 excels here",
        "k": 4
    }
]

for query_info in queries:
    # Run both searches
    vector_results = vector_retriever.search(query, k=k)
    bm25_results = bm25_retriever.search(query, k=k)

    # Compare results side-by-side
    print_comparison(query, vector_results, bm25_results)
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
cd levels/02-semantic-vs-exact

# Step 3: Run the example
python main.py

# Step 4: View the comparison
cat output/comparison.txt
```

## Configuration & Tuning

In `utils/config.py`, you can adjust:

```python
TOP_K = 5              # Number of results to retrieve per query
```

### Adding More Queries

You can easily add more comparison queries in `main.py`:

```python
queries = [
    {
        "query": "How do I submit travel expenses?",
        "description": "Semantic query - Vector search excels here",
        "k": TOP_K
    },
    {
        "query": "784-1992-7856432-1",
        "description": "Exact ID match - BM25 excels here",
        "k": 4
    },
    # Add your own queries here
    {
        "query": "API authentication",
        "description": "Technical term - compare both approaches",
        "k": TOP_K
    }
]
```

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

### Issue 2: Vector search doesn't find exact matches
**Solution**: This is expected! Vector search prioritizes semantic similarity over exact matching:
- For IDs, codes, or serial numbers, use BM25
- For semantic queries, use vector search
- Understanding this trade-off is the key learning of this level

### Issue 3: Different methods return completely different results
**Solution**: This is intentional and demonstrates each method's strengths:
- Vector search finds semantically similar documents
- BM25 finds exact keyword matches
- Compare the outputs in `comparison.txt` to understand when to use each

### Issue 4: Scores are not comparable between methods
**Solution**: Don't compare raw scores across methods:
- Vector search uses cosine similarity (0-1 range)
- BM25 uses term frequency-based scoring (unbounded)
- Focus on rank order, not absolute scores

## Key Takeaways

- ✅ Vector search excels at semantic understanding and contextual queries
- ✅ BM25 excels at exact matches (IDs, codes, specific terminology)
- ✅ Neither approach is universally better - they have different strengths
- ✅ Understanding query type helps you choose the right approach
- ✅ Real production systems often combine both (hybrid search - covered in advanced topics)

## Real-World Applications

### When to Use Vector Search
- **Customer Support**: "How do I reset my password?" (semantic understanding)
- **Knowledge Base**: "What's the vacation policy?" (conceptual queries)
- **Content Discovery**: "Articles about machine learning" (broad topics)

### When to Use BM25
- **ID Lookup**: "Order #12345" or "Invoice INV-2024-001" (exact matches)
- **Technical Search**: "ERROR_CODE_404" or "API_KEY_INVALID" (specific terms)
- **Citizen Records**: "784-1992-7856432-1" (unique identifiers)

### When to Use Hybrid (Advanced)
- **E-commerce**: Combine product names with semantic understanding
- **Legal Search**: Exact statutes + similar case precedents
- **Medical Records**: Patient IDs + symptom descriptions

## Going Further

Ideas for extending this level:

1. **Add More Query Types**: Test with technical queries, natural language, IDs
2. **Create Custom Documents**: Add your own document types and test both approaches
3. **Implement Query Classifier**: Auto-detect query type and choose the right method
4. **Custom Tokenization**: Implement better tokenization using nltk or spaCy for BM25
5. **Score Visualization**: Plot score distributions from each method

Example: Auto-select search method based on query:
```python
def select_search_method(query: str) -> str:
    """Choose search method based on query pattern."""
    # Check if query looks like an ID (numbers, dashes, specific format)
    if re.match(r'^\d{3}-\d{4}-\d{7}-\d$', query):
        return 'bm25'  # Exact match for IDs
    # Check for question words (semantic queries)
    elif any(word in query.lower() for word in ['how', 'why', 'what', 'when']):
        return 'vector'  # Semantic understanding
    # Default: try both and compare
    return 'both'
```

## What's Next?

Now you understand when to use semantic vs exact matching. But we've been treating documents as monolithic chunks. What if your documents are thousands of words long? You'll need an effective chunking strategy.

---

👈 Back to [Level 01: Basic Vector Search](../01-basic-vector-search/README.md) | 👉 Continue to [Level 03: Chunking Strategies](../03-chunking-strategies/README.md)
