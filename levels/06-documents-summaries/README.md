# Level 06: Multi-Document Executive Summaries

Learn how to create executive summaries from hundreds of documents using hierarchical map-reduce summarization.

## What You'll Learn

- **Map-Reduce pattern** for parallel document processing
- **Hierarchical summarization** that scales to hundreds of documents
- **Tree-based batching** for efficient LLM usage
- **Async/parallel processing** to maximize speed
- **Cost-effective summarization** with O(log n) complexity

## The Problem

When you have 10, 50, or 100+ documents to summarize:

- ❌ Sending all documents to an LLM at once hits context limits
- ❌ Processing sequentially is too slow
- ❌ Simple concatenation loses important details
- ❌ Costs explode with large document counts

**Solution**: Hierarchical Map-Reduce with batched reduction.

## How It Works

### Architecture: Tree-Based Map-Reduce

```
Level 0 (Map):     [Doc1] [Doc2] [Doc3] ... [Doc30]  ← 30 docs
                      ↓      ↓      ↓           ↓
                   [Sum1] [Sum2] [Sum3] ... [Sum30]  ← Parallel summarization

Level 1 (Reduce):  [Sum1-10] [Sum11-20] [Sum21-30]  ← Batch of 10
                       ↓          ↓          ↓
                     [Comb1]   [Comb2]    [Comb3]   ← Parallel combination

Level 2 (Reduce):        [Comb1-3]                  ← Final batch
                             ↓
                    [Executive Summary]             ← Polished output
```

### Key Features

1. **Map Phase**: Each document summarized independently (fully parallel)
2. **Reduce Phase**: Summaries combined in batches of 10 (parallel within each level)
3. **Hierarchical**: Continues until single summary remains
4. **Scalable**: O(log₁₀ n) levels for n documents

### Why Batch Size of 10?

- Keeps context manageable for LLM
- Balances parallelism vs depth
- Optimal for most LLM context windows
- Easy to adjust based on your needs

## Quick Start

```bash
# 1. Navigate to this level
cd levels/06-documents-summaries

# 2. Add your PDF documents
cp /path/to/your/*.pdf documents/

# 3. Ensure .env is configured with LLM credentials
# (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)

# 4. Run summarization
python main.py

# 5. (Optional) Provide custom instructions when prompted:
#    "Focus on financial metrics and ROI"
#    "Highlight technical challenges and solutions"
#    "Summarize in bullet points only"
#    Or press Enter to use default format
```

## Example Output

After processing 30 documents:

```
============================================================
Starting hierarchical summarization of 30 documents
============================================================

[Level 0: Map Phase] Summarizing 30 documents in parallel...
✓ Completed 30 summaries | Tokens: 24,500

[Level 1: Reduce Phase] Combining 30 summaries...
  → Processing 3 batches of up to 10 summaries each
✓ Generated 3 combined summaries | Tokens: 8,200

[Level 2: Reduce Phase] Combining 3 summaries...
  → Processing 1 batch of up to 10 summaries each
✓ Generated 1 combined summary | Tokens: 3,100

[Final Phase] Creating executive summary...

============================================================
✓ Summarization complete!
  Documents processed: 30
  Total tokens used: 35,800
  Processing time: 45.3s
  Levels in hierarchy: 2
============================================================
```

## Output Files

The system creates organized outputs in `output/`:

```
output/
├── 01_individual_summaries.json     # Each document's summary
├── 02_level_1_summaries.json        # First reduce level
├── 02_level_2_summaries.json        # Second reduce level (if needed)
├── 03_executive_summary.txt         # Final polished summary
└── 04_metadata.json                 # Processing statistics
```

### Example: Individual Summaries

```json
[
  {
    "filename": "quarterly_report_q1.pdf",
    "summary": "Q1 revenue increased 15% YoY to $2.3M. Key drivers were enterprise sales and product expansion. Operating expenses remained flat at $1.1M. Net profit margin improved from 12% to 18%.",
    "tokens_used": 823
  },
  {
    "filename": "quarterly_report_q2.pdf",
    "summary": "Q2 showed continued growth with revenue reaching $2.7M. Customer acquisition costs decreased by 20%. New product line contributed $400K in revenue...",
    "tokens_used": 891
  }
]
```

### Example: Executive Summary

```
EXECUTIVE SUMMARY
=================================================================

Overview:
The organization demonstrated strong performance across all quarters,
with consistent revenue growth, improved operational efficiency, and
successful product launches. Key metrics show a 45% annual growth rate
with expanding profit margins.

Key Findings:
• Revenue Growth: Sustained 15-20% quarter-over-quarter growth
• Profitability: Net margins improved from 12% to 24%
• Product Expansion: New product lines contributed $1.2M annually
• Customer Acquisition: CAC reduced by 35% through optimization
• Market Position: Expanded to 3 new geographic markets

Critical Takeaways:
The company is well-positioned for continued growth with strong
fundamentals, efficient operations, and successful product
diversification. Focus areas for next period include scaling
international operations and maintaining current growth trajectory.
=================================================================
```

## Scalability Analysis

### Token Usage by Document Count

| Documents | Map Phase | Reduce Levels | Total Tokens | Est. Cost* |
|-----------|-----------|---------------|--------------|------------|
| 10        | ~8K       | 1             | ~12K         | $0.02      |
| 30        | ~24K      | 2             | ~36K         | $0.05      |
| 100       | ~80K      | 3             | ~120K        | $0.18      |
| 300       | ~240K     | 3             | ~350K        | $0.52      |
| 1000      | ~800K     | 4             | ~1.1M        | $1.65      |

*Based on GPT-4-mini pricing ($0.0015/1K tokens). Adjust for your model.

### Performance Characteristics

**Time Complexity**: O(log₁₀ n) levels
- 10 docs → 1 level
- 100 docs → 2 levels
- 1000 docs → 3 levels

**Parallelism**:
- Map phase: All docs in parallel (limited by `max_concurrent`)
- Reduce phase: All batches per level in parallel

**Typical Processing Times** (with max_concurrent=5):
- 10 docs: ~20-30 seconds
- 100 docs: ~2-3 minutes
- 1000 docs: ~15-20 minutes

## Custom Instructions

### Tailoring Your Executive Summary

When you run the script, you'll be prompted to provide optional custom instructions:

```
[Step 2/4] Custom instructions (optional)
──────────────────────────────────────────────────────────────────────
You can provide custom instructions for the executive summary.
Examples:
  • Focus on financial metrics and ROI
  • Highlight technical challenges and solutions
  • Summarize in bullet points only
  • Extract action items and deadlines

Press Enter to skip, or type your instructions:
──────────────────────────────────────────────────────────────────────
>
```

### Example Custom Instructions

**For Business Documents:**
```
Focus on financial metrics, revenue growth, and strategic initiatives.
Highlight risks and opportunities. Use a professional tone suitable for C-level executives.
```

**For Technical Documents:**
```
Emphasize technical architecture, implementation challenges, and performance metrics.
Include specific technologies mentioned and their trade-offs.
```

**For Research Papers:**
```
Summarize the research methodology, key findings, and conclusions.
Highlight novel contributions and future work suggestions.
```

**For Meeting Notes:**
```
Extract all action items, decisions made, and deadlines.
List who is responsible for each action item.
```

**For Legal/Compliance:**
```
Focus on regulatory requirements, compliance issues, and risk factors.
Highlight any deadlines or mandatory actions.
```

### How It Works

The custom instructions are passed to the LLM in the final executive summary generation:

```
## User Instructions:
{your custom instructions here}

Follow these user instructions carefully when creating the executive summary.
```

This allows you to:
- **Focus** the summary on specific aspects
- **Format** the output in your preferred style
- **Extract** particular types of information
- **Tailor** the tone and depth to your audience

The instructions are also saved in the metadata file for reference.

## Configuration

### Adjusting Parameters

In `main.py`, you can customize:

```python
summarizer = HierarchicalSummarizer(
    api_key=Config.LLM_API_KEY,
    base_url=Config.LLM_BASE_URL,
    model=Config.LLM_MODEL,
    batch_size=10,        # Summaries per reduce step
    max_concurrent=5      # Parallel LLM calls
)
```

**batch_size**:
- Smaller (5-7): More levels, less context per call
- Larger (15-20): Fewer levels, more context per call
- Default 10: Good balance for most use cases

**max_concurrent**:
- Lower (3-5): Gentler on API rate limits
- Higher (10-20): Faster but may hit rate limits
- Consider your API tier limits

### Environment Variables

Required in `.env`:

```bash
# LLM for summarization
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3-coder-30b-a3b-instruct

# Not required for this level (no embeddings used)
# EMBEDDING_API_KEY=...
# QDRANT_HOST=...
```

## How the Code Works

### 1. PDF Extraction (`utils/pdf_extractor.py`)

```python
extractor = PDFExtractor()
extracted_docs = extractor.extract_from_directory(documents_dir)
# Returns: [(filename, full_text), ...]
```

Simple extraction without chunking—we summarize entire documents.

### 2. Document Summarization - Map Phase

```python
async def summarize_document(document_text: str, filename: str):
    """Summarize single document (3-5 sentences)"""
    # Uses LLM to create concise summary
    # Runs in parallel with other documents
```

Each document gets its own focused summary preserving key information.

### 3. Summary Combination - Reduce Phase

```python
async def combine_summaries(summaries: List[str], level: int):
    """Combine up to 10 summaries into one"""
    # Identifies common themes
    # Removes redundancy
    # Maintains completeness
```

Batches of summaries are synthesized while preserving all critical info.

### 4. Hierarchical Processing

```python
current_summaries = [s['summary'] for s in doc_summaries]
while len(current_summaries) > 1:
    batches = split_into_batches(current_summaries, batch_size=10)
    current_summaries = await process_batches_parallel(batches)
```

Tree structure ensures logarithmic scaling.

### 5. Executive Summary Generation

```python
executive_summary = await create_executive_summary(
    final_summary,
    total_docs
)
```

Final polish with professional formatting and structure.

## Cost Optimization Strategies

### 1. **Use Cheaper Models for Map Phase**

```python
# Map phase: Use cheaper model for initial summaries
map_model = "gpt-4o-mini"  # $0.00015/1K tokens

# Reduce/Final: Use better model for synthesis
reduce_model = "gpt-4"     # $0.03/1K tokens
```

### 2. **Cache Individual Summaries**

```python
# Save individual summaries
# Reuse when re-generating executive summary with different prompts
summaries_cache = load_cache('01_individual_summaries.json')
```

### 3. **Adjust Batch Size**

```python
# Larger batches = fewer levels = fewer LLM calls
# But more tokens per call
batch_size = 15  # Instead of 10
```

### 4. **Selective Re-summarization**

```python
# Only re-summarize changed documents
# Reuse cached summaries for unchanged docs
```

## When to Use This Approach

### ✅ Best For:

- **Large document sets** (10-1000+ documents)
- **Executive summaries** across multiple sources
- **Quarterly/annual reports** from various departments
- **Research synthesis** from multiple papers
- **News aggregation** across many articles

### ❌ Not Ideal For:

- **Single document** summarization (just use one LLM call)
- **Real-time** requirements (latency of multiple levels)
- **Interactive** Q&A (use RAG instead - see Level 05)
- **Fine-grained** information retrieval (use vector search)

## Comparison with Other Approaches

### vs. Single LLM Call

**Single Call**: Send all docs to LLM at once
- ❌ Context limit issues (>100K tokens)
- ❌ Expensive (all tokens in one call)
- ✅ Simple implementation

**Map-Reduce**: Hierarchical processing
- ✅ No context limits
- ✅ Parallelizable
- ✅ Scalable to thousands of docs

### vs. RAG-Based Summarization

**RAG**: Retrieve relevant chunks, summarize those
- ✅ Very token-efficient
- ❌ May miss important content
- ✅ Good for targeted/themed summaries

**Map-Reduce**: Process all documents completely
- ✅ Comprehensive coverage
- ❌ Higher token usage
- ✅ Better for complete overviews

## Common Issues

### Issue: Rate Limits

```
Error: Rate limit exceeded
```

**Solution**: Reduce `max_concurrent`:

```python
summarizer = HierarchicalSummarizer(
    max_concurrent=3  # Instead of 5
)
```

### Issue: Timeout Errors

```
Error: Request timeout
```

**Solution**:
- Check your API endpoint is responsive
- Reduce document size or increase timeout
- Use more reliable model endpoint

### Issue: Context Length Errors

```
Error: Context length exceeded
```

**Solution**: Reduce `batch_size`:

```python
summarizer = HierarchicalSummarizer(
    batch_size=7  # Instead of 10
)
```

## Advanced Customization

### Custom Summary Prompts

Edit `utils/summarizer.py` to customize prompts:

```python
# For technical documents
prompt = """Summarize focusing on:
- Technical specifications
- Implementation details
- Performance metrics
..."""

# For business documents
prompt = """Summarize focusing on:
- Financial metrics
- Strategic initiatives
- Risk factors
..."""
```

### Domain-Specific Summaries

```python
async def summarize_technical_doc(document_text: str):
    """Custom summarizer for technical docs"""
    # Add domain-specific instructions

async def summarize_financial_doc(document_text: str):
    """Custom summarizer for financial docs"""
    # Add financial metrics extraction
```

### Multi-Stage Processing

```python
# Stage 1: Extract key metrics
# Stage 2: Summarize content
# Stage 3: Combine metrics + summaries
```

## Next Steps

After completing this level:

1. **Try different document types**: Technical papers, business reports, news articles
2. **Experiment with batch sizes**: Find optimal settings for your use case
3. **Customize prompts**: Tailor summaries to your domain
4. **Add filtering**: Preprocess documents to focus on relevant sections
5. **Integrate with Level 05**: Combine with agentic RAG for interactive exploration

### Potential Enhancements

- **Metadata extraction**: Pull out dates, authors, key figures
- **Theme clustering**: Group documents by topic before summarizing
- **Quality scoring**: Rank summary quality and iterate if needed
- **Multi-format output**: Generate summaries in different formats (bullets, prose, tables)

---

**Previous Level**: [Level 05: Agentic RAG](../05-agentic-rag/README.md)

**Next**: Build custom applications combining RAG + Summarization!
