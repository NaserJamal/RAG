# Simple RAG - Progressive RAG Training

Learn Retrieval-Augmented Generation (RAG) through hands-on, production-ready examples.

## What You'll Build

Retrieval-Augmented Generation (RAG) is transforming how AI systems access and use information. Instead of relying solely on knowledge baked into model weights, RAG systems retrieve relevant information from external sources and use it to generate accurate, up-to-date responses.

This repository teaches RAG fundamentals through progressively complex levels. You'll start with basic vector search, progress through hybrid retrieval strategies, master chunking techniques, and understand when to apply each approach. Each level includes:

- **Working code** you can run immediately
- **Real-world sample documents** that demonstrate realistic use cases
- **Cost-conscious implementations** with optimization strategies
- **Detailed explanations** of why and when to use each technique

By the end, you'll understand the core building blocks of production RAG systems and have modular, adaptable code you can use in your own projects.

## Prerequisites

- **Python 3.8+**
- **OpenAI API key** (or OpenRouter for cheaper alternatives)
- **Basic understanding of Python** and API usage
- **Familiarity with command line** operations

No prior experience with RAG, embeddings, or vector search required – we'll teach you everything.

## Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd simple-rag

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run Level 01
cd levels/01-basic-vector-search
python main.py

# You should see results printed and saved to output/
```

That's it! You've just run your first vector search.

## Levels Overview

### 🎯 Level 01: Basic Vector Search
Learn the fundamentals of semantic search using embeddings and cosine similarity.

**What you'll learn**: Text embeddings, vector similarity, basic retrieval pipeline

**Time**: 15 minutes | **Cost**: ~$0.001

[Start Level 01 →](levels/01-basic-vector-search/README.md)

---

### 🔀 Level 02: Hybrid Search
Combine semantic search with keyword matching for more robust retrieval.

**What you'll learn**: BM25 keyword search, Reciprocal Rank Fusion, balancing semantic vs exact matching

**Time**: 25 minutes | **Cost**: ~$0.002

[Start Level 02 →](levels/02-hybrid-search/README.md)

---

### ✂️ Level 03: Chunking Strategies
Master the art of splitting documents for optimal retrieval quality.

**What you'll learn**: Fixed-size chunking, recursive splitting, semantic chunking, chunk overlap strategies

**Time**: 30 minutes | **Cost**: ~$0.01

[Start Level 03 →](levels/03-chunking-strategies/README.md)

---

### 🤖 Level 04: Agentic RAG
**Coming Soon** - Advanced patterns with self-correction, tool use, and multi-step reasoning.

[Learn more →](levels/04-agentic-rag/README.md)

---

## Installation

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for API calls

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd simple-rag
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (for Level 03):
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Configuration

### Setting Up Your API Key

1. **Copy the example environment file**:
```bash
cp .env.example .env
```

2. **Edit `.env` and add your API key**:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

### Using OpenRouter (Cheaper Alternative)

OpenRouter provides access to multiple LLM providers, often at lower cost:

```bash
# In .env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your-openrouter-key-here
```

Get a key at [openrouter.ai](https://openrouter.ai)

### Configuration Options

Each level has a `config.py` (or configuration at the top of `main.py`) where you can adjust:

- **Embedding models**: Switch between different OpenAI embedding models
- **Chunk sizes**: Adjust for your document types
- **Top-K results**: Number of retrieved documents/chunks
- **Search weights**: Balance semantic vs keyword search

## What Makes This Different?

- ✅ **Production-ready code**, not toy examples
  - Error handling, type hints, docstrings
  - Modular architecture you can adapt
  - Clean code following best practices

- ✅ **Cost-conscious** with optimization strategies
  - Detailed cost breakdowns for every level
  - Tips for reducing API expenses
  - Caching and efficiency recommendations

- ✅ **Progressive complexity** - one concept per level
  - Start simple, add sophistication gradually
  - Each level builds on previous knowledge
  - Clear explanations of when and why

- ✅ **Real-world sample documents**
  - Company policies, technical docs, news articles
  - Diverse content types to test different scenarios
  - Realistic document lengths and structures

- ✅ **Comprehensive documentation**
  - Concept explanations with diagrams
  - Code walkthroughs
  - Common issues and solutions
  - Ideas for extending each level

## Repository Structure

```
simple-rag/
├── README.md                        # You are here
├── requirements.txt                 # Python dependencies
├── .env.example                     # API configuration template
├── .gitignore                       # Git ignore rules
│
├── documents/                       # Shared sample documents
│   ├── company-kb/                  # Company knowledge base
│   │   ├── vacation-policy.txt
│   │   ├── expense-reimbursement.txt
│   │   ├── product-cloudstore-overview.txt
│   │   └── remote-work-policy.txt
│   ├── technical-docs/              # Technical documentation
│   │   ├── api-authentication-guide.txt
│   │   └── python-sdk-quickstart.txt
│   ├── news-articles/               # Current events articles
│   │   ├── ai-healthcare-breakthroughs-2025.txt
│   │   └── climate-tech-investments-surge.txt
│   └── mixed-content/               # Various document types
│       ├── short-note-meeting-summary.txt
│       └── comprehensive-ml-systems-design.txt
│
└── levels/                          # Progressive learning levels
    ├── 01-basic-vector-search/
    │   ├── README.md                # Level documentation
    │   ├── main.py                  # Single-file implementation
    │   └── output/                  # Generated results
    │
    ├── 02-hybrid-search/
    │   ├── README.md
    │   ├── main.py                  # Entry point
    │   ├── output/
    │   └── utils/                   # Modular package
    │       ├── config.py
    │       ├── embedder.py
    │       ├── vector_retriever.py
    │       ├── bm25_retriever.py
    │       └── hybrid_retriever.py
    │
    ├── 03-chunking-strategies/
    │   ├── README.md
    │   ├── main.py
    │   ├── output/
    │   └── utils/
    │       ├── config.py
    │       ├── embedder.py
    │       ├── fixed_chunker.py
    │       ├── recursive_chunker.py
    │       ├── semantic_chunker.py
    │       └── chunk_evaluator.py
    │
    └── 04-agentic-rag/
        └── README.md                # Coming soon placeholder
```

## Learning Path

### For Beginners
1. Start with **Level 01** to understand vector search fundamentals
2. Progress to **Level 02** to see why hybrid search matters
3. Complete **Level 03** to master chunking (essential for production)
4. Experiment with the code, try different documents and queries

### For Experienced Developers
- You can jump to any level, but we recommend at least skimming Level 01
- Level 02's RRF implementation is particularly elegant
- Level 03's semantic chunking showcases advanced techniques

### Time Commitment
- **Fast path**: 1-2 hours to run all levels and read READMEs
- **Deep learning**: 4-6 hours to understand code, experiment, extend
- **Production adaptation**: Days to weeks, depending on your use case

## Cost Transparency

All levels use OpenAI's `text-embedding-3-small` model for cost efficiency:

| Level | One-Time Setup | Per Query | 100 Queries Total |
|-------|---------------|-----------|-------------------|
| 01 - Basic Vector | $0.0001 | $0.00001 | $0.001 |
| 02 - Hybrid Search | $0.0001 | $0.00001 | $0.001 |
| 03 - Chunking (Fixed/Recursive) | $0.0005 | $0.00001 | $0.002 |
| 03 - Chunking (Semantic) | $0.01 | $0.0001 | $0.02 |

**Complete all levels: Less than $0.05 total**

### Cost Optimization Tips

1. **Cache embeddings** - Don't regenerate on each run
2. **Use batch operations** - Process multiple texts together
3. **Start small** - Test with a few documents first
4. **Monitor usage** - Check OpenAI dashboard regularly
5. **Consider alternatives** - OpenRouter, local embeddings (sentence-transformers)

## Common Issues & Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### "openai.AuthenticationError: Invalid API key"
**Solution**: Check your `.env` file has correct `OPENAI_API_KEY`

### "Rate limit exceeded"
**Solution**: Wait 60 seconds or reduce the number of documents you're processing

### "punkt not found" (Level 03)
**Solution**: Download NLTK data: `python -c "import nltk; nltk.download('punkt')"`

### Results seem irrelevant
**Solution**:
- Try different queries (be specific)
- Check if documents actually contain relevant information
- Adjust TOP_K to retrieve more results
- For Level 02+, tune the alpha/threshold parameters

### Slow performance
**Solution**:
- This is expected for semantic chunking (Level 03)
- Cache embeddings to avoid regenerating
- Use fewer documents for testing
- Consider vector databases (Pinecone, Weaviate) for production scale

## Going to Production

This repository teaches fundamentals. For production RAG systems, consider:

### Vector Databases
- **Pinecone**: Managed, easy to use, good DX
- **Weaviate**: Open source, feature-rich
- **Qdrant**: High performance, local or cloud
- **Chroma**: Simple, local-first

### Frameworks
- **LangChain**: Comprehensive, lots of integrations
- **LlamaIndex**: RAG-focused, excellent docs
- **Haystack**: Flexible, production-oriented

### Additional Considerations
- **Monitoring**: Track retrieval quality, latency, costs
- **Evaluation**: Implement metrics (MRR, NDCG, answer quality)
- **Scalability**: Handle millions of documents efficiently
- **Updates**: Strategy for adding/updating documents
- **Security**: Access control, data privacy, PII handling

## Contributing

Found a bug? Have a suggestion? Want to add a level?

- **Issues**: Open an issue on GitHub
- **Pull Requests**: Contributions welcome!
- **Ideas**: Share in discussions

Please ensure:
- Code follows existing style (type hints, docstrings)
- Documentation is comprehensive
- Costs are calculated and documented
- Examples are tested and working

## Resources

### Learning More About RAG
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [OpenAI: Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone: RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

### Vector Search & Embeddings
- [Jay Alammar: Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [Vicki Boykis: What are embeddings?](https://vickiboykis.com/what_are_embeddings/)

### Related Topics
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [BM25 Algorithm Explained](https://en.wikipedia.org/wiki/Okapi_BM25)

## License

MIT License - feel free to use this code in your projects, commercial or otherwise.

## Acknowledgments

This repository was inspired by patterns from production RAG systems and educational resources from the community. Special thanks to:

- OpenAI for accessible embedding models
- The open-source RAG community
- Everyone building and sharing RAG knowledge

---

**Ready to start?** Head to [Level 01: Basic Vector Search](levels/01-basic-vector-search/README.md)

**Questions?** Open an issue or start a discussion.

**Building something cool with this?** We'd love to hear about it!
