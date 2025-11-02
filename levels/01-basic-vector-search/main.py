"""
Level 01: Basic Vector Search
A simple implementation of vector similarity search using OpenAI embeddings.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = 5
DOCUMENTS_PATH = Path(__file__).parent.parent.parent / "documents"
OUTPUT_PATH = Path(__file__).parent / "output"


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


def generate_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def cosine_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and document embeddings.

    Args:
        query_embedding: 1D array of query embedding
        doc_embeddings: 2D array of document embeddings (n_docs x embedding_dim)

    Returns:
        1D array of similarity scores
    """
    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.dot(doc_norms, query_norm)

    return similarities


def search(query: str, documents: List[Dict], embeddings: np.ndarray, client: OpenAI, k: int = TOP_K) -> List[Dict]:
    """
    Search for the most relevant documents for a given query.

    Args:
        query: Search query string
        documents: List of document dictionaries
        embeddings: Document embeddings matrix
        client: OpenAI client instance
        k: Number of top results to return

    Returns:
        List of top-k most relevant documents with scores
    """
    # Generate query embedding
    query_embedding = generate_embeddings([query], client)[0]

    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]

    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "score": float(similarities[idx]),
            "document": documents[idx]["id"],
            "path": documents[idx]["path"],
            "preview": documents[idx]["content"][:500] + "..." if len(documents[idx]["content"]) > 500 else documents[idx]["content"]
        })

    return results


def save_results(query: str, results: List[Dict], output_dir: Path):
    """Save search results to JSON and text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    output_data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "metadata": {
            "retrieval_method": "vector_search",
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K
        }
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    # Save as readable text
    with open(output_dir / "query_log.txt", 'w') as f:
        f.write(f"Query: {query}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"{'=' * 80}\n\n")

        for result in results:
            f.write(f"Rank {result['rank']}:\n")
            f.write(f"  Document: {result['document']}\n")
            f.write(f"  Similarity Score: {result['score']:.4f}\n")
            f.write(f"  Preview: {result['preview'][:200]}...\n")
            f.write(f"{'-' * 80}\n")


def main():
    """Main execution function."""
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} documents")

    print("\n🔢 Generating embeddings...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    doc_contents = [doc["content"] for doc in documents]
    embeddings = generate_embeddings(doc_contents, client)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Example query
    query = "What is the vacation policy?"
    print(f"\n🔍 Searching for: '{query}'")

    results = search(query, documents, embeddings, client)
    print(f"✅ Found {len(results)} results")

    # Display results
    print(f"\n{'=' * 80}")
    print("Top Results:")
    print(f"{'=' * 80}\n")

    for result in results:
        print(f"#{result['rank']} - {result['document']} (Score: {result['score']:.4f})")
        print(f"   {result['preview'][:150]}...")
        print()

    # Save results
    print("💾 Saving results...")
    save_results(query, results, OUTPUT_PATH)
    print(f"✅ Results saved to {OUTPUT_PATH}/")
    print(f"   - results.json")
    print(f"   - query_log.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
