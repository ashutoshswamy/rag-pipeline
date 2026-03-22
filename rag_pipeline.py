"""
RAG Pipeline using Google Gemini API
=====================================
Models used:
    - Embeddings : gemini-embedding-001
    - Generation : gemini-3.1-flash-lite-preview

Install dependencies:
    pip install google-genai numpy

Usage:
  Set GEMINI_API_KEY env var, then run:
        python rag_pipeline.py
"""

import os
import math
import re
import time
import textwrap
from typing import Optional
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

GEMINI_API_KEY = "your api key"

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite-preview"

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks
TOP_K = 3  # number of chunks to retrieve per query


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)


# ──────────────────────────────────────────────
# Step 1: Chunking
# ──────────────────────────────────────────────


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:       The full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Ensure the cursor always advances even with short sentence-boundary chunks.
        start = max(end - overlap, start + 1)

    return chunks


# ──────────────────────────────────────────────
# Step 2: Embedding
# ──────────────────────────────────────────────


def _extract_retry_delay_seconds(error: Exception) -> Optional[float]:
    """Parse provider retry hints from error text like: 'Please retry in 30.06s'."""
    match = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(error))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _embed_one_text_with_retry(
    text: str,
    task_type: str,
    max_retries: int = 5,
) -> list[float]:
    """Embed a single text with retry/backoff for transient API failures."""
    for attempt in range(max_retries + 1):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return result.embeddings[0].values
        except Exception as exc:
            if attempt == max_retries:
                raise

            retry_after = _extract_retry_delay_seconds(exc)
            backoff = min(2**attempt, 30)
            wait_seconds = max(retry_after or 0.0, backoff)

            print(
                f"[embed] API error ({type(exc).__name__}). "
                f"Retrying in {wait_seconds:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_seconds)


def embed_texts(
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the configured embedding model.

    Args:
        texts:     List of strings to embed.
        task_type: One of RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    return [_embed_one_text_with_retry(text, task_type=task_type) for text in texts]


# ──────────────────────────────────────────────
# Step 3: Vector Store (in-memory)
# ──────────────────────────────────────────────


class VectorStore:
    """
    Simple in-memory vector store backed by cosine similarity.
    Drop-in replacement for Pinecone / Chroma / Qdrant for local use.
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ):
        """Store chunks with their embeddings."""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(chunks))

    def query(self, query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
        """
        Find top-k most similar chunks using cosine similarity.

        Returns:
            List of dicts: { chunk, score, metadata }
        """
        scores = [
            self._cosine_similarity(query_embedding, emb) for emb in self.embeddings
        ]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "chunk": self.chunks[i],
                "score": round(score, 4),
                "metadata": self.metadata[i],
            }
            for i, score in ranked
        ]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def __len__(self):
        return len(self.chunks)


# ──────────────────────────────────────────────
# Step 4: Retrieval
# ──────────────────────────────────────────────


def retrieve(query: str, store: VectorStore, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and retrieve the most relevant chunks.

    Args:
        query:  User question.
        store:  Populated VectorStore.
        top_k:  Number of chunks to return.

    Returns:
        List of retrieved results with chunk text and score.
    """
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    return store.query(query_embedding, top_k=top_k)


# ──────────────────────────────────────────────
# Step 5: Generation
# ──────────────────────────────────────────────


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate a grounded answer using Gemini with retrieved context.

    Args:
        query:            The user's question.
        retrieved_chunks: Top-k retrieved chunks from the vector store.

    Returns:
        The generated answer string.
    """
    if not retrieved_chunks:
        return "I don't have enough information to answer that."

    context = "\n\n---\n\n".join(
        f"[Source chunk, relevance score: {r['score']}]\n{r['chunk']}"
        for r in retrieved_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# ──────────────────────────────────────────────
# RAG Pipeline (orchestrator)
# ──────────────────────────────────────────────


class RAGPipeline:
    """
    End-to-end RAG pipeline backed by Google Gemini.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(document_text, source="my_doc.txt")
        answer = pipeline.query("What is this document about?")
        print(answer)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.store = VectorStore()

    def ingest(self, text: str, source: str = "document") -> int:
        """
        Chunk, embed, and store a document.

        Args:
            text:   Raw document text.
            source: Label for metadata (e.g. filename).

        Returns:
            Number of chunks stored.
        """
        print(f"[ingest] Chunking '{source}'...")
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        print(f"[ingest] {len(chunks)} chunks created. Embedding...")

        embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
        metadata = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        self.store.add(chunks, embeddings, metadata)
        print(f"[ingest] Done. Vector store now has {len(self.store)} chunks.\n")
        return len(chunks)

    def query(self, question: str, verbose: bool = False) -> str:
        """
        Run a full RAG query: retrieve relevant chunks, then generate an answer.

        Args:
            question: Natural language question.
            verbose:  If True, prints retrieved chunks before answering.

        Returns:
            Generated answer string.
        """
        if len(self.store) == 0:
            raise ValueError("No documents ingested. Call ingest(...) before query().")

        print(f"[query] '{question}'")
        results = retrieve(question, self.store, self.top_k)

        if verbose:
            print(f"\n[retrieve] Top {self.top_k} chunks:")
            for i, r in enumerate(results, 1):
                preview = r["chunk"][:120].replace("\n", " ")
                print(f"  {i}. score={r['score']} | {preview}...")

        answer = generate_answer(question, results)
        return answer


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────


def load_document(path: str) -> str:
    """Load a UTF-8 text document from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def main():
    print("=" * 60)
    print("RAG Pipeline — Google Gemini")
    print("=" * 60)

    pipeline = RAGPipeline(chunk_size=400, overlap=80, top_k=3)

    document_path = os.path.join(os.path.dirname(__file__), "jwst_overview.txt")
    try:
        document_text = load_document(document_path)
    except FileNotFoundError:
        print(f"Could not find document: {document_path}")
        return

    pipeline.ingest(document_text, source=os.path.basename(document_path))

    print("Ask any question about the ingested document.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Your question: ").strip()

        if not q:
            print("Please enter a question.\n")
            continue

        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("-" * 60)
        answer = pipeline.query(q, verbose=True)
        print(f"\nAnswer:\n{textwrap.fill(answer, width=70)}\n")


if __name__ == "__main__":
    main()
