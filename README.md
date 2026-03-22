# Gemini RAG Pipeline

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![AI-Powered](https://img.shields.io/badge/powered%20by-Google%20Gemini-orange)

A lightweight, production-ready Retrieval-Augmented Generation (RAG) pipeline built with the Google Gemini API, featuring smart chunking, robust retry logic, and a built-in in-memory vector store.

## 📖 Description

This project provides a complete, end-to-end RAG workflow that allows you to ingest documents and perform grounded Q&A. It uses `gemini-embedding-001` for high-dimensional vector representations and `gemini-3.1-flash-lite-preview` for fast, accurate response generation.

## ✨ Features

- 🧩 **Smart Chunking**: Automatically splits documents based on character limits while respecting sentence boundaries.
- 🧠 **In-Memory Vector Store**: A built-in, lightweight vector database using cosine similarity—no external database (like Pinecone or Chroma) required for local testing.
- 🔄 **Resilient API Handling**: Includes automatic retry logic with exponential backoff to handle rate limits and transient provider errors.
- 🛡️ **Grounded Generation**: System prompts are configured to ensure the model only answers based on the provided context, reducing hallucinations.
- 🚀 **Minimal Dependencies**: Built primarily on the official `google-genai` SDK.

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **LLM Provider:** [Google Gemini API](https://ai.google.dev/)
- **Models:**
  - Embedding: `gemini-embedding-001`
  - Generation: `gemini-3.1-flash-lite-preview`
- **Math/Logic:** Standard Python `math` library for vector similarity.

## 📋 Prerequisites

Before you begin, ensure you have:

1.  A Google AI Studio API Key. [Get one here](https://aistudio.google.com/).
2.  Python 3.9 or higher installed.

## 🚀 Installation & Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ashutoshswamy/rag-pipeline.git
   cd rag-pipeline
   ```

2. **Install dependencies:**

   ```bash
   pip install google-genai
   ```

3. **Set your API Key:**
   You can either edit the `GEMINI_API_KEY` variable in `rag_pipeline.py` or set it as an environment variable (recommended):

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

4. **Run the demo:**
   ```bash
   python rag_pipeline.py
   ```

## 💡 Usage Examples

### Basic Implementation

You can integrate the `RAGPipeline` class into your own applications:

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(chunk_size=500, overlap=100)

# Ingest a document
document_text = "The James Webb Space Telescope is a space telescope designed primarily to conduct infrared astronomy..."
pipeline.ingest(document_text, source="webb_telescope.txt")

# Query the document
answer = pipeline.query("What is the primary purpose of the Webb telescope?")
print(f"Response: {answer}")
```

### Advanced Configuration

Fine-tune the retrieval process by adjusting the `top_k` and `chunk_size` parameters:

```python
pipeline = RAGPipeline(
    chunk_size=400,
    overlap=50,
    top_k=5  # Retrieve more context chunks for complex queries
)
```

## 📂 Project Structure

```text
rag-pipeline/
├── rag_pipeline.py      # Main library logic and orchestrator
├── jwst_overview.txt    # Sample data for testing
├── README.md            # Project documentation
└── .gitignore           # Standard Python gitignore
```

## 🛠️ How it Works

1.  **Ingestion**: Text is broken into overlapping chunks. Every chunk is passed to the Gemini embedding model.
2.  **Storage**: Vectors and text chunks are stored in an internal `VectorStore` class.
3.  **Retrieval**: When a user asks a question, the query is embedded, and the most similar chunks are found using cosine similarity.
4.  **Generation**: The top-K chunks and the user query are wrapped in a strict "context-only" prompt and sent to Gemini Flash for the final answer.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

**Disclaimer**: _This project is for educational and development purposes. Ensure you monitor your Google GenAI usage to avoid unexpected costs._
