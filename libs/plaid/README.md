# 🦜️🔗 LangChain NextPlaid

This repository contains 1 package with [NextPlaid](https://github.com/meetdoshi90/next-plaid) integrations with LangChain:

- [langchain-plaid](https://pypi.org/project/langchain-plaid/) (WIP)

## Overview

NextPlaid is a high-performance ColBERT-style multi-vector search engine built in Rust.
This package provides a LangChain `VectorStore` integration, enabling late-interaction
retrieval with full metadata filtering support.

## Installation
```bash
pip install langchain-plaid
```

## Usage
```python
from langchain_plaid import NextPlaidVectorStore

vectorstore = NextPlaidVectorStore(
    url="http://localhost:8080",
    index_name="my_index",
    embedding=your_embedding_model,
)

# Add documents
vectorstore.add_documents(documents)

# Similarity search
results = vectorstore.similarity_search("query", k=4)

# Similarity search with metadata filter
results = vectorstore.similarity_search(
    "query",
    k=4,
    filter={"category": "science"},
)
```

## Running the NextPlaid Server
```bash
cargo run --release -p next-plaid-api -- --index-dir /tmp/indices --port 8080
```

See the [next-plaid repository](https://github.com/meetdoshi90/next-plaid) for full server documentation.

## Development
```bash
# Install dependencies
cd libs/plaid
pip install -e ".[dev]"

# Run integration tests (requires a running NextPlaid server)
python -m pytest tests/integration_tests/
```