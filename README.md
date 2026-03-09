# 🦜️🔗 LangChain NextPlaid

This repository contains the `langchain-plaid` package — a LangChain integration for [NextPlaid](https://github.com/meetdoshi90/next-plaid), a high-performance multi-vector (ColBERT-style) search server.

- [langchain-plaid on PyPI](https://pypi.org/project/langchain-plaid/) (WIP)

---

## What is NextPlaid?

NextPlaid is a Rust-based vector search server implementing ColBERT late-interaction retrieval (MaxSim scoring). Unlike single-vector stores, each document is stored as a **matrix of token embeddings** rather than a single vector, enabling significantly higher retrieval quality on tasks where semantic nuance matters.

Key properties:
- Multi-vector per document (ColBERT / ColPali style)
- MaxSim scoring at query time
- Quantized PLAID index (2-bit or 4-bit)
- Metadata filtering via SQL conditions
- Async writes, synchronous reads

---

## Installation

```bash
pip install langchain-plaid
```

Start the NextPlaid server (requires Rust / `cargo`):

```bash
cargo run --release -p next-plaid-api -- \
  --host 0.0.0.0 \
  --port 8080 \
  --index-dir /tmp/my-indices
```

Or with Docker:

```bash
docker run -p 8080:8080 -v /tmp/my-indices:/data \
  your-org/next-plaid-api --index-dir /data
```

---

## Quick Start

### Text retrieval with ColBERT

```python
from langchain_plaid import NextPlaidVectorStore
from pylate import models

class ColBERTEmbeddings:
    def __init__(self):
        self._model = models.ColBERT("lightonai/ColBERT-Zero")

    def embed_documents(self, texts):
        # Returns List[List[List[float]]] — one matrix per document
        return [e.tolist() for e in self._model.encode(texts, is_query=False)]

    def embed_query(self, text):
        # Returns List[List[float]] — one matrix for the query
        return self._model.encode([text], is_query=True)[0].tolist()

embeddings = ColBERTEmbeddings()

store = NextPlaidVectorStore(
    url="http://localhost:8080",
    index_name="my_docs",
    embedding=embeddings,
)

store.add_texts(
    ["ColBERT improves retrieval via late interaction.",
     "MaxSim scores each query token against all document tokens."],
    metadatas=[{"source": "paper"}, {"source": "paper"}],
)

docs = store.similarity_search("how does late interaction work?", k=5)
```

### Image retrieval with ColPali (ColModernVBert)

```python
from PIL import Image
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

class ColPaliEmbeddings:
    def __init__(self):
        self.processor = ColModernVBertProcessor.from_pretrained("ModernVBERT/colmodernvbert")
        self.model     = ColModernVBert.from_pretrained("ModernVBERT/colmodernvbert")

    def embed_images(self, images):
        # Returns List[List[List[float]]] — one (n_patches × dim) matrix per image
        inputs = self.processor.process_images(images)
        out    = self.model(**inputs)
        return [out[i].tolist() for i in range(len(images))]

    def embed_query(self, text):
        inputs = self.processor.process_texts([text])
        return self.model(**inputs)[0].tolist()

    def embed_documents(self, texts):
        inputs = self.processor.process_texts(texts)
        out    = self.model(**inputs)
        return [out[i].tolist() for i in range(len(texts))]

embeddings = ColPaliEmbeddings()
store = NextPlaidVectorStore(url="http://localhost:8080", index_name="pdf_pages", embedding=embeddings)

pages = [Image.open(f"page_{i}.png") for i in range(10)]
store.add_images(
    pages,
    metadatas=[{"file": "report.pdf", "page": i} for i in range(10)],
)

docs = store.similarity_search("quarterly revenue breakdown", k=3)
```

### Mixed text + image indexing

```python
store.add_items(
    ["Introduction text", Image.open("figure1.png"), "Conclusion text"],
    metadatas=[
        {"source_type": "text"},
        {"source_type": "image", "caption": "Figure 1"},
        {"source_type": "text"},
    ],
)
```

---

## API Reference

### `NextPlaidVectorStore`

```python
NextPlaidVectorStore(
    url="http://localhost:8080",    # NextPlaid server base URL
    index_name="my_index",          # Index name (created if not exists)
    embedding=my_embeddings,        # ColBERT-compatible embeddings object
    nbits=4,                        # Quantization: 2 or 4 bits (default 4)
    create_index_if_not_exists=True,
    write_timeout=0.0,              # >0 = synchronous writes (blocks until indexed)
)
```

#### Write methods

| Method | Input | Notes |
|---|---|---|
| `add_texts(texts, metadatas, ids)` | `List[str]` | Standard LangChain method |
| `add_images(images, metadatas, ids)` | `List[PIL.Image]` | Requires `embed_images` on embeddings |
| `add_items(items, metadatas, ids)` | `List[str \| PIL.Image]` | Mixed batch, two encoder calls |
| `delete(ids)` | `List[str]` | Delete by LangChain ID |

All write methods use **upsert semantics**: if an ID already exists it is deleted then reinserted.

#### Search methods

| Method | Returns |
|---|---|
| `similarity_search(query, k, filter)` | `List[Document]` |
| `similarity_search_with_score(query, k, filter)` | `List[Tuple[Document, float]]` |
| `similarity_search_by_vector(embedding, k, filter)` | `List[Document]` |
| `similarity_search_by_vector_with_score(embedding, k, filter)` | `List[Tuple[Document, float]]` |

Queries are always text strings — even when the corpus contains images, the query is encoded via `embed_query(text)`.

#### Metadata filtering

```python
# Simple equality filter (dict syntax)
docs = store.similarity_search("query", k=5, filter={"source": "wiki"})
```

#### LangChain retriever

```python
retriever = store.as_retriever(search_kwargs={"k": 10})

# Use in any LangChain chain
from langchain_core.runnables import RunnablePassthrough
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

---

## Embeddings shape contract

The `embedding` object passed to `NextPlaidVectorStore` must implement:

```python
# Required for text indexing and all queries
embed_documents(texts: List[str]) -> List[List[List[float]]]      # (n_docs, n_tokens, dim)
embed_query(text: str)            -> List[List[float]]             # (n_tokens, dim)

# Required only for add_images / add_items
embed_images(images: List[PIL.Image]) -> List[List[List[float]]]  # (n_imgs, n_patches, dim)
```

No base class is required — duck typing is used throughout.

---

## Synchronous vs asynchronous writes

By default, `add_texts` / `add_images` return immediately after the HTTP request is accepted (the server indexes asynchronously). If you need to guarantee the data is queryable before proceeding:

```python
store = NextPlaidVectorStore(
    ...,
    write_timeout=60.0,  # block up to 60s waiting for indexing to complete
)
```

With `write_timeout > 0`, writes use `?wait=true` on the server endpoint and poll until the documents appear in metadata.

---

## Development

```bash
cd libs/plaid
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT