"""Integration tests for NextPlaidVectorStore.

Requires a running NextPlaid API server. Set NEXT_PLAID_URL to override
the default (http://localhost:8080).

Run with:
    pytest tests/integration_tests/ -v

The server can be started with:
    docker compose up -d
or:
    cargo run --release -p next-plaid-api
"""
import os
import time

import pytest
import requests

from langchain_plaid.vectorstores.plaid import NextPlaidVectorStore

NEXT_PLAID_URL = os.environ.get("NEXT_PLAID_URL", "http://localhost:8080")
TEST_INDEX = "langchain_integration_test"


# ---------------------------------------------------------------------------
# Fake ColBERT embeddings — deterministic, dim=4, 3 tokens per doc/query
# ---------------------------------------------------------------------------

class FakeColBERTEmbeddings:
    dim = 4
    n_tokens = 3

    def embed_documents(self, texts):
        # Different embeddings per text so similarity scores differ
        return [
            [
                [float(ord(text[0]) + i + j) for j in range(self.dim)]
                for i in range(self.n_tokens)
            ]
            for text in texts
        ]

    def embed_query(self, text):
        return [
            [float(ord(text[0]) + i + j) for j in range(self.dim)]
            for i in range(self.n_tokens)
        ]


EMBEDDINGS = FakeColBERTEmbeddings()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    try:
        resp = requests.get(f"{NEXT_PLAID_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def cleanup_index():
    """Delete the test index before and after each test."""
    requests.delete(f"{NEXT_PLAID_URL}/indices/{TEST_INDEX}")
    yield
    requests.delete(f"{NEXT_PLAID_URL}/indices/{TEST_INDEX}")


@pytest.fixture
def store():
    return NextPlaidVectorStore(
        url=NEXT_PLAID_URL,
        index_name=TEST_INDEX,
        embedding=EMBEDDINGS,
        create_index_if_not_exists=True,
    )


def _wait_for_index(url: str, index: str, min_docs: int = 1, timeout: float = 10.0):
    """Poll until the index has at least min_docs documents."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/indices/{index}")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("num_documents", 0) >= min_docs:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    pytest.fail(
        f"Index '{index}' did not reach {min_docs} docs within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_add_and_search(store):
    texts = ["hello world", "foo bar baz", "machine learning is great"]
    ids = store.add_texts(texts, metadatas=[{"source": f"doc{i}"} for i in range(3)])

    assert len(ids) == 3

    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=3)

    docs = store.similarity_search("hello", k=2)
    assert len(docs) == 2
    assert all(isinstance(d.page_content, str) for d in docs)
    assert all("langchain_id" not in d.metadata for d in docs)
    assert all("_subset_" not in d.metadata for d in docs)


@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_similarity_search_with_score(store):
    store.add_texts(["hello world", "unrelated content"])
    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=2)

    results = store.similarity_search_with_score("hello", k=2)
    assert len(results) == 2
    for doc, score in results:
        assert isinstance(score, float)
        assert score >= 0


@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_similarity_search_with_filter(store):
    store.add_texts(
        ["hello from wiki", "hello from news"],
        metadatas=[{"source": "wiki"}, {"source": "news"}],
    )
    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=2)

    docs = store.similarity_search("hello", k=5, filter={"source": "wiki"})
    assert all(d.metadata.get("source") == "wiki" for d in docs)


@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_delete(store):
    ids = store.add_texts(["to be deleted", "to be kept"])
    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=2)

    result = store.delete([ids[0]])
    assert result is True

    # Give delete batch time to process
    time.sleep(3.0)

    resp = requests.get(f"{NEXT_PLAID_URL}/indices/{TEST_INDEX}")
    remaining = resp.json()["num_documents"]
    assert remaining == 1


@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_from_texts():
    store = NextPlaidVectorStore.from_texts(
        texts=["alpha", "beta", "gamma"],
        embedding=EMBEDDINGS,
        url=NEXT_PLAID_URL,
        index_name=TEST_INDEX,
        metadatas=[{"idx": i} for i in range(3)],
    )
    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=3)

    docs = store.similarity_search("alpha", k=1)
    assert len(docs) == 1


@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_from_documents():
    from langchain_core.documents import Document

    documents = [
        Document(page_content="doc one", metadata={"id": "1"}),
        Document(page_content="doc two", metadata={"id": "2"}),
    ]
    store = NextPlaidVectorStore.from_documents(
        documents=documents,
        embedding=EMBEDDINGS,
        url=NEXT_PLAID_URL,
        index_name=TEST_INDEX,
    )
    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=2)

    docs = store.similarity_search("doc", k=2)
    assert len(docs) == 2
    assert all("id" in d.metadata for d in docs)


@pytest.mark.skipif(not _server_available(), reason="NextPlaid server not available")
def test_metadata_preserved(store):
    store.add_texts(
        ["test document"],
        metadatas=[{"category": "science", "year": 2024}],
        ids=["meta-test-id"],
    )
    _wait_for_index(NEXT_PLAID_URL, TEST_INDEX, min_docs=1)

    docs = store.similarity_search("test", k=1)
    assert docs[0].metadata["category"] == "science"
    assert docs[0].metadata["year"] == 2024