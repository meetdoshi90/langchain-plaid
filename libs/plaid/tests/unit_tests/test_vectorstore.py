"""Unit tests for NextPlaidVectorStore.

These tests mock HTTP calls and do not require a running NextPlaid server.
"""

from unittest.mock import MagicMock, patch

import pytest

from langchain_plaid.vectorstores.plaid import NextPlaidVectorStore, _dict_to_sql_filter

# ---------------------------------------------------------------------------
# Fake embeddings: returns deterministic multi-vector matrices
# ---------------------------------------------------------------------------


class FakeColBERTEmbeddings:
    """Fake ColBERT embeddings for unit testing.

    Returns 3 tokens × 4 dimensions per document/query.
    """

    dim = 4
    n_tokens = 3

    def embed_documents(self, texts):
        return [
            [[float(i + j) for j in range(self.dim)] for i in range(self.n_tokens)]
            for _ in texts
        ]

    def embed_query(self, text):
        return [[float(i) for i in range(self.dim)] for _ in range(self.n_tokens)]


FAKE_EMBEDDINGS = FakeColBERTEmbeddings()

# ---------------------------------------------------------------------------
# Helper: mock a requests.Response
# ---------------------------------------------------------------------------


def _mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# _dict_to_sql_filter
# ---------------------------------------------------------------------------


def test_dict_to_sql_filter_single():
    condition, params = _dict_to_sql_filter({"source": "wiki"})
    assert condition == "source = ?"
    assert params == ["wiki"]


def test_dict_to_sql_filter_multiple():
    condition, params = _dict_to_sql_filter({"source": "wiki", "year": 2024})
    assert "source = ?" in condition
    assert "year = ?" in condition
    assert "AND" in condition
    assert "wiki" in params
    assert 2024 in params


def test_dict_to_sql_filter_empty():
    condition, params = _dict_to_sql_filter({})
    assert condition == ""
    assert params == []


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_init_creates_index_when_not_found(mock_post, mock_get):
    mock_get.return_value = _mock_response(status_code=404)
    mock_post.return_value = _mock_response(status_code=200)

    NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
    )

    mock_get.assert_called_once_with("http://localhost:8080/indices/test")
    mock_post.assert_called_once()
    call_json = mock_post.call_args.kwargs["json"]
    assert call_json["name"] == "test"
    assert call_json["config"]["nbits"] == 4


@patch("langchain_plaid.vectorstores.plaid.requests.get")
def test_init_skips_create_when_index_exists(mock_get):
    mock_get.return_value = _mock_response(status_code=200)

    NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
    )
    # POST should NOT have been called
    mock_get.assert_called_once()


@patch("langchain_plaid.vectorstores.plaid.requests.get")
def test_init_skip_create_flag(mock_get):
    NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# add_texts
# ---------------------------------------------------------------------------


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.delete")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_add_texts_returns_ids(mock_post, mock_delete, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_delete.return_value = _mock_response(status_code=404)
    mock_post.return_value = _mock_response(status_code=202)

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    ids = store.add_texts(["hello", "world"], ids=["id1", "id2"])
    assert ids == ["id1", "id2"]


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.delete")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_add_texts_generates_ids_when_none(mock_post, mock_delete, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_delete.return_value = _mock_response(status_code=404)
    mock_post.return_value = _mock_response(status_code=202)

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    ids = store.add_texts(["hello", "world"])
    assert len(ids) == 2
    # Should be UUID4 strings
    import uuid

    for id_ in ids:
        uuid.UUID(id_)  # raises if not valid UUID


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.delete")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_add_texts_payload_structure(mock_post, mock_delete, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_delete.return_value = _mock_response(status_code=404)
    mock_post.return_value = _mock_response(status_code=202)

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    store.add_texts(
        ["hello"],
        metadatas=[{"source": "wiki"}],
        ids=["abc"],
    )

    call_json = mock_post.call_args.kwargs["json"]
    assert len(call_json["documents"]) == 1
    assert "embeddings" in call_json["documents"][0]
    assert call_json["metadata"][0]["langchain_id"] == "abc"
    assert call_json["metadata"][0]["page_content"] == "hello"
    assert call_json["metadata"][0]["source"] == "wiki"


@patch("langchain_plaid.vectorstores.plaid.requests.get")
def test_add_texts_empty_returns_empty(mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    ids = store.add_texts([])
    assert ids == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.delete")
def test_delete_sends_correct_condition(mock_delete, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_delete.return_value = _mock_response(status_code=202)

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    result = store.delete(["id1", "id2"])
    assert result is True
    assert mock_delete.call_count == 1  # single batched request

    call_json = mock_delete.call_args.kwargs["json"]
    assert "IN" in call_json["condition"]
    assert "id1" in call_json["parameters"]
    assert "id2" in call_json["parameters"]


@patch("langchain_plaid.vectorstores.plaid.requests.get")
def test_delete_empty_ids(mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    assert store.delete([]) is True
    assert store.delete(None) is True


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

FAKE_SEARCH_RESPONSE = {
    "results": [
        {
            "query_id": 0,
            "document_ids": [0, 1],
            "scores": [18.5, 12.3],
            "metadata": [
                {
                    "langchain_id": "abc",
                    "page_content": "hello world",
                    "source": "wiki",
                    "_subset_": 0,
                },
                {
                    "langchain_id": "def",
                    "page_content": "foo bar",
                    "_subset_": 1,
                },
            ],
        }
    ],
    "num_queries": 1,
}


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_similarity_search_returns_documents(mock_post, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_post.return_value = _mock_response(
        status_code=200, json_data=FAKE_SEARCH_RESPONSE
    )

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    docs = store.similarity_search("hello", k=2)

    assert len(docs) == 2
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "wiki"
    # Internal fields must be stripped
    assert "langchain_id" not in docs[0].metadata
    assert "_subset_" not in docs[0].metadata


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_similarity_search_with_score(mock_post, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_post.return_value = _mock_response(
        status_code=200, json_data=FAKE_SEARCH_RESPONSE
    )

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    results = store.similarity_search_with_score("hello", k=2)

    assert len(results) == 2
    doc, score = results[0]
    assert doc.page_content == "hello world"
    assert score == pytest.approx(18.5)


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_similarity_search_uses_filtered_endpoint_with_filter(mock_post, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_post.return_value = _mock_response(
        status_code=200, json_data=FAKE_SEARCH_RESPONSE
    )

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    store.similarity_search("hello", k=2, filter={"source": "wiki"})

    call_url = mock_post.call_args.args[0]
    assert "filtered" in call_url

    call_json = mock_post.call_args.kwargs["json"]
    assert call_json["filter_condition"] == "source = ?"
    assert call_json["filter_parameters"] == ["wiki"]


@patch("langchain_plaid.vectorstores.plaid.requests.get")
@patch("langchain_plaid.vectorstores.plaid.requests.post")
def test_similarity_search_uses_plain_endpoint_without_filter(mock_post, mock_get):
    mock_get.return_value = _mock_response(status_code=200)
    mock_post.return_value = _mock_response(
        status_code=200, json_data=FAKE_SEARCH_RESPONSE
    )

    store = NextPlaidVectorStore(
        url="http://localhost:8080",
        index_name="test",
        embedding=FAKE_EMBEDDINGS,
        create_index_if_not_exists=False,
    )
    store.similarity_search("hello", k=2)

    call_url = mock_post.call_args.args[0]
    assert "filtered" not in call_url
