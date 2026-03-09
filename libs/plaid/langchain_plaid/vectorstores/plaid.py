"""NextPlaid vector store integration for LangChain."""
from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type

import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class NextPlaidVectorStore(VectorStore):
    """LangChain vector store backed by NextPlaid multi-vector search.

    NextPlaid uses ColBERT-style late interaction (MaxSim) for retrieval,
    storing multiple vectors per document instead of a single embedding.
    This requires a ColBERT-compatible embeddings object — standard single-vector
    LangChain embeddings (OpenAI, HuggingFace etc.) will not work.

    The embeddings object must implement:
        embed_documents(texts: List[str]) -> List[List[List[float]]]
            Returns one multi-vector matrix per document.
        embed_query(text: str) -> List[List[float]]
            Returns a single multi-vector matrix for the query.

    For image support (e.g. ColPali), the embeddings object should also implement:
        embed_images(images: List[PIL.Image.Image]) -> List[List[List[float]]]
            Returns one multi-vector matrix per image.

    Args:
        url: Base URL of the NextPlaid API server (e.g. "http://localhost:8080").
        index_name: Name of the NextPlaid index to use.
        embedding: ColBERT-compatible embeddings object (see shape contract above).
        nbits: Quantization bits for index creation (2 or 4, default 4).
        create_index_if_not_exists: Create the index if it does not exist yet.

    Example:
        .. code-block:: python

            from langchain_plaid import NextPlaidVectorStore

            store = NextPlaidVectorStore(
                url="http://localhost:8080",
                index_name="my_docs",
                embedding=my_colbert_embeddings,
            )
            store.add_texts(["hello world", "foo bar"])
            docs = store.similarity_search("hello")
    """

    def __init__(
        self,
        url: str,
        index_name: str,
        embedding: Any,
        nbits: int = 4,
        create_index_if_not_exists: bool = True,
        write_timeout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._url = url.rstrip("/")
        self._index_name = index_name
        self._embedding = embedding
        self._nbits = nbits
        self._write_timeout = write_timeout

        if create_index_if_not_exists:
            self._create_index_if_not_exists()

    # ------------------------------------------------------------------
    # LangChain required property
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Any:
        return self._embedding

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _create_index_if_not_exists(self) -> None:
        resp = requests.get(f"{self._url}/indices/{self._index_name}")
        if resp.status_code == 404:
            resp = requests.post(
                f"{self._url}/indices",
                json={
                    "name": self._index_name,
                    "config": {"nbits": self._nbits},
                },
            )
            if resp.status_code == 409:
                return
            resp.raise_for_status()
        elif resp.status_code != 200:
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def _prepare_ids_and_metadatas(
        self,
        n: int,
        ids: Optional[List[str]],
        metadatas: Optional[List[dict]],
        page_contents: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[dict]]:
        """Validate and fill in ids/metadatas, attaching page_content when provided."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        else:
            ids = [i if i is not None else str(uuid.uuid4()) for i in ids]

        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        if len(ids) != n:
            raise ValueError(f"ids length ({len(ids)}) must match items length ({n})")
        if len(metadatas) != n:
            raise ValueError(f"metadatas length ({len(metadatas)}) must match items length ({n})")

        if page_contents is not None:
            metadatas = [
                {"langchain_id": doc_id, "page_content": text, **meta}
                for doc_id, text, meta in zip(ids, page_contents, metadatas)
            ]
        else:
            metadatas = [
                {"langchain_id": doc_id, **meta}
                for doc_id, meta in zip(ids, metadatas)
            ]

        return ids, metadatas

    def _add_embeddings(
        self,
        embeddings: List[List[List[float]]],
        metadatas: List[dict],
        ids: List[str],
    ) -> List[str]:
        """Shared implementation: upsert pre-computed embeddings into the index.

        Handles delete-before-insert (upsert semantics), constructs the payload,
        and fires the /update request. Called by add_texts, add_images, add_items.
        """
        self._delete_by_ids_silent(ids)
        if self._write_timeout > 0:
            self._wait_for_delete(ids, timeout=self._write_timeout)

        documents = [{"embeddings": emb} for emb in embeddings]

        use_wait = self._write_timeout > 0
        resp = requests.post(
            f"{self._url}/indices/{self._index_name}/update",
            params={"wait": "true"} if use_wait else {},
            json={"documents": documents, "metadata": metadatas},
            timeout=self._write_timeout + 10 if use_wait else None,
        )
        resp.raise_for_status()
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the index.

        If IDs are provided and already exist, the documents are replaced
        (delete-then-insert), making add_texts idempotent for the same IDs.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        ids, meta_payload = self._prepare_ids_and_metadatas(
            len(texts_list), ids, metadatas, page_contents=texts_list
        )
        embeddings = self._embedding.embed_documents(texts_list)
        return self._add_embeddings(embeddings, meta_payload, ids)

    def add_images(
        self,
        images: List[Any],  # List[PIL.Image.Image]
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add images to the index.

        Requires the embeddings object to implement:
            embed_images(images: List[PIL.Image.Image]) -> List[List[List[float]]]

        ``source_type="image"`` is injected into metadata automatically unless
        already set. ``page_content`` is stored as an empty string.

        Example:
            .. code-block:: python

                from PIL import Image
                imgs = [Image.open("page1.png"), Image.open("page2.png")]
                store.add_images(
                    imgs,
                    metadatas=[{"file": "doc.pdf", "page": 1},
                               {"file": "doc.pdf", "page": 2}],
                )
        """
        if not images:
            return []

        if not hasattr(self._embedding, "embed_images"):
            raise TypeError(
                f"{type(self._embedding).__name__} does not implement embed_images(). "
                "Use a ColPali-compatible embeddings class."
            )

        if metadatas is None:
            metadatas = [{"source_type": "image"} for _ in images]
        else:
            metadatas = [
                {"source_type": "image", **m} for m in metadatas
            ]

        ids, meta_payload = self._prepare_ids_and_metadatas(
            len(images), ids, metadatas, page_contents=[""] * len(images)
        )
        embeddings = self._embedding.embed_images(images)
        return self._add_embeddings(embeddings, meta_payload, ids)

    def add_items(
        self,
        items: List[Any],  # List[Union[str, PIL.Image.Image]]
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add a mixed list of texts and images to the index.

        Requires the embeddings object to implement both embed_documents()
        and embed_images(). Items are encoded in two batched calls (one for
        all texts, one for all images) and recombined in original order.

        Example:
            .. code-block:: python

                from PIL import Image
                store.add_items(
                    ["some text", Image.open("fig1.png"), "more text"],
                    metadatas=[
                        {"source_type": "text"},
                        {"source_type": "image", "page": 1},
                        {"source_type": "text"},
                    ],
                )
        """
        if not items:
            return []

        if not hasattr(self._embedding, "embed_images"):
            raise TypeError(
                f"{type(self._embedding).__name__} does not implement embed_images(). "
                "Use a ColPali-compatible embeddings class for mixed input."
            )

        n = len(items)
        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        # Split into text and image buckets, preserving original indices
        text_indices  = [i for i, x in enumerate(items) if isinstance(x, str)]
        image_indices = [i for i, x in enumerate(items) if not isinstance(x, str)]

        texts  = [items[i] for i in text_indices]
        images = [items[i] for i in image_indices]

        # Encode each bucket in one batched call
        text_embeddings  = self._embedding.embed_documents(texts)  if texts  else []
        image_embeddings = self._embedding.embed_images(images)     if images else []

        # Reconstruct in original order
        text_iter  = iter(text_embeddings)
        image_iter = iter(image_embeddings)
        embeddings: List[List[List[float]]] = [None] * n  # type: ignore[list-item]
        page_contents: List[str] = [""] * n

        for i in text_indices:
            embeddings[i]    = next(text_iter)
            page_contents[i] = items[i]

        for i in image_indices:
            embeddings[i] = next(image_iter)
            if "source_type" not in metadatas[i]:
                metadatas[i] = {"source_type": "image", **metadatas[i]}

        ids, meta_payload = self._prepare_ids_and_metadatas(
            n, ids, metadatas, page_contents=page_contents
        )
        return self._add_embeddings(embeddings, meta_payload, ids)

    def delete(self, ids=None, **kwargs):
        if not ids:
            return True
        self._delete_by_ids_silent(ids)
        if self._write_timeout > 0:
            self._wait_for_delete(ids, timeout=self._write_timeout)
        return True

    def _wait_for_delete(self, ids: List[str], timeout: float = 30.0) -> None:
        """Poll until all IDs have disappeared from metadata."""
        import time
        valid_ids = [i for i in ids if i is not None]
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = {doc.id for doc in self.get_by_ids(valid_ids)}
            if not any(i in found for i in valid_ids):
                return
            time.sleep(0.5)

    def _delete_by_ids_silent(self, ids: List[str]) -> None:
        """Delete by langchain_id in a single batched request."""
        valid_ids = [i for i in ids if i is not None]
        if not valid_ids:
            return

        resp = requests.delete(
            f"{self._url}/indices/{self._index_name}/documents",
            json={
                "condition": f"langchain_id IN ({','.join('?' * len(valid_ids))})",
                "parameters": valid_ids,
            },
        )
        if resp.status_code in (404, 503):
            return
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # get_by_ids
    # ------------------------------------------------------------------

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Fetch documents by their LangChain IDs using the metadata/get endpoint."""
        if not ids:
            return []

        valid_ids = [i for i in ids if i is not None]
        if not valid_ids:
            return []

        try:
            resp = requests.post(
                f"{self._url}/indices/{self._index_name}/metadata/get",
                json={
                    "condition": f"langchain_id IN ({','.join('?' * len(valid_ids))})",
                    "parameters": valid_ids,
                },
            )
            if resp.status_code in (404, 400):
                return []
            resp.raise_for_status()
        except requests.HTTPError:
            return []

        data = resp.json()
        rows = data.get("metadata", [])
        if not rows:
            return []

        docs = []
        for row in rows:
            meta = {k: v for k, v in row.items() if v is not None}
            doc_id = meta.pop("langchain_id", None)
            page_content = meta.pop("page_content", "")
            meta.pop("_subset_", None)
            docs.append(Document(page_content=page_content, metadata=meta, id=doc_id))

        return docs

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return the top-k most similar documents to the query."""
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents and their MaxSim scores for the given query."""
        query_embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            query_embedding, k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector( # type: ignore[override]
        self,
        embedding: List[List[float]],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search using a pre-computed multi-vector query embedding."""
        docs_and_scores = self.similarity_search_by_vector_with_score(
            embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[List[float]],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search using a pre-computed multi-vector query embedding, returning scores.
        Returns an empty list if the index has no documents yet (404).
        """
        query_payload = {"embeddings": embedding}

        if filter:
            user_condition, user_parameters = _dict_to_sql_filter(filter)
            endpoint = f"{self._url}/indices/{self._index_name}/search/filtered"
            payload: dict = {
                "queries": [query_payload],
                "params": {"top_k": k},
                "filter_condition": user_condition,
                "filter_parameters": user_parameters,
            }
        else:
            endpoint = f"{self._url}/indices/{self._index_name}/search"
            payload = {
                "queries": [query_payload],
                "params": {"top_k": k},
            }

        resp = requests.post(endpoint, json=payload)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        result = data["results"][0]
        docs_and_scores: List[Tuple[Document, float]] = []
        for meta, score in zip(result["metadata"], result["scores"]):
            if meta is None:
                meta = {}
            else:
                meta = {key: v for key, v in meta.items() if v is not None}
            page_content = meta.pop("page_content", "")
            doc_id = meta.pop("langchain_id", None)
            if doc_id is None:
                continue
            meta.pop("_subset_", None)
            docs_and_scores.append(
                (Document(page_content=page_content, metadata=meta, id=doc_id), float(score))
            )
        return docs_and_scores

    # ------------------------------------------------------------------
    # Class methods (constructors)
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type[NextPlaidVectorStore],
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        url: str = "http://localhost:8080",
        index_name: str = "langchain",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> NextPlaidVectorStore:
        """Create a NextPlaidVectorStore from a list of texts."""
        store = cls(
            url=url,
            index_name=index_name,
            embedding=embedding,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type[NextPlaidVectorStore],
        documents: List[Document],
        embedding: Any,
        url: str = "http://localhost:8080",
        index_name: str = "langchain",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> NextPlaidVectorStore:
        """Create a NextPlaidVectorStore from a list of Documents."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            url=url,
            index_name=index_name,
            ids=ids,
            **kwargs,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dict_to_sql_filter(filter: dict) -> Tuple[str, List[Any]]:
    """Convert a flat dict to a SQL WHERE clause and parameter list.

    Example:
        >>> _dict_to_sql_filter({"source": "wiki", "year": 2024})
        ("source = ? AND year = ?", ["wiki", 2024])
    """
    clauses = [f"{k} = ?" for k in filter]
    parameters = list(filter.values())
    return " AND ".join(clauses), parameters