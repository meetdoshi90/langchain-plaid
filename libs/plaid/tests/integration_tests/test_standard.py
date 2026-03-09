from typing import Generator, List

import pytest
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_plaid import NextPlaidVectorStore


class FakeColBERTEmbeddings:
    """Text-dependent embeddings so similarity ranking is deterministic."""

    dim = 128
    n_tokens = 3

    def _text_to_vec(self, text: str) -> List[List[float]]:
        # Each token gets a vector seeded by the text hash — distinct per text
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = __import__("random").Random(seed)
        return [
            [rng.gauss(0, 1) for _ in range(self.dim)] for _ in range(self.n_tokens)
        ]

    def embed_documents(self, texts):
        return [self._text_to_vec(t) for t in texts]

    def embed_query(self, text):
        return self._text_to_vec(text)


class TestNextPlaidStandard(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[NextPlaidVectorStore, None, None]:
        store = NextPlaidVectorStore(
            url="http://localhost:8080",
            index_name="langchain_standard_test",
            embedding=FakeColBERTEmbeddings(),
            write_timeout=30,  # <-- wait for async writes
        )
        yield store
        import requests

        requests.delete("http://localhost:8080/indices/langchain_standard_test")
