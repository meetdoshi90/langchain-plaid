"""
SciFact evaluation with pytrec_eval
=====================================
Evaluates NextPlaid + ColBERT retrieval on SciFact using proper BEIR metrics.

pip install pytrec-eval-terrier pylate beir datasets langchain-plaid torch

Usage:
    # Evaluate existing index (default):
    python eval_beir_colbertzero.py

    # Delete existing index, rebuild from scratch, then evaluate:
    python eval_beir_colbertzero.py --rebuild
"""

import argparse
import time
import requests
import torch
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict

import pytrec_eval

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from pylate import models

from langchain_plaid import NextPlaidVectorStore

# ── Config ────────────────────────────────────────────────────────────────────
NEXT_PLAID_URL  = "http://localhost:8080"
INDEX_NAME      = "scifact_test"
COLBERT_MODEL   = "lightonai/ColBERT-Zero"
BATCH_SIZE      = 64
INDEX_BATCH_SIZE = 1000   # docs per add_texts call
TOP_K           = 100     # retrieve more for NDCG/MAP accuracy; we report @10
# ─────────────────────────────────────────────────────────────────────────────


class PyLateColBERTEmbeddings:
    def __init__(self, model_name: str = COLBERT_MODEL, batch_size: int = BATCH_SIZE):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ColBERT] Loading model '{model_name}' on {device} …")
        self._model = models.ColBERT(model_name_or_path=model_name, device=device)
        self._batch_size = batch_size
        print("[ColBERT] Model loaded.")

    def embed_documents(self, texts: List[str]) -> List[List[List[float]]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            is_query=False,
            prompt_name="document",
            show_progress_bar=len(texts) > 50,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[List[float]]:
        embeddings = self._model.encode(
            [text],
            batch_size=1,
            is_query=True,
            prompt_name="query",
            show_progress_bar=False,
        )
        return embeddings[0].tolist()


def wait_for_index_stable(url: str, index_name: str, expected_docs: int, stable_secs: int = 10):
    """Poll until doc count reaches expected_docs and stays stable."""
    import requests
    prev = -1
    stable_since = None
    print(f"[Index] Waiting for {expected_docs} docs to be indexed …")
    while True:
        try:
            r = requests.get(f"{url}/indices/{index_name}", timeout=10)
            r.raise_for_status()
            info = r.json()
            count = info.get("num_documents", 0)
            print(f"\r[Index] {count}/{expected_docs} docs indexed …", end="", flush=True)
            if count >= expected_docs:
                if count == prev:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= stable_secs:
                        print(f"\n[Index] Stable at {count} docs.")
                        return
                else:
                    stable_since = None
            else:
                stable_since = None
            prev = count
        except Exception as e:
            print(f"\n[Index] Warning: {e}")
        time.sleep(2)


def delete_index_if_exists(url: str, index_name: str):
    """Delete the index if it already exists, then wait until it's gone."""
    r = requests.get(f"{url}/indices/{index_name}", timeout=10)
    if r.status_code == 404:
        print(f"[Index] Index '{index_name}' does not exist, nothing to delete.")
        return
    r.raise_for_status()
    print(f"[Index] Deleting existing index '{index_name}' …")
    r = requests.delete(f"{url}/indices/{index_name}", timeout=30)
    r.raise_for_status()
    # Wait until the index is gone
    for _ in range(30):
        time.sleep(1)
        r = requests.get(f"{url}/indices/{index_name}", timeout=10)
        if r.status_code == 404:
            print(f"[Index] Index '{index_name}' deleted.")
            return
    raise RuntimeError(f"Index '{index_name}' still exists after deletion timeout")


def index_corpus(
    store: NextPlaidVectorStore,
    corpus: dict,
    batch_size: int = INDEX_BATCH_SIZE,
):
    """Encode and add all corpus documents to the vector store in batches."""
    doc_ids   = list(corpus.keys())
    doc_texts = [corpus[d]["title"] + " " + corpus[d]["text"] for d in doc_ids]

    print(f"[Index] Indexing {len(doc_ids):,} documents in batches of {batch_size} …")
    for start in tqdm(range(0, len(doc_ids), batch_size), desc="[Index] Batches"):
        end       = min(start + batch_size, len(doc_ids))
        store.add_texts(
            doc_texts[start:end],
            metadatas=[{"beir_id": bid} for bid in doc_ids[start:end]],
            ids=doc_ids[start:end],
        )
    print("[Index] All batches submitted.")


def run_retrieval(
    store: NextPlaidVectorStore,
    queries: Dict[str, str],
    top_k: int = TOP_K,
) -> Dict[str, Dict[str, float]]:
    """
    Run all queries and return a pytrec_eval-compatible run dict.

    run[qid][doc_id] = score
    """
    run = {}
    qids = list(queries.keys())

    for qid in tqdm(qids, desc="[Eval] Retrieving"):
        results = store.similarity_search_with_score(queries[qid], k=top_k)

        run[qid] = {}
        for doc, score in results:
            beir_id = doc.metadata.get("beir_id")
            if beir_id:
                # pytrec_eval needs float scores; higher = more relevant
                run[qid][str(beir_id)] = float(score)

    return run


def evaluate(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 5, 10, 100],
) -> Dict[str, float]:
    """
    Evaluate using pytrec_eval. Returns NDCG@k, Recall@k, MAP@k, MRR@10.

    qrels format: qrels[qid][doc_id] = relevance_int  (BEIR uses 0/1/2)
    run   format: run[qid][doc_id]   = score_float
    """
    # pytrec_eval expects string keys and int relevance
    qrels_str = {
        str(qid): {str(did): int(rel) for did, rel in rels.items()}
        for qid, rels in qrels.items()
        if rels  # skip queries with no relevant docs
    }

    # Filter run to only queries that have qrels
    run_str = {
        str(qid): {str(did): float(s) for did, s in docs.items()}
        for qid, docs in run.items()
        if str(qid) in qrels_str
    }

    # Diagnostics: how many queries have at least one result?
    queries_with_results = sum(1 for v in run_str.values() if v)
    queries_total = len(qrels_str)
    print(f"\n[Eval] Queries with results: {queries_with_results}/{queries_total}")

    # Queries with results but no relevant doc retrieved
    zero_recall = 0
    for qid, retrieved in run_str.items():
        relevant = set(qrels_str.get(qid, {}).keys())
        if not relevant & set(retrieved.keys()):
            zero_recall += 1
    print(f"[Eval] Queries with zero relevant docs retrieved (@{max(k_values)}): {zero_recall}/{queries_total}")

    # Build metrics list for pytrec_eval
    metrics = set()
    for k in k_values:
        metrics.add(f"ndcg_cut_{k}")
        metrics.add(f"recall_{k}")
        metrics.add(f"map_cut_{k}")
    metrics.add("recip_rank")  # MRR

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_str, metrics)
    results = evaluator.evaluate(run_str)

    # Average over queries
    agg: Dict[str, float] = defaultdict(float)
    n = len(results)
    for qid_results in results.values():
        for metric, val in qid_results.items():
            agg[metric] += val
    agg = {k: v / n for k, v in agg.items()}

    return agg


def print_results(agg: Dict[str, float], k_values: List[int] = [1, 5, 10, 100]):
    print("\n" + "=" * 55)
    print(f"  {'Metric':<25} {'Score':>10}")
    print("=" * 55)
    for k in k_values:
        for prefix in ["ndcg_cut", "recall", "map_cut"]:
            key = f"{prefix}_{k}"
            if key in agg:
                label = f"{prefix.replace('_cut', '')}@{k}".upper()
                print(f"  {label:<25} {agg[key]*100:>9.2f}%")
    if "recip_rank" in agg:
        print(f"  {'MRR':<25} {agg['recip_rank']*100:>9.2f}%")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing index and rebuild it from scratch before evaluating.",
    )
    args = parser.parse_args()

    # 1. Load data
    print("[BEIR] Downloading / loading SciFact …")
    data_path = util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "./evaluation_datasets/",
    )
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"[BEIR] Corpus: {len(corpus):,}  |  Queries: {len(queries):,}")

    # 2. Build embeddings wrapper
    embeddings = PyLateColBERTEmbeddings(model_name=COLBERT_MODEL, batch_size=BATCH_SIZE)

    # 3. Optionally delete and rebuild the index
    if args.rebuild:
        delete_index_if_exists(NEXT_PLAID_URL, INDEX_NAME)

    # 4. Connect to store (creates index if it doesn't exist)
    print(f"\n[Store] Connecting to {NEXT_PLAID_URL}, index='{INDEX_NAME}'")
    store = NextPlaidVectorStore(
        url=NEXT_PLAID_URL,
        index_name=INDEX_NAME,
        embedding=embeddings,
        create_index_if_not_exists=True,
    )

    # 5. Index corpus if rebuilding (or if the index is empty)
    if args.rebuild:
        index_corpus(store, corpus)

    # 6. Wait for all docs to be indexed and stable
    wait_for_index_stable(NEXT_PLAID_URL, INDEX_NAME, expected_docs=len(corpus))

    # 7. Run retrieval over all 300 queries
    run = run_retrieval(store, queries, top_k=TOP_K)

    # 8. Spot-check: print a few retrieved IDs vs qrels for first query
    qid0 = list(queries.keys())[0]
    print(f"\n[Debug] Query '{qid0}': {queries[qid0]}")
    print(f"  Relevant docs: {list(qrels.get(qid0, {}).keys())}")
    print(f"  Top-5 retrieved: {list(run.get(qid0, {}).keys())[:5]}")

    # 9. Evaluate with pytrec_eval
    agg = evaluate(run, qrels, k_values=[1, 5, 10, 100])
    print_results(agg, k_values=[1, 5, 10, 100])


if __name__ == "__main__":
    main()