"""
ColModernVBert evaluation on ViDoRe v3 using NextPlaid
=======================================================
Indexes corpus page images into NextPlaid via langchain-plaid,
retrieves with text queries, and reports NDCG/Recall/Hit-Rate.

Requirements:
    pip install colpali-engine pylate pytrec-eval-terrier datasets langchain-plaid torch

Usage:
    # Start NextPlaid server first:
    #   cargo run --release -p next-plaid-api -- --index-dir /tmp/vidore-indices
    python eval_vidore_colmodernvbert.py [--rebuild]
"""

import argparse
import json
import os
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import requests
from PIL import Image
from tqdm import tqdm
import pytrec_eval
from datasets import load_dataset
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

from langchain_plaid import NextPlaidVectorStore

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
NEXT_PLAID_URL  = "http://localhost:8080"
IMAGE_MULTI_MODEL = "ModernVBERT/colmodernvbert"
BATCH_SIZE_IMG  = 8
BATCH_SIZE_Q    = 32
INDEX_BATCH_SIZE = 100
TOP_K           = 100
K_VALUES        = [1, 3, 5, 10, 25, 50, 100]
SPLIT           = "test"
RESULTS_DIR     = "results"
VIDORE_DATASETS = [
    "vidore/vidore_v3_hr",
    # "vidore/vidore_v3_finance_en",
    # "vidore/vidore_v3_industrial",
    # "vidore/vidore_v3_pharmaceuticals",
    # "vidore/vidore_v3_computer_science",
]
os.makedirs(RESULTS_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Embeddings wrapper (duck-typed — no base class needed)
# ══════════════════════════════════════════════════════════════════════════════

class ColModernVBertEmbeddings:
    """
    Wraps ColModernVBert for use with NextPlaidVectorStore.

    Shape contract (same as ColBERT):
        embed_images(images)  -> List[List[List[float]]]  # (n_patches × dim) per image
        embed_documents(texts)-> List[List[List[float]]]  # (n_tokens  × dim) per text
        embed_query(text)     -> List[List[float]]         # (n_tokens  × dim)
    """

    def __init__(self, model_id: str = IMAGE_MULTI_MODEL):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[ColModernVBert] Loading '{model_id}' on {self.device} …")
        self.processor = ColModernVBertProcessor.from_pretrained(model_id)
        self.model     = ColModernVBert.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.model.to(self.device).eval()
        logger.info("[ColModernVBert] Model loaded.")

    @torch.no_grad()
    def embed_images(
        self, images: List[Image.Image], batch_size: int = BATCH_SIZE_IMG
    ) -> List[List[List[float]]]:
        """Encode page images. Returns one (n_patches × dim) matrix per image."""
        all_embs = []
        for i in tqdm(range(0, len(images), batch_size),
                      desc="[Embed] images", leave=False):
            batch  = [img.convert("RGB") for img in images[i : i + batch_size]]
            inputs = {
                k: v.to(self.device)
                for k, v in self.processor.process_images(batch).items()
            }
            out = self.model(**inputs)  # (B, n_patches, dim)
            all_embs.extend(out[b].cpu().float().numpy().tolist() for b in range(out.shape[0]))
        return all_embs

    @torch.no_grad()
    def embed_documents(
        self, texts: List[str], batch_size: int = BATCH_SIZE_Q
    ) -> List[List[List[float]]]:
        """Encode text passages (used if corpus has markdown text instead of images)."""
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="[Embed] texts", leave=False):
            inputs = {
                k: v.to(self.device)
                for k, v in self.processor.process_texts(texts[i : i + batch_size]).items()
            }
            out = self.model(**inputs)
            all_embs.extend(out[b].cpu().float().numpy().tolist() for b in range(out.shape[0]))
        return all_embs

    @torch.no_grad()
    def embed_query(self, text: str) -> List[List[float]]:
        """Encode a single query string."""
        inputs = {
            k: v.to(self.device)
            for k, v in self.processor.process_texts([text]).items()
        }
        out = self.model(**inputs)  # (1, n_tokens, dim)
        return out[0].cpu().float().numpy().tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_vidore_data(
    dataset_name: str,
) -> Tuple[List[str], List[Image.Image], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load corpus images, queries, and qrels from a ViDoRe v3 HuggingFace dataset.

    Returns:
        corpus_ids    : list of str corpus IDs
        corpus_images : list of PIL images (aligned with corpus_ids)
        queries       : {query_id: query_text}   (English only)
        qrels         : {query_id: {corpus_id: relevance}}
    """
    logger.info(f"[Data] Loading corpus  : {dataset_name}")
    corpus_ds  = load_dataset(dataset_name, "corpus",  split=SPLIT)
    logger.info(f"[Data] Loading queries : {dataset_name}")
    queries_ds = load_dataset(dataset_name, "queries", split=SPLIT)
    logger.info(f"[Data] Loading qrels   : {dataset_name}")
    qrels_ds   = load_dataset(dataset_name, "qrels",   split=SPLIT)

    corpus_ids:    List[str]         = []
    corpus_images: List[Image.Image] = []
    for row in corpus_ds:
        corpus_ids.append(str(row["corpus_id"]))
        corpus_images.append(row["image"].convert("RGB"))

    queries: Dict[str, str] = {
        str(row["query_id"]): row["query"]
        for row in queries_ds
        if row.get("language", "english").lower() == "english"
    }

    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        if qid not in queries:
            continue
        qrels.setdefault(qid, {})[cid] = int(row["score"])

    logger.info(
        f"[Data]   Corpus: {len(corpus_ids)} pages | "
        f"Queries: {len(queries)} | Qrels pairs: {sum(len(v) for v in qrels.values())}"
    )
    return corpus_ids, corpus_images, queries, qrels


# ══════════════════════════════════════════════════════════════════════════════
# Index management
# ══════════════════════════════════════════════════════════════════════════════

def delete_index_if_exists(url: str, index_name: str):
    r = requests.get(f"{url}/indices/{index_name}", timeout=10)
    if r.status_code == 404:
        return
    r.raise_for_status()
    logger.info(f"[Index] Deleting '{index_name}' …")
    requests.delete(f"{url}/indices/{index_name}", timeout=30).raise_for_status()
    for _ in range(30):
        time.sleep(1)
        if requests.get(f"{url}/indices/{index_name}", timeout=10).status_code == 404:
            logger.info(f"[Index] '{index_name}' deleted.")
            return
    raise RuntimeError(f"Index '{index_name}' still present after deletion timeout.")


def wait_for_index_stable(url: str, index_name: str, expected_docs: int, stable_secs: int = 10):
    """Poll until num_documents == expected_docs and stays there for stable_secs."""
    prev         = -1
    stable_since = None
    logger.info(f"[Index] Waiting for {expected_docs} docs …")
    while True:
        try:
            r     = requests.get(f"{url}/indices/{index_name}", timeout=10)
            r.raise_for_status()
            count = r.json().get("num_documents", 0)
            print(f"\r[Index] {count}/{expected_docs} docs indexed …", end="", flush=True)
            if count >= expected_docs:
                if count == prev:
                    stable_since = stable_since or time.time()
                    if time.time() - stable_since >= stable_secs:
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


def build_index(
    store: NextPlaidVectorStore,
    corpus_ids: List[str],
    corpus_images: List[Image.Image],
    batch_size: int = 50,
):
    """Add all corpus images to the store in batches."""
    n = len(corpus_ids)
    logger.info(f"[Index] Indexing {n} images in batches of {batch_size} …")
    for start in tqdm(range(0, n, batch_size), desc="[Index] Batches"):
        end = min(start + batch_size, n)
        store.add_images(
            corpus_images[start:end],
            metadatas=[{"corpus_id": cid} for cid in corpus_ids[start:end]],
            ids=corpus_ids[start:end],
        )
    logger.info("[Index] All batches submitted.")


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def run_retrieval(
    store: NextPlaidVectorStore,
    queries: Dict[str, str],
    top_k: int = TOP_K,
) -> Dict[str, Dict[str, float]]:
    """
    Query the NextPlaid index for all queries.

    Returns pytrec_eval-compatible run dict: {query_id: {corpus_id: score}}

    Note: NextPlaid assigns internal integer document IDs, but we store
    the original corpus_id in metadata and recover it here.
    """
    run: Dict[str, Dict[str, float]] = {}
    for qid, qtext in tqdm(queries.items(), desc="[Eval] Retrieving"):
        results = store.similarity_search_with_score(qtext, k=top_k)
        run[qid] = {}
        for doc, score in results:
            # langchain_id was set to corpus_id during add_images
            corpus_id = doc.id or doc.metadata.get("corpus_id")
            if corpus_id:
                run[qid][str(corpus_id)] = float(score)
    return run


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

TREC_MEASURES = {
    *{f"ndcg_cut.{k}" for k in K_VALUES},
    *{f"map_cut.{k}"  for k in K_VALUES},
    *{f"recall.{k}"   for k in K_VALUES},
    *{f"P.{k}"        for k in K_VALUES},
}


def evaluate_run(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict:
    int_qrels = {
        qid: {did: int(s) for did, s in docs.items()}
        for qid, docs in qrels.items()
    }
    evaluator = pytrec_eval.RelevanceEvaluator(int_qrels, TREC_MEASURES)
    per_query = evaluator.evaluate(run)

    def avg(key: str) -> float:
        vals = [per_query[qid][key] for qid in per_query if key in per_query[qid]]
        return round(float(np.mean(vals)), 5) if vals else 0.0

    ndcg      = {f"NDCG@{k}":      avg(f"ndcg_cut_{k}") for k in K_VALUES}
    _map      = {f"MAP@{k}":       avg(f"map_cut_{k}")   for k in K_VALUES}
    recall    = {f"Recall@{k}":    avg(f"recall_{k}")    for k in K_VALUES}
    precision = {f"Precision@{k}": avg(f"P_{k}")         for k in K_VALUES}

    total_q  = len(per_query)
    hit_rate = {}
    for k in K_VALUES:
        hits = sum(
            1 for qid, scores in run.items()
            if set(sorted(scores, key=scores.get, reverse=True)[:k])
               & set(qrels.get(qid, {}).keys())
        )
        hit_rate[f"Hit@{k}"] = round(hits / total_q, 5) if total_q else 0.0

    return {"ndcg": ndcg, "map": _map, "recall": recall,
            "precision": precision, "hit_rate": hit_rate}


def print_metrics(label: str, metrics: Dict):
    print("\n" + "=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print("\n── Hit Rate ──────────────────────────────────────────")
    for k in K_VALUES:
        print(f"  Hit@{k:<4}: {metrics['hit_rate'][f'Hit@{k}']:.4f}")
    print("\n── NDCG ──────────────────────────────────────────────")
    for k, v in metrics["ndcg"].items():
        print(f"  {k}: {v:.4f}")
    print("\n── Recall ────────────────────────────────────────────")
    for k, v in metrics["recall"].items():
        print(f"  {k}: {v:.4f}")
    print("\n── Precision ─────────────────────────────────────────")
    for k, v in metrics["precision"].items():
        print(f"  {k}: {v:.4f}")
    print("\n── MAP ───────────────────────────────────────────────")
    for k, v in metrics["map"].items():
        print(f"  {k}: {v:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete and rebuild each dataset's index before evaluating."
    )
    args = parser.parse_args()

    logger.info(f"[Main] Loading ColModernVBert: {IMAGE_MULTI_MODEL}")
    embeddings = ColModernVBertEmbeddings(model_id=IMAGE_MULTI_MODEL)

    all_metrics: Dict[str, Dict] = {}

    for dataset_name in VIDORE_DATASETS:
        short_name = dataset_name.split("/")[-1]
        index_name = f"vidore_{short_name}"

        logger.info(f"\n{'#'*65}\n  DATASET: {short_name}\n{'#'*65}")

        # 1. Load data
        corpus_ids, corpus_images, queries, qrels = load_vidore_data(dataset_name)

        # 2. Optionally wipe and rebuild
        if args.rebuild:
            delete_index_if_exists(NEXT_PLAID_URL, index_name)

        # 3. Connect / create index
        store = NextPlaidVectorStore(
            url=NEXT_PLAID_URL,
            index_name=index_name,
            embedding=embeddings,
            create_index_if_not_exists=True,
        )

        # 4. Index corpus images if rebuilding
        if args.rebuild:
            build_index(store, corpus_ids, corpus_images, batch_size=INDEX_BATCH_SIZE)

        # 5. Wait for full indexing
        wait_for_index_stable(NEXT_PLAID_URL, index_name, expected_docs=len(corpus_ids))

        # 6. Spot-check
        qid0   = list(queries.keys())[0]
        logger.info(f"[Debug] Query '{qid0}': {queries[qid0]}")
        logger.info(f"[Debug] Relevant docs: {list(qrels.get(qid0, {}).keys())}")

        # 7. Retrieve
        run = run_retrieval(store, queries, top_k=TOP_K)

        top5 = list(run.get(qid0, {}).keys())[:5]
        logger.info(f"[Debug] Top-5 retrieved: {top5}")

        # 8. Evaluate
        metrics = evaluate_run(run, qrels)
        metrics["model"]   = IMAGE_MULTI_MODEL
        metrics["dataset"] = dataset_name
        all_metrics[short_name] = metrics

        print_metrics(f"ColModernVBert | {short_name}", metrics)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("  NDCG@10 SUMMARY — ColModernVBert via NextPlaid")
    print("=" * 90)

    short_names = [ds.split("/")[-1] for ds in VIDORE_DATASETS]
    header = f"{'Dataset':<40} {'NDCG@10':>10} {'Hit@10':>10} {'Recall@10':>10}"
    print(header)
    print("-" * 90)

    ndcg10_scores = []
    for sn in short_names:
        m      = all_metrics.get(sn, {})
        ndcg10 = m.get("ndcg",     {}).get("NDCG@10",  0.0)
        hit10  = m.get("hit_rate", {}).get("Hit@10",   0.0)
        rec10  = m.get("recall",   {}).get("Recall@10", 0.0)
        ndcg10_scores.append(ndcg10)
        print(f"  {sn:<38} {ndcg10:>10.4f} {hit10:>10.4f} {rec10:>10.4f}")

    avg_ndcg10 = float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0
    print("-" * 90)
    print(f"  {'AVERAGE':<38} {avg_ndcg10:>10.4f}")
    print("=" * 90)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "colmodernvbert_nextplaid_vidore_v3.json")
    with open(out_path, "w") as f:
        json.dump(
            {"model": IMAGE_MULTI_MODEL, "server": NEXT_PLAID_URL,
             "datasets": VIDORE_DATASETS, "metrics": all_metrics},
            f, indent=4,
        )
    logger.info(f"[Main] Results saved → {out_path}")


if __name__ == "__main__":
    main()