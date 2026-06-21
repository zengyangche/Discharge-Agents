"""
Few-shot example retriever based on vector search.

Given the current case EHR context, retrieve the top-K most semantically similar
examples from the example pool for dynamic few-shot prompt construction
(replacing fixed/random examples).

Retrieval strategy:
  1. Read BHC/DI labeled examples from examples_dir (outputs/test_labels)
  2. Prefer loading corresponding raw EHR text from ehr_dir (outputs/test_inputs)
     as the embedding anchor (fall back to BHC text if not found)
  3. Precompute and cache vectors via OpenAI-compatible Embedding API
     (text-embedding-3-small)
  4. At inference, embed the current case context, rank by cosine similarity,
     return top-K

Usage:
    from utils.few_shot_retriever import FewShotRetriever

    retriever = FewShotRetriever()
    retriever.build_index()            # First call builds and caches the index

    examples = retriever.retrieve(query_text=bhc_context, k=5)
    # examples: [{"bhc": str, "di": str, "file": str, "score": float}, ...]
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure the project root is in sys.path regardless of where this module is called from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


class FewShotRetriever:
    """
    Few-shot example retriever based on vector similarity.

    Embedding text priority (per example):
      1. Corresponding raw EHR text (outputs/test_inputs/case_*.txt) — closest to
         inference input semantically
      2. Example BHC text — fallback

    Inference query text:
      - BHC retrieval: bhc_specific_context (or shared_context + bhc_specific_context)
      - DI retrieval: di_specific_context (or shared_context + di_specific_context)
    """

    MAX_EMBED_CHARS = 3000  # Truncation length to avoid exceeding API token limit

    def __init__(
        self,
        examples_dir: str = "outputs/test_labels",
        ehr_dir: str = "outputs/test_inputs",
        embedding_model: str = "text-embedding-3-small",
        cache_dir: str = "outputs/few_shot_index",
        exclude_stems: Optional[List[str]] = None,
    ):
        # ⚠️  Data leakage warning:
        # The default examples_dir points to outputs/test_labels (test set annotations).
        # If the retrieval corpus overlaps with the current evaluation set, retrieved examples
        # may contain gold labels for the current test samples, inflating evaluation metrics.
        # For production/evaluation, change examples_dir to the training annotations directory,
        # or use exclude_stems to exclude all samples in the current evaluation batch.
        self.examples_dir = Path(examples_dir)
        self.ehr_dir = Path(ehr_dir)
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        # Set of filename stems (without extension); these samples are skipped during retrieval
        self.exclude_stems: set = set(exclude_stems) if exclude_stems else set()

        self._examples: List[Dict] = []           # {"bhc", "di", "file"}
        self._embeddings: Optional[np.ndarray] = None  # shape [N, D]
        self._index_built = False

    # ── Public API ───────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Build (or load from cache) the embedding index.

        Args:
            force_rebuild: If True, ignore cache and force recompute embeddings.

        Returns:
            Number of examples in the index.
        """
        cache_meta = self.cache_dir / f"{self.embedding_model}_meta.json"
        cache_emb = self.cache_dir / f"{self.embedding_model}_embeddings.npy"

        if not force_rebuild and cache_meta.exists() and cache_emb.exists():
            try:
                all_examples = json.load(open(cache_meta, "r", encoding="utf-8"))
                all_embeddings = np.load(str(cache_emb))
                # Apply exclude_stems filtering (required even on cache hit)
                if self.exclude_stems:
                    keep = [
                        i for i, ex in enumerate(all_examples)
                        if Path(ex.get("file", "")).stem not in self.exclude_stems
                    ]
                    self._examples = [all_examples[i] for i in keep]
                    self._embeddings = all_embeddings[keep]
                else:
                    self._examples = all_examples
                    self._embeddings = all_embeddings
                self._index_built = True
                excluded = len(all_examples) - len(self._examples)
                logger.info(
                    f"[FewShotRetriever] Loaded index from cache: {len(self._examples)} examples"
                    f" (excluded {excluded}, model: {self.embedding_model}, cache: {self.cache_dir})"
                )
                return len(self._examples)
            except Exception as e:
                logger.warning(f"[FewShotRetriever] Cache load failed, rebuilding: {e}")

        # Load examples from files
        raw_examples = self._load_examples_from_dir()
        if not raw_examples:
            logger.warning("[FewShotRetriever] No valid examples found, index is empty")
            self._index_built = True
            return 0

        # Compute embeddings
        embed_texts = [ex.pop("embed_text") for ex in raw_examples]
        self._examples = raw_examples

        logger.info(
            f"[FewShotRetriever] Computing embeddings for {len(embed_texts)} examples"
            f" (model: {self.embedding_model})…"
        )
        self._embeddings = self._embed_texts(embed_texts)

        # Persist cache to disk
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump(self._examples, f, ensure_ascii=False, indent=2)
        np.save(str(cache_emb), self._embeddings)
        logger.info(f"[FewShotRetriever] Index cached at {self.cache_dir}")

        self._index_built = True
        return len(self._examples)

    def retrieve(self, query: str, k: int) -> List[Dict]:
        """
        Given query text, return the top-K most similar examples.

        Args:
            query: Query text (EHR context slice for the current case,
                   e.g. bhc_specific_context)
            k:     Number of examples to return

        Returns:
            list of {"bhc": str, "di": str, "file": str, "score": float},
            sorted by similarity descending. Returns all if fewer than k examples.
        """
        if not self._index_built:
            self.build_index()

        if not self._examples or self._embeddings is None or len(self._examples) == 0:
            logger.warning("[FewShotRetriever] Index is empty, returning empty list")
            return []

        k = min(k, len(self._examples))
        if k == 0:
            return []

        # Truncate query and compute embedding
        query_vec = self._embed_texts([query[: self.MAX_EMBED_CHARS]])[0]  # shape [D]

        # Cosine similarity: L2-normalize library vectors then compute dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9
        normed = self._embeddings / norms
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        similarities = normed @ q_norm  # shape [N]

        top_idx = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_idx:
            ex = self._examples[idx]
            results.append(
                {
                    "bhc": ex["bhc"],
                    "di": ex["di"],
                    "file": ex.get("file", ""),
                    "score": float(similarities[idx]),
                }
            )

        logger.debug(
            f"[FewShotRetriever] Retrieved top-{k}, score range: "
            f"{results[0]['score']:.3f} ~ {results[-1]['score']:.3f}"
        )
        return results

    def __len__(self) -> int:
        return len(self._examples)

    # ── Internal methods ───────────────────────────────────────────────────────────────

    def _load_examples_from_dir(self) -> List[Dict]:
        """
        Read all case_*.txt from examples_dir and extract BHC/DI text;
        also try loading corresponding raw EHR text from ehr_dir as the
        embedding anchor.

        Each record in the returned list includes an "embed_text" field
        (removed after the index is built).
        """
        examples = []
        if not self.examples_dir.exists():
            logger.warning(
                f"[FewShotRetriever] Examples directory not found: {self.examples_dir}"
            )
            return examples

        files = sorted(self.examples_dir.glob("case_*.txt"))
        for fp in files:
            if fp.stem in self.exclude_stems:
                logger.debug(f"[FewShotRetriever] Skipping excluded example: {fp.name}")
                continue
            try:
                content = fp.read_text(encoding="utf-8")

                bhc_m = re.search(
                    r"Brief Hospital Course.*?={80}\s*(.*?)\s*={80}",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                di_m = re.search(
                    r"Discharge Instructions.*?={80}\s*(.*?)(?:\s*={80}|\Z)",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if not (bhc_m and di_m):
                    continue

                bhc = bhc_m.group(1).strip()
                di = di_m.group(1).strip()

                # Filter out examples that are too short
                if len(bhc.split()) < 30 or len(di.split()) < 20:
                    continue

                # Prefer EHR raw text as the embedding anchor
                ehr_text = self._load_ehr_text(fp.stem)
                embed_text = (ehr_text or bhc)[: self.MAX_EMBED_CHARS]

                examples.append(
                    {
                        "bhc": bhc,
                        "di": di,
                        "file": fp.name,
                        "embed_text": embed_text,
                    }
                )
            except Exception as e:
                logger.debug(f"[FewShotRetriever] Skipping {fp.name}: {e}")

        logger.info(
            f"[FewShotRetriever] Loaded {len(examples)} valid examples"
            f" (from {self.examples_dir})"
        )
        return examples

    def _load_ehr_text(self, stem: str) -> Optional[str]:
        """
        Given a test_labels filename stem (e.g. case_12345_67890),
        find the matching .txt in ehr_dir and read it.
        """
        if not self.ehr_dir.exists():
            return None
        matches = list(self.ehr_dir.glob(f"{stem}.txt"))
        if not matches:
            return None
        try:
            return matches[0].read_text(encoding="utf-8")
        except Exception:
            return None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Batch-call OpenAI-compatible Embedding API, up to 100 texts per batch.
        On batch API failure, pad with zero vectors (does not abort the pipeline).
        """
        from utils.config import OPENAI_API_KEY, OPENAI_API_BASE_URL
        import openai

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE_URL,
        )

        all_embeddings: List[List[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = [t[: self.MAX_EMBED_CHARS] for t in texts[i: i + batch_size]]
            try:
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                batch_emb = [item.embedding for item in response.data]
                all_embeddings.extend(batch_emb)
                logger.debug(
                    f"[FewShotRetriever] Embedding progress: {i + len(batch)}/{len(texts)}"
                )
            except Exception as e:
                logger.error(f"[FewShotRetriever] Embedding API call failed (batch {i}): {e}")
                # Placeholder dimension: reuse existing vector dimension, default to 1536
                dim = len(all_embeddings[0]) if all_embeddings else 1536
                all_embeddings.extend([[0.0] * dim] * len(batch))

        return np.array(all_embeddings, dtype=np.float32)


# ── Command-line quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build/test FewShotRetriever index")
    parser.add_argument(
        "--build",
        action="store_true",
        help="(Re)build the embedding index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute embeddings (ignore cache)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Test query text (run one retrieval and print results if provided)",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of examples to return (default: 5)")
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="outputs/test_labels",
    )
    parser.add_argument(
        "--ehr-dir",
        type=str,
        default="outputs/test_inputs",
    )
    args = parser.parse_args()

    retriever = FewShotRetriever(
        examples_dir=args.examples_dir,
        ehr_dir=args.ehr_dir,
    )

    if args.build or args.force:
        n = retriever.build_index(force_rebuild=args.force)
        print(f"Index built, {n} examples total")
    else:
        n = retriever.build_index()
        print(f"Index loaded, {n} examples total")

    if args.query:
        results = retriever.retrieve(args.query, k=args.k)
        print(f"\nTop-{len(results)} retrieval results:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {r['score']:.4f}  File: {r['file']}")
            print(f"    BHC (first 150 chars): {r['bhc'][:150]}…")
