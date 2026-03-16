import time
import faiss
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Any


class DOMRetriever:
    """Helper for embedding and searching a list of DOM element descriptions.

    Uses FAISS with normalized embeddings and cosine similarity for fast
    semantic search. sentence-transformers embeddings are normalized by
    default, and we use IndexFlatIP (inner product) which on normalized
    vectors equals cosine similarity.

    The agent can call ``BrowserManager.retrieve_relevant_elements`` which
    will delegate to an instance of this class.
    """

    def __init__(self, model_name: str = "all-MiniLm-L6-v2"):
        # sentence-transformers downloads models automatically the first time.
        # we keep the model around so that repeated calls are fast.
        self.model_name = model_name
        self.model = None
        self.index: Any = None
        self._elements: List[str] = []

    def _ensure_model_loaded(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            logging.info(f"Loading SentenceTransformer model: {self.model_name}")
            start = time.time()
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Model loaded in {time.time() - start:.2f} seconds")

    def build_index(self, elements: List[str]) -> None:
        """Encode and index the provided element strings.

        ``elements`` is usually the output of ``BrowserManager._extract_interactive_elements``
          which already returns human-readable strings.
        
        Filters out empty or whitespace-only strings to avoid indexing noise.
        Embeddings are normalized during encoding to ensure proper cosine
        similarity scoring with IndexFlatIP.
        """
        # Filter out empty or whitespace-only elements
        logging.info(f"Building DOM index with {len(elements)} elements (before filtering)")
        self._elements = [e for e in elements if e and e.strip()]
        logging.info(f"{len(self._elements)} elements remain after filtering empty/whitespace entries")
        if not self._elements:
            self.index = None
            return

        self._ensure_model_loaded()
        embeddings = self.model.encode(
            self._elements,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        
        dim = embeddings.shape[1]
        logging.info(f"Created FAISS index with {dim} dimensions")
        # IndexFlatIP = inner product; on L2-normalized vectors, this is cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def query(self, text: str, k: int = 30) -> List[str]:
        """Return up to ``k`` elements semantically similar to ``text``.

        Uses cosine similarity (inner product on normalized embeddings).
        If the index has not been built or is empty, returns an empty list.
        """
        if self.index is None:
            return []
        
        self._ensure_model_loaded()
        q_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        logging.info(f"Query embedding shape: {q_emb.shape}")
        
        distances, ids = self.index.search(q_emb, k)
        results = []
        for idx in ids[0]:
            if 0 <= idx < len(self._elements):
                results.append(self._elements[idx])
        return results
