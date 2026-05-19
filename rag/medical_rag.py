import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .corpus_loader import CorpusSource, default_corpus_sources, load_documents, parse_sources_manifest
from .chunking import normalize_whitespace


try:
    import faiss
    from sentence_transformers import SentenceTransformer

    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False


class MedicalRAG:
    """
    Medical retrieval over authoritative text sources (FAISS over SentenceTransformer embeddings).

    Notes:
    - Uses cosine similarity via IndexFlatIP + L2-normalized embeddings.
    - Persists index + corpus to disk; reloads if config matches.
    """

    INDEX_FILENAME = "medical_rag.faiss"
    CORPUS_FILENAME = "medical_rag_corpus.json"
    CONFIG_FILENAME = "medical_rag_config.json"

    def __init__(
        self,
        *,
        index_dir: str,
        corpus_manifest_path: Optional[str] = None,
        corpus_sources: Optional[List[Dict[str, Any]]] = None,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        force_rebuild: bool = False,
        chunk_size_words: int = 200,
        chunk_overlap_words: int = 40,
        min_chunk_words: int = 8,
        min_text_chars: int = 80,
    ):
        if not _RAG_AVAILABLE:
            raise RuntimeError("RAG requires: pip install faiss-cpu sentence-transformers")

        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.embed_model_name = embed_model_name
        self.top_k = top_k
        self.force_rebuild = force_rebuild

        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.min_chunk_words = min_chunk_words
        self.min_text_chars = min_text_chars

        self._encoder: Optional[SentenceTransformer] = None
        self._index = None
        self._docs: List[Dict[str, Any]] = []

        self.corpus_sources: List[CorpusSource] = self._resolve_sources(
            corpus_manifest_path=corpus_manifest_path,
            corpus_sources=corpus_sources,
        )

        if force_rebuild:
            self._build_index()
        else:
            self._load_or_build()

    @property
    def encoder(self) -> "SentenceTransformer":
        if self._encoder is None:
            print(f"[RAG] Loading embedding model: {self.embed_model_name}")
            self._encoder = SentenceTransformer(self.embed_model_name)
        return self._encoder

    def _resolve_sources(
        self,
        *,
        corpus_manifest_path: Optional[str],
        corpus_sources: Optional[List[Dict[str, Any]]],
    ) -> List[CorpusSource]:
        if corpus_manifest_path:
            sources = parse_sources_manifest(corpus_manifest_path)
            if not sources:
                raise ValueError(f"Empty manifest: {corpus_manifest_path}")
            return sources

        if corpus_sources:
            parsed: List[CorpusSource] = []
            for s in corpus_sources:
                if not isinstance(s, dict):
                    continue
                parsed.append(
                    CorpusSource(
                        path=s["path"],
                        format=s.get("format", "jsonl"),
                        source_id=s.get("source_id", ""),
                        optional=bool(s.get("optional", False)),
                        text_fields=s.get("text_fields"),
                    )
                )
            if parsed:
                return parsed

        # Default authoritative sources
        return default_corpus_sources()

    def _fingerprint(self) -> str:
        payload = {
            "embed_model_name": self.embed_model_name,
            "chunk_size_words": self.chunk_size_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "min_chunk_words": self.min_chunk_words,
            "min_text_chars": self.min_text_chars,
            "corpus_sources": [
                {
                    "path": s.path,
                    "format": s.format,
                    "source_id": s.source_id,
                    "optional": s.optional,
                    "text_fields": s.text_fields,
                }
                for s in self.corpus_sources
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _config_path(self) -> str:
        return os.path.join(self.index_dir, self.CONFIG_FILENAME)

    def _index_path(self) -> str:
        return os.path.join(self.index_dir, self.INDEX_FILENAME)

    def _corpus_path(self) -> str:
        return os.path.join(self.index_dir, self.CORPUS_FILENAME)

    def _load_or_build(self) -> None:
        cfg_path = self._config_path()
        idx_path = self._index_path()
        corpus_path = self._corpus_path()

        if os.path.exists(cfg_path) and os.path.exists(idx_path) and os.path.exists(corpus_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("fingerprint") == self._fingerprint():
                    print(f"[RAG] Loading existing index from: {self.index_dir}")
                    import faiss  # re-import for mypy

                    self._index = faiss.read_index(idx_path)
                    with open(corpus_path, "r", encoding="utf-8") as f:
                        self._docs = json.load(f)
                    return
                else:
                    print("[RAG] Index config mismatch; rebuilding.")
            except Exception:
                print("[RAG] Failed to load existing index config; rebuilding.")

        self._build_index()

    def _build_index(self) -> None:
        fingerprint = self._fingerprint()

        print("[RAG] Building index from authoritative corpus (first-time setup).")
        docs = load_documents(
            self.corpus_sources,
            chunk_size_words=self.chunk_size_words,
            chunk_overlap_words=self.chunk_overlap_words,
            min_chunk_words=self.min_chunk_words,
            min_text_chars=self.min_text_chars,
        )

        # Store docs as plain list to keep it reloadable.
        self._docs = docs
        if not self._docs:
            # Determine embedding dim from encoder, to keep FAISS happy.
            dim = int(self.encoder.get_sentence_embedding_dimension())
            self._index = faiss.IndexFlatIP(dim)
            self._save_empty(fingerprint)
            return

        texts = [normalize_whitespace(d["text"]) for d in self._docs]
        print(f"[RAG] Encoding {len(texts)} chunks ...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine via inner product
        ).astype("float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        faiss.write_index(self._index, self._index_path())
        with open(self._corpus_path(), "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False)
        with open(self._config_path(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "embed_model_name": self.embed_model_name,
                    "corpus_sources": [
                        {"path": s.path, "format": s.format, "source_id": s.source_id}
                        for s in self.corpus_sources
                    ],
                    "chunk_size_words": self.chunk_size_words,
                    "chunk_overlap_words": self.chunk_overlap_words,
                    "min_chunk_words": self.min_chunk_words,
                    "min_text_chars": self.min_text_chars,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[RAG] Index saved: chunks={len(self._docs)}, dim={dim}")

    def _save_empty(self, fingerprint: str) -> None:
        # Create empty files for consistent behavior.
        faiss.write_index(self._index, self._index_path())
        with open(self._corpus_path(), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
        with open(self._config_path(), "w", encoding="utf-8") as f:
            json.dump(
                {"fingerprint": fingerprint},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        k = top_k or self.top_k
        query = normalize_whitespace(query)
        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            i = int(idx)
            if 0 <= i < len(self._docs):
                doc = self._docs[i]
                results.append(
                    {
                        "text": doc["text"],
                        "score": float(score),
                        "metadata": doc.get("metadata", {}),
                    }
                )
        return results

