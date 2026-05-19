import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .chunking import chunk_text_words, normalize_whitespace


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CorpusSource:
    path: str
    format: str
    source_id: str = ""
    optional: bool = False

    # Generic text fields (used by json/jsonl formats)
    text_fields: Optional[List[str]] = None


def _strip_xml_tags(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    exts = tuple(exts)
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(exts):
                yield os.path.join(dirpath, fname)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_text_from_fields(obj: Dict[str, Any], text_fields: List[str]) -> str:
    parts: List[str] = []
    for field in text_fields:
        val = obj.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n".join(parts).strip()


def _safe_merge_metadata(base: Dict[str, Any], extra: Any) -> Dict[str, Any]:
    """
    Merge per-record metadata into the document metadata if present.

    Expected format in JSON/JSONL records:
      {"metadata": {...}, ...}
    """
    if not isinstance(extra, dict):
        return base
    out = dict(base)
    # Avoid letting records override required keys.
    for k, v in extra.items():
        if k in ("source", "path"):
            continue
        out[k] = v
    return out


def parse_sources_manifest(manifest_path: str) -> List[CorpusSource]:
    """Load corpus sources from a JSON manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sources = manifest.get("sources", [])
    parsed: List[CorpusSource] = []
    for s in sources:
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
    return parsed


def default_corpus_sources() -> List[CorpusSource]:
    """Default to peer-reviewed authoritative text (PMC full texts)."""
    return [
        CorpusSource(
            path="/home/wshenah/LLaVA-Med/data/pmc_articles",
            format="pmc_xml_dir",
            source_id="pmc",
            optional=False,
        )
    ]


def load_documents(
    sources: List[CorpusSource],
    *,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 40,
    min_chunk_words: int = 8,
    min_text_chars: int = 80,
) -> List[Dict[str, Any]]:
    """Return a list of {'text': str, 'metadata': {...}} ready for embedding."""
    docs: List[Dict[str, Any]] = []

    for src in sources:
        if not os.path.exists(src.path):
            if src.optional:
                continue
            raise FileNotFoundError(f"Corpus source not found: {src.path}")

        if src.format in ("pmc_xml_dir", "pmc_xml"):
            for fpath in _iter_files(src.path, (".xml", ".nxml", ".txt")):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    text = _strip_xml_tags(raw)
                    if len(text) < min_text_chars:
                        continue
                    for chunk in chunk_text_words(
                        text,
                        chunk_size_words=chunk_size_words,
                        chunk_overlap_words=chunk_overlap_words,
                        min_chunk_words=min_chunk_words,
                    ):
                        docs.append(
                            {
                                "text": chunk,
                                "metadata": {"source": src.source_id, "path": fpath},
                            }
                        )
                except Exception:
                    continue

        elif src.format in ("text_dir", "txt_dir"):
            for fpath in _iter_files(src.path, (".txt",)):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    text = normalize_whitespace(raw)
                    if len(text) < min_text_chars:
                        continue
                    for chunk in chunk_text_words(
                        text,
                        chunk_size_words=chunk_size_words,
                        chunk_overlap_words=chunk_overlap_words,
                        min_chunk_words=min_chunk_words,
                    ):
                        docs.append(
                            {
                                "text": chunk,
                                "metadata": {"source": src.source_id, "path": fpath},
                            }
                        )
                except Exception:
                    continue

        elif src.format in ("jsonl", "jsonl_path"):
            text_fields = src.text_fields or ["title", "abstract", "body", "text"]
            for obj in _load_jsonl(src.path):
                if not isinstance(obj, dict):
                    continue
                text = _get_text_from_fields(obj, text_fields)
                text = normalize_whitespace(text)
                if len(text) < min_text_chars:
                    continue
                for chunk in chunk_text_words(
                    text,
                    chunk_size_words=chunk_size_words,
                    chunk_overlap_words=chunk_overlap_words,
                    min_chunk_words=min_chunk_words,
                ):
                    md = _safe_merge_metadata(
                        {"source": src.source_id, "path": src.path},
                        obj.get("metadata"),
                    )
                    docs.append(
                        {
                            "text": chunk,
                            "metadata": md,
                        }
                    )

        elif src.format in ("json", "json_path"):
            text_fields = src.text_fields or ["title", "abstract", "body", "text"]
            data = _load_json(src.path)
            items = data if isinstance(data, list) else []
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                text = _get_text_from_fields(obj, text_fields)
                text = normalize_whitespace(text)
                if len(text) < min_text_chars:
                    continue
                for chunk in chunk_text_words(
                    text,
                    chunk_size_words=chunk_size_words,
                    chunk_overlap_words=chunk_overlap_words,
                    min_chunk_words=min_chunk_words,
                ):
                    md = _safe_merge_metadata(
                        {"source": src.source_id, "path": src.path},
                        obj.get("metadata"),
                    )
                    docs.append(
                        {
                            "text": chunk,
                            "metadata": md,
                        }
                    )

        elif src.format in ("llava_med_instruct_json",):
            # Compatibility with your original LLaVA-Med instruct JSON.
            # Extract assistant/gpt values + caption/abstract-like fields.
            if not src.path.endswith(".json"):
                continue
            data = _load_json(src.path)
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                for conv in item.get("conversations", []):
                    if not isinstance(conv, dict):
                        continue
                    if conv.get("from") == "gpt":
                        text = str(conv.get("value", "")).strip()
                        if len(text) >= min_text_chars:
                            for chunk in chunk_text_words(
                                text,
                                chunk_size_words=chunk_size_words,
                                chunk_overlap_words=chunk_overlap_words,
                                min_chunk_words=min_chunk_words,
                            ):
                                docs.append(
                                    {
                                        "text": chunk,
                                        "metadata": {"source": src.source_id, "path": src.path},
                                    }
                                )
                for field in ("caption", "fig_caption", "title", "abstract"):
                    val = item.get(field)
                    if isinstance(val, str) and len(val.strip()) >= min_text_chars:
                        for chunk in chunk_text_words(
                            val,
                            chunk_size_words=chunk_size_words,
                            chunk_overlap_words=chunk_overlap_words,
                            min_chunk_words=min_chunk_words,
                        ):
                            docs.append(
                                {
                                    "text": chunk,
                                    "metadata": {"source": src.source_id, "path": src.path},
                                }
                            )

        else:
            raise ValueError(f"Unsupported corpus source format: {src.format}")

    return docs

