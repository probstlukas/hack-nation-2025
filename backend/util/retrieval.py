from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .pdf2text import chunk_documents


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


_DOC_INDEX: Dict[str, Dict[str, Any]] = {}


def _ensure_index(doc_id: str, chunk_size: int = 900, overlap: int = 100) -> None:
    if doc_id in _DOC_INDEX:
        return
    docs = chunk_documents(doc_id, chunk_size=chunk_size, overlap=overlap)
    chunks: List[str] = [d.page_content for d in docs]
    meta: List[Dict[str, Any]] = [d.metadata for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(chunks)
    _DOC_INDEX[doc_id] = {
        "vectorizer": vectorizer,
        "tfidf": tfidf,
        "chunks": chunks,
        "meta": meta,
    }


def retrieve(
    doc_id: str, question: str, top_k: int = 4
) -> List[Tuple[str, Dict[str, Any], float]]:
    _ensure_index(doc_id)
    store = _DOC_INDEX[doc_id]
    vectorizer: TfidfVectorizer = store["vectorizer"]
    tfidf = store["tfidf"]
    chunks: List[str] = store["chunks"]
    meta: List[Dict[str, Any]] = store["meta"]

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    return [(chunks[i], meta[i], float(sims[i])) for i in idxs]


def answer_question(doc_id: str, question: str, top_k: int = 4) -> Dict[str, Any]:
    """Simple extractive QA: returns top snippets as the answer with scores.

    This is a lightweight baseline to replace rule-based responses.
    """
    hits = retrieve(doc_id, question, top_k=top_k)
    if not hits:
        return {
            "answer": "No relevant content found in the document.",
            "sources": [],
            "confidence": 0.0,
        }

    # Concatenate top snippets as the answer body
    snippets = []
    sources = []
    scores = []
    for text, md, score in hits:
        page = md.get("page", md.get("source", ""))
        snippets.append(text.strip())
        sources.append(f"page {page}")
        scores.append(score)

    answer = "\n\n---\n\n".join(snippets)
    avg_score = sum(scores) / max(1, len(scores))
    # map cosine [0..1] to rough confidence [0.6..0.95]
    confidence = 0.6 + 0.35 * max(0.0, min(1.0, avg_score))

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 3),
    }
