import os
import re
from hashlib import md5
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import faiss
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

_CONFIGURED_API_KEY: str | None = None
_BROKEN_LOCAL_PROXY_TARGETS = {"http://127.0.0.1:9", "https://127.0.0.1:9", "http://localhost:9", "https://localhost:9"}


TXT_SOURCE_PATH = "data/banking_docs.txt"
PDF_SOURCE_PATH = "data/banking_policy.pdf"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.5-flash"


@dataclass
class SourceDocument:
    source_name: str
    source_type: str
    text: str
    page: int | None = None


@dataclass
class ChunkRecord:
    text: str
    source_name: str
    source_type: str
    page: int | None
    chunk_id: int


def _load_documents() -> List[SourceDocument]:
    documents: List[SourceDocument] = []

    if os.path.exists(TXT_SOURCE_PATH):
        with open(TXT_SOURCE_PATH, "r", encoding="utf-8") as file:
            documents.append(
                SourceDocument(
                    source_name="Banking Knowledge Base",
                    source_type="TXT",
                    text=file.read().strip(),
                )
            )

    if os.path.exists(PDF_SOURCE_PATH):
        reader = PdfReader(PDF_SOURCE_PATH)
        for page_index, page in enumerate(reader.pages, start=1):
            extracted_text = (page.extract_text() or "").strip()
            if extracted_text:
                documents.append(
                    SourceDocument(
                        source_name="Banking Policy Manual",
                        source_type="PDF",
                        text=extracted_text,
                        page=page_index,
                    )
                )

    return documents


def _chunk_documents(
    documents: List[SourceDocument], chunk_size: int = 700, overlap: int = 120
) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    chunk_id = 0

    for document in documents:
        text = " ".join(document.text.split())
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    ChunkRecord(
                        text=chunk_text,
                        source_name=document.source_name,
                        source_type=document.source_type,
                        page=document.page,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1

            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

    return chunks


@lru_cache(maxsize=1)
def _load_generation_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(GENERATION_MODEL_NAME)


def _clear_invalid_proxy_env() -> None:
    proxy_keys = [
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    for key in proxy_keys:
        value = (os.environ.get(key) or "").strip().lower()
        if value in _BROKEN_LOCAL_PROXY_TARGETS:
            os.environ.pop(key, None)


def _ensure_gemini_configured() -> str:
    global _CONFIGURED_API_KEY

    load_dotenv(override=True)
    _clear_invalid_proxy_env()
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your environment or .env file.")

    if api_key != _CONFIGURED_API_KEY:
        genai.configure(api_key=api_key)
        _CONFIGURED_API_KEY = api_key
        _load_generation_model.cache_clear()
        if "_build_vector_store" in globals():
            _build_vector_store.cache_clear()

    return api_key


def _fallback_embed_text(text: str, dimensions: int = 256) -> np.ndarray:
    vector = np.zeros(dimensions, dtype="float32")
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        token_hash = int(md5(token.encode("utf-8")).hexdigest(), 16)
        vector[token_hash % dimensions] += 1.0

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _embed_text(text: str, task_type: str) -> np.ndarray:
    try:
        _ensure_gemini_configured()
        response = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type=task_type,
            request_options={"timeout": 20},
        )
        embedding = np.array(response["embedding"], dtype="float32")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    except Exception:
        return _fallback_embed_text(text)


@lru_cache(maxsize=1)
def _build_vector_store() -> Dict[str, object]:
    documents = _load_documents()
    chunks = _chunk_documents(documents)
    if not chunks:
        raise ValueError("No banking documents were found in the data directory.")

    embeddings = np.vstack(
        [_embed_text(chunk.text, task_type="retrieval_document") for chunk in chunks]
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "documents": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
    }


def get_knowledge_base_stats() -> Dict[str, int]:
    store = _build_vector_store()
    pdf_pages = sum(1 for doc in store["documents"] if doc.source_type == "PDF")
    return {
        "documents": len(store["documents"]),
        "chunks": len(store["chunks"]),
        "pdf_pages": pdf_pages,
    }


def retrieve_context(query: str, top_k: int = 4) -> List[Dict[str, object]]:
    store = _build_vector_store()
    query_embedding = _embed_text(query, task_type="retrieval_query").reshape(1, -1)

    distances, indices = store["index"].search(query_embedding, min(top_k, len(store["chunks"])))
    results: List[Dict[str, object]] = []

    for score, chunk_index in zip(distances[0], indices[0]):
        if chunk_index < 0:
            continue
        chunk = store["chunks"][chunk_index]
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_name": chunk.source_name,
                "source_type": chunk.source_type,
                "page": chunk.page,
                "score": float(score),
            }
        )

    return results


def _format_chat_history(chat_history: List[Dict[str, str]] | None) -> str:
    if not chat_history:
        return "No prior conversation."
    return "\n".join(
        f"{item['role'].title()}: {item['content']}" for item in chat_history[-6:]
    )


def _confidence_label(score: float) -> str:
    if score >= 0.72:
        return "High"
    if score >= 0.50:
        return "Moderate"
    return "Low"


def ask_question(query: str, chat_history: List[Dict[str, str]] | None = None) -> Dict[str, object]:
    _ensure_gemini_configured()

    retrieved_chunks = retrieve_context(query)
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant banking knowledge in the current knowledge base.",
            "sources": [],
            "confidence_label": "Low",
            "confidence_score": 0.0,
        }

    context = "\n\n".join(
        [
            (
                f"[Source: {item['source_name']} | Type: {item['source_type']}"
                + (f" | Page: {item['page']}]\n{item['text']}" if item["page"] else f"]\n{item['text']}")
            )
            for item in retrieved_chunks
        ]
    )

    prompt = f"""
You are an enterprise-grade AI banking analyst helping customers and internal teams.

Rules:
- Answer only from the supplied context.
- If the context is insufficient, say clearly that the knowledge base does not contain enough information.
- Be precise, professional, and practical.
- Structure the answer in concise markdown with short paragraphs or bullets when useful.
- Mention compliance or risk implications only when supported by the context.

Conversation history:
{_format_chat_history(chat_history)}

Retrieved context:
{context}

User question:
{query}

Response:
"""

    response = _load_generation_model().generate_content(
        prompt,
        request_options={"timeout": 30},
    )
    top_score = max(item["score"] for item in retrieved_chunks)

    return {
        "answer": getattr(response, "text", "").strip()
        or "The model did not return a valid answer. Please try again.",
        "sources": retrieved_chunks,
        "confidence_label": _confidence_label(top_score),
        "confidence_score": round(top_score, 3),
    }
