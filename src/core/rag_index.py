from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import settings
from src.models.llm import embed_texts


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    path: str
    title: str
    heading: str
    content: str
    start: int
    end: int
    embedding: tuple[float, ...] | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{1}|[a-zA-Z0-9_]+", text.lower())


def _keyword_score(query: str, chunk: KnowledgeChunk) -> float:
    query_terms = _tokenize(query)
    if not query_terms:
        return 0.0
    haystack = f"{chunk.title}\n{chunk.heading}\n{chunk.content}".lower()
    score = 0.0
    for term in query_terms:
        hits = haystack.count(term)
        if hits:
            score += 2.0 if term in chunk.title.lower() or term in chunk.heading.lower() else 1.0
            score += min(hits, 8) * 0.25
    return score


class MarkdownKnowledgeIndex:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._signature: str | None = None
        self._chunks: list[KnowledgeChunk] = []
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._faiss_chunk_indices: list[int] = []

    async def semantic_search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        await self._ensure_loaded(with_embeddings=True)
        if self._faiss_index is None or not self._faiss_chunk_indices:
            return await self.keyword_search(query, limit=limit)
        query_embedding = (await embed_texts([query]))[0]
        query_vector = self._normalize_vectors(np.asarray([query_embedding], dtype="float32"))
        scores, indices = self._faiss_index.search(query_vector, min(limit, len(self._faiss_chunk_indices)))
        results: list[dict[str, Any]] = []
        for score, faiss_index in zip(scores[0], indices[0]):
            if faiss_index < 0:
                continue
            chunk = self._chunks[self._faiss_chunk_indices[int(faiss_index)]]
            results.append(self._format_result(chunk, score=float(score), keyword_score=_keyword_score(query, chunk)))
        return {
            "mode": "semantic",
            "engine": "faiss",
            "query": query,
            "results": results,
        }

    async def keyword_search(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        await self._ensure_loaded(with_embeddings=False)
        ranked = sorted(
            ((_keyword_score(query, chunk), chunk) for chunk in self._chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        results = [
            self._format_result(chunk, score=score)
            for score, chunk in ranked[:limit]
            if score > 0
        ]
        if not results:
            results = [self._format_result(chunk, score=0.0) for chunk in self._chunks[:limit]]
        return {"mode": "keyword", "query": query, "results": results}

    async def _ensure_loaded(self, *, with_embeddings: bool) -> None:
        signature = self._docs_signature()
        if self._is_ready(signature, with_embeddings):
            return
        async with self._lock:
            signature = self._docs_signature()
            if self._is_ready(signature, with_embeddings):
                return
            chunks = self._load_chunks()
            faiss_index: faiss.IndexFlatIP | None = None
            faiss_chunk_indices: list[int] = []
            if with_embeddings and chunks:
                try:
                    embeddings: list[list[float]] = []
                    batch_size = 32
                    for start in range(0, len(chunks), batch_size):
                        batch = chunks[start : start + batch_size]
                        embeddings.extend(await embed_texts([self._embedding_text(chunk) for chunk in batch]))
                    chunks = [
                        KnowledgeChunk(**{**chunk.__dict__, "embedding": tuple(embedding)})
                        for chunk, embedding in zip(chunks, embeddings)
                    ]
                    faiss_index, faiss_chunk_indices = self._build_faiss_index(chunks)
                except Exception:
                    # Keep keyword fallback available even when embedding service fails.
                    pass
            self._chunks = chunks
            self._faiss_index = faiss_index
            self._faiss_chunk_indices = faiss_chunk_indices
            self._signature = signature

    def _is_ready(self, signature: str, with_embeddings: bool) -> bool:
        if self._signature != signature or not self._chunks:
            return False
        if not with_embeddings:
            return True
        return self._faiss_index is not None and bool(self._faiss_chunk_indices)

    def _build_faiss_index(self, chunks: list[KnowledgeChunk]) -> tuple[faiss.IndexFlatIP | None, list[int]]:
        vectors: list[tuple[float, ...]] = []
        chunk_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            if chunk.embedding:
                vectors.append(chunk.embedding)
                chunk_indices.append(index)
        if not vectors:
            return None, []
        matrix = np.asarray(vectors, dtype="float32")
        matrix = self._normalize_vectors(matrix)
        faiss_index = faiss.IndexFlatIP(matrix.shape[1])
        faiss_index.add(matrix)
        return faiss_index, chunk_indices

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _docs_signature(self) -> str:
        root = settings.knowledge_doc_dir
        if not root.exists():
            return "missing"
        parts: list[str] = []
        for path in sorted(root.rglob("*.md")):
            stat = path.stat()
            parts.append(f"{path.relative_to(root)}:{stat.st_mtime_ns}:{stat.st_size}")
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def _load_chunks(self) -> list[KnowledgeChunk]:
        root = settings.knowledge_doc_dir
        if not root.exists():
            return []
        chunks: list[KnowledgeChunk] = []
        for path in sorted(root.rglob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(self._chunk_markdown(path, text))
        return chunks

    def _chunk_markdown(self, path: Path, text: str) -> list[KnowledgeChunk]:
        relative_path = str(path.relative_to(settings.knowledge_doc_dir))
        title = path.stem
        current_heading = title
        blocks: list[tuple[str, str, int, int]] = []
        cursor = 0
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, flags=re.MULTILINE):
            if match.start() > cursor:
                blocks.append((current_heading, text[cursor : match.start()].strip(), cursor, match.start()))
            current_heading = match.group(2).strip()
            if match.group(1) == "#":
                title = current_heading
            cursor = match.end()
        if cursor < len(text):
            blocks.append((current_heading, text[cursor:].strip(), cursor, len(text)))

        chunks: list[KnowledgeChunk] = []
        chunk_size = max(settings.rag_chunk_size, 300)
        overlap = max(0, min(settings.rag_chunk_overlap, chunk_size // 2))
        for heading, block, block_start, _ in blocks:
            if not block:
                continue
            start = 0
            while start < len(block):
                end = min(len(block), start + chunk_size)
                content = block[start:end].strip()
                if content:
                    chunk_id = hashlib.sha1(f"{relative_path}:{block_start + start}:{content}".encode("utf-8")).hexdigest()[:16]
                    chunks.append(
                        KnowledgeChunk(
                            id=chunk_id,
                            path=relative_path,
                            title=title,
                            heading=heading,
                            content=content,
                            start=block_start + start,
                            end=block_start + end,
                        )
                    )
                if end >= len(block):
                    break
                start = max(end - overlap, start + 1)
        return chunks

    def _embedding_text(self, chunk: KnowledgeChunk) -> str:
        return f"文档：{chunk.title}\n章节：{chunk.heading}\n内容：{chunk.content}"

    def _format_result(self, chunk: KnowledgeChunk, *, score: float, keyword_score: float | None = None) -> dict[str, Any]:
        result = {
            "chunkId": chunk.id,
            "source": chunk.path,
            "title": chunk.title,
            "heading": chunk.heading,
            "content": chunk.content,
            "score": round(float(score), 6),
        }
        if keyword_score is not None:
            result["keywordScore"] = round(float(keyword_score), 6)
        return result


markdown_knowledge_index = MarkdownKnowledgeIndex()
