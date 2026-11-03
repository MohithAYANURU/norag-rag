r"""RAG V2 web application.

Run with:
    .venv\Scripts\python.exe -m uvicorn v2_app:app --reload
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = ROOT.parent / "kestrel-kb" / "docs"
OWN_ROOT = ROOT / "docs"
USER_ROOT = ROOT / "user_docs"
DB_ROOT = ROOT / "chroma_db_v2"
WEB_ROOT = ROOT / "web"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 10
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

app = FastAPI(title="RAG V2", version="2.0")
chroma_client = chromadb.PersistentClient(path=str(DB_ROOT))
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


class DocumentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_. -]+$")
    content: str = Field(min_length=1)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[str] | None = None
    chunk_size: int = Field(DEFAULT_CHUNK_SIZE, ge=100, le=5000)
    overlap_percent: int = Field(DEFAULT_OVERLAP, ge=0, le=80)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    generate: bool = True


class Citation(BaseModel):
    document_id: str
    source: str
    chunk_id: str
    text: str
    distance: float | None = None


class SearchResponse(BaseModel):
    answer: str
    citations: list[Citation]
    question: str
    configuration: dict[str, Any]


class WordsRequest(BaseModel):
    document_ids: list[str] | None = None
    word: str | None = None
    question: str | None = None


class EvaluationRequest(BaseModel):
    corpus: str = Field(pattern=r"^(kestrel|own)$")
    document_ids: list[str] | None = None
    chunk_size: int = Field(DEFAULT_CHUNK_SIZE, ge=100, le=5000)
    overlap_percent: int = Field(DEFAULT_OVERLAP, ge=0, le=80)
    top_k: int = Field(10, ge=1, le=20)


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _builtin_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for root, corpus in ((CORPUS_ROOT, "kestrel"), (OWN_ROOT, "own")):
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                documents.append(
                    {
                        "id": f"{corpus}:{path.name}",
                        "name": path.name,
                        "corpus": corpus,
                        "path": path,
                        "managed": False,
                    }
                )
    if USER_ROOT.exists():
        for path in sorted(USER_ROOT.glob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                documents.append(
                    {
                        "id": f"user:{path.name}",
                        "name": path.name,
                        "corpus": "user",
                        "path": path,
                        "managed": True,
                    }
                )
    return documents


def document_map() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in _builtin_documents()}


def split_text(text: str, chunk_size: int, overlap_percent: int) -> list[str]:
    cleaned = "\n\n".join(
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if len(paragraph.strip()) >= 20
    )
    overlap = int(chunk_size * overlap_percent / 100)
    step = max(1, chunk_size - overlap)
    return [
        chunk.strip()
        for start in range(0, len(cleaned), step)
        if (chunk := cleaned[start : start + chunk_size]).strip()
    ]


def _collection(document_ids: list[str], chunk_size: int, overlap: int):
    key = "|".join(sorted(document_ids)) + f"|{chunk_size}|{overlap}"
    name = f"v2_{_safe_id(key)}"
    collection = chroma_client.get_or_create_collection(
        name=name, embedding_function=embedding_function
    )
    if collection.count() == 0:
        docs = document_map()
        for document_id in document_ids:
            document = docs.get(document_id)
            if not document:
                raise HTTPException(404, f"Unknown document: {document_id}")
            text = document["path"].read_text(encoding="utf-8")
            chunks = split_text(text, chunk_size, overlap)
            if not chunks:
                raise HTTPException(422, f"Document is empty: {document['name']}")
            collection.add(
                documents=chunks,
                ids=[f"{document_id}:{index}" for index in range(len(chunks))],
                metadatas=[
                    {
                        "document_id": document_id,
                        "source": document["name"],
                        "chunk_index": index,
                    }
                    for index in range(len(chunks))
                ],
            )
    return collection


def retrieve(
    question: str,
    document_ids: list[str] | None,
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> list[Citation]:
    docs = document_map()
    selected = document_ids or list(docs)
    if not selected:
        raise HTTPException(422, "No documents are available")
    collection = _collection(selected, chunk_size, overlap)
    result = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))
    texts = result.get("documents", [[]])[0]
    metadata = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        Citation(
            document_id=item["document_id"],
            source=item["source"],
            chunk_id=f"{item['source']}:{item['chunk_index']}",
            text=text,
            distance=distances[index] if index < len(distances) else None,
        )
        for index, (text, item) in enumerate(zip(texts, metadata))
    ]


def generate_answer(question: str, citations: list[Citation]) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(503, "LLM is not configured: set GOOGLE_API_KEY")
    context = "\n\n---\n\n".join(
        f"[{citation.source} / {citation.chunk_id}]\n{citation.text}"
        for citation in citations
    )
    prompt = (
        "Answer using only the supplied excerpts. If they do not contain the "
        "answer, reply exactly that the information is not in the documents. "
        "Do not invent facts or citations.\n\n"
        f"EXCERPTS:\n{context}\n\nQUESTION:\n{question}"
    )
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=os.getenv("RAG_MODEL", "gemini-3.5-flash-lite"),
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise HTTPException(502, "LLM returned an empty answer")
    return answer


def word_analysis(document_ids: list[str] | None, word: str | None, question: str | None):
    docs = document_map()
    selected = document_ids or list(docs)
    chunks: list[tuple[str, str, str]] = []
    for document_id in selected:
        document = docs.get(document_id)
        if not document:
            raise HTTPException(404, f"Unknown document: {document_id}")
        for index, text in enumerate(
            split_text(
                document["path"].read_text(encoding="utf-8"),
                DEFAULT_CHUNK_SIZE,
                DEFAULT_OVERLAP,
            )
        ):
            chunks.append((document_id, f"{document['name']}:{index}", text))

    counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    locations: dict[str, list[dict[str, Any]]] = {}
    for document_id, chunk_id, text in chunks:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        counts.update(tokens)
        for token in set(tokens):
            chunk_counts[token] += 1
            locations.setdefault(token, []).append(
                {"document_id": document_id, "chunk_id": chunk_id, "count": tokens.count(token)}
            )
    if word:
        normalized = word.lower()
        return {
            "word": normalized,
            "count": counts.get(normalized, 0),
            "chunks": locations.get(normalized, []),
        }
    if question:
        words = sorted(set(token.lower() for token in TOKEN_RE.findall(question)))
        return {
            "question": question,
            "words": [
                {
                    "word": token,
                    "exists": token in counts,
                    "count": counts.get(token, 0),
                    "chunks": locations.get(token, []),
                }
                for token in words
            ],
        }
    return {
        "vocabulary": [
            {"word": token, "count": counts[token], "chunk_count": chunk_counts[token]}
            for token in sorted(counts)
        ]
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (WEB_ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/api/documents")
def list_documents():
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "corpus": item["corpus"],
            "managed": item["managed"],
        }
        for item in document_map().values()
    ]


@app.post("/api/documents", status_code=201)
def add_document(payload: DocumentCreate):
    USER_ROOT.mkdir(exist_ok=True)
    path = USER_ROOT / payload.name
    path.write_text(payload.content, encoding="utf-8")
    return {"id": f"user:{path.name}", "name": path.name, "corpus": "user", "managed": True}


@app.delete("/api/documents/{document_id:path}")
def delete_document(document_id: str):
    document = document_map().get(document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    if not document["managed"]:
        raise HTTPException(409, "Built-in documents cannot be deleted")
    document["path"].unlink(missing_ok=False)
    return {"deleted": document_id}


@app.post("/api/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    citations = retrieve(
        payload.question,
        payload.document_ids,
        payload.chunk_size,
        payload.overlap_percent,
        payload.top_k,
    )
    answer = generate_answer(payload.question, citations) if payload.generate else ""
    return SearchResponse(
        answer=answer,
        citations=citations,
        question=payload.question,
        configuration={
            "chunk_size": payload.chunk_size,
            "overlap_percent": payload.overlap_percent,
            "top_k": payload.top_k,
            "embedding_model": EMBEDDING_MODEL,
        },
    )


@app.post("/api/words")
def words(payload: WordsRequest):
    if payload.word and payload.question:
        raise HTTPException(422, "Choose word lookup or question analysis, not both")
    return word_analysis(payload.document_ids, payload.word, payload.question)


@app.post("/api/evaluate")
def evaluate(payload: EvaluationRequest):
    question_file = (
        ROOT.parent / "kestrel-kb" / "eval" / "questions.starter.jsonl"
        if payload.corpus == "kestrel"
        else ROOT / "eval" / "questions.own.jsonl"
    )
    if not question_file.exists():
        raise HTTPException(404, f"Question set not found: {question_file.name}")
    questions = [
        json.loads(line)
        for line in question_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    document_ids = payload.document_ids or [
        item["id"] for item in document_map().values() if item["corpus"] == payload.corpus
    ]
    rows = []
    for question in questions:
        citations = retrieve(
            question["question"],
            document_ids,
            payload.chunk_size,
            payload.overlap_percent,
            payload.top_k,
        )
        hit_rates = {}
        if question["answerable"]:
            for k in (1, 3, 5, 10):
                hit_rates[str(k)] = any(
                    citation.source == evidence["source"]
                    and evidence["quote"].lower() in citation.text.lower()
                    for citation in citations[:k]
                    for evidence in question["evidence"]
                )
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "answerable": question["answerable"],
                "hit_rate": hit_rates,
                "citations": [citation.model_dump() for citation in citations],
            }
        )
    answerable = [row for row in rows if row["answerable"]]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in answerable:
        by_type.setdefault(row["type"], []).append(row)
    return {
        "corpus": payload.corpus,
        "configuration": payload.model_dump(),
        "results": rows,
        "summary": {
            f"hit_rate@{k}": sum(row["hit_rate"].get(str(k), False) for row in answerable)
            / len(answerable)
            if answerable
            else None
            for k in (1, 3, 5, 10)
        },
        "summary_by_type": {
            question_type: {
                f"hit_rate@{k}": sum(
                    row["hit_rate"].get(str(k), False) for row in type_rows
                )
                / len(type_rows)
                for k in (1, 3, 5, 10)
            }
            for question_type, type_rows in by_type.items()
        },
    }
