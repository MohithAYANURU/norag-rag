from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequency = [Counter(tokenize(item["text"])) for item in chunks]
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequency:
            self.document_frequency.update(frequencies.keys())
        self.average_length = (
            sum(sum(item.values()) for item in self.term_frequency) / len(chunks)
            if chunks
            else 0
        )

    def search(self, question: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        query_terms = set(tokenize(question))
        total = len(self.chunks)
        scored: list[tuple[dict[str, Any], float]] = []
        for index, frequencies in enumerate(self.term_frequency):
            length = sum(frequencies.values())
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                normalizer = 1 - self.b + self.b * length / max(self.average_length, 1)
                score += idf * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * normalizer
                )
            if score:
                scored.append((self.chunks[index], score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


class Entity(BaseModel):
    id: str
    name: str
    type: str
    chunk_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    chunk_ids: list[str] = Field(default_factory=list)


class ConceptGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "ConceptGraph":
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"Invalid concept graph {path}: {exc}") from exc

    def retrieve(self, question: str, chunks: dict[str, dict[str, Any]], top_k: int):
        query = set(tokenize(question))
        matched = {
            entity.id
            for entity in self.entities
            if query.intersection(tokenize(entity.name))
        }
        if not matched:
            return [], {"entities": [], "edges": []}
        edges = [
            relation
            for relation in self.relations
            if relation.subject in matched or relation.object in matched
        ]
        chunk_scores: defaultdict[str, float] = defaultdict(float)
        for relation in edges:
            for chunk_id in relation.chunk_ids:
                chunk_scores[chunk_id] += 1.0
        ranked = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:top_k]
        return [
            (chunks[chunk_id], chunk_scores[chunk_id])
            for chunk_id in ranked
            if chunk_id in chunks
        ], {
            "entities": sorted(matched),
            "edges": [relation.model_dump() for relation in edges],
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[dict[str, Any], float]]],
    top_k: int,
) -> list[tuple[dict[str, Any], float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    items: dict[str, dict[str, Any]] = {}
    for results in ranked_lists:
        for rank, (item, _score) in enumerate(results, 1):
            key = item["chunk_id"]
            items[key] = item
            scores[key] += 1 / (60 + rank)
    return [
        (items[key], score)
        for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    ]
