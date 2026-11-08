"""Build the Kestrel concept graph with validated Gemini triples.

The output is deliberately a small JSON artifact so graph retrieval is local
and repeatable after the one-time extraction cost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from retrieval_v3 import ConceptGraph, Entity, Relation

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / "kestrel-kb" / "docs"
OUTPUT = ROOT / "graph_kestrel.json"
load_dotenv(ROOT / ".env")


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


def extract(client: OpenAI, chunk_id: str, text: str) -> Extraction:
    prompt = (
        "Extract only facts explicitly stated in this text. Return JSON with "
        "entities (id, name, type, chunk_ids) and relations "
        "(subject, predicate, object, chunk_ids). Use stable lowercase ids, "
        f"and set chunk_ids to ['{chunk_id}'] for every extracted item.\n\n{text}"
    )
    response = client.chat.completions.create(
        model=os.getenv("RAG_MODEL", "gemini-3.5-flash-lite"),
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Empty graph response for {chunk_id}")
    return Extraction.model_validate_json(content)


def chunks(text: str) -> list[str]:
    cleaned = "\n\n".join(
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if len(paragraph.strip()) >= 20
    )
    return [
        cleaned[start : start + 800]
        for start in range(0, len(cleaned), 720)
        if cleaned[start : start + 800].strip()
    ]


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY before building the graph")
    if not CORPUS.exists():
        raise SystemExit(f"Kestrel corpus not found: {CORPUS}")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}
    for path in sorted(CORPUS.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunks(text)):
            chunk_id = f"{path.name}:{index}"
            extraction = extract(client, chunk_id, chunk)
            for entity in extraction.entities:
                entities.setdefault(entity.id, entity)
            for relation in extraction.relations:
                key = (relation.subject, relation.predicate, relation.object)
                relations.setdefault(key, relation)
    graph = ConceptGraph(entities=list(entities.values()), relations=list(relations.values()))
    OUTPUT.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {len(graph.entities)} entities and {len(graph.relations)} relations to {OUTPUT}")


if __name__ == "__main__":
    main()
