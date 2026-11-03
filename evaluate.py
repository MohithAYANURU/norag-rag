r"""Repeatable retrieval evaluation for the RAG V2 API.

Start the app first, then run:
    .venv\Scripts\python.exe evaluate.py --questions ..\kestrel-kb\eval\questions.starter.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
K_VALUES = (1, 3, 5, 10)


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relevant(citation: dict[str, Any], evidence: list[dict[str, str]]) -> bool:
    return any(
        citation["source"] == item["source"] and item["quote"].lower() in citation["text"].lower()
        for item in evidence
    )


def evaluate_retrieval(questions: list[dict[str, Any]], config: dict[str, Any]):
    rows = []
    for question in questions:
        response = requests.post(
            f"{API_URL}/api/search",
            json={**config, "question": question["question"], "generate": False},
            timeout=180,
        )
        response.raise_for_status()
        citations = response.json()["citations"]
        row = {
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "answerable": question["answerable"],
            "citations": citations,
            "hit_rate": {},
        }
        if question["answerable"]:
            for k in K_VALUES:
                row["hit_rate"][str(k)] = any(
                    relevant(citation, question["evidence"]) for citation in citations[:k]
                )
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": {}, "by_type": {}}
    answerable = [row for row in rows if row["answerable"]]
    for key in ("overall", "by_type"):
        groups = {"all": answerable} if key == "overall" else defaultdict(list)
        if key == "by_type":
            for row in answerable:
                groups[row["type"]].append(row)
        for group, group_rows in groups.items():
            result[key][group] = {
                f"hit_rate@{k}": round(
                    sum(row["hit_rate"].get(str(k), False) for row in group_rows)
                    / len(group_rows),
                    4,
                )
                if group_rows
                else None
                for k in K_VALUES
            }
    return result


def no_context(question: str) -> dict[str, str]:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return {"status": "skipped", "answer": "SKIPPED: GOOGLE_API_KEY is not configured"}
    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("RAG_MODEL", "gemini-3.5-flash-lite"),
            messages=[{"role": "user", "content": question}],
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    return {
        "status": "ok",
        "answer": response.choices[0].message.content or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip the API retrieval pass and write only the no-context results",
    )
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    questions = load_questions(args.questions)
    config = {
        "chunk_size": args.chunk_size,
        "overlap_percent": args.overlap,
        "top_k": args.top_k,
    }
    if args.document_ids:
        config["document_ids"] = args.document_ids
    rows = [] if args.no_retrieval else evaluate_retrieval(questions, config)
    output = {
        "questions_file": str(args.questions),
        "configuration": config,
        "summary": summarize(rows),
        "results": rows,
    }
    if args.no_context:
        output["no_context"] = [
            {"id": question["id"], **no_context(question["question"])}
            for question in questions
        ]
    if args.answers:
        for row in rows:
            response = requests.post(
                f"{API_URL}/api/search",
                json={**config, "question": row["question"], "generate": True},
                timeout=180,
            )
            response.raise_for_status()
            row["answer"] = response.json()["answer"]
    target = args.output or ROOT / "eval" / "results.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
