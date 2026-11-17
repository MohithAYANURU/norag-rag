import json
import time
import uuid
from pathlib import Path
from typing import Any
from privacy import redact_pii

TRACE_FILE = Path(__file__).resolve().parent / "traces.jsonl"


def redact_trace(value):
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, list):
        return [redact_trace(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_trace(item) for key, item in value.items()}
    return value


class Trace:
    def __init__(self, name: str):
        self.trace_id = str(uuid.uuid4())
        self.name = name
        self.spans = []
        self.start_time = time.time()

    def add_span(self, name: str, metadata: dict[str, Any] | None = None):
        return Span(self, name, metadata or {})

    def finalize(self):
        duration = time.time() - self.start_time
        trace_data = {
            "trace_id": self.trace_id,
            "name": self.name,
            "duration_ms": round(duration * 1000, 2),
            "spans": self.spans,
        }
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(redact_trace(trace_data)) + "\n")
        return self.trace_id


class Span:
    def __init__(self, trace: Trace, name: str, metadata: dict):
        self.trace = trace
        self.name = name
        self.metadata = metadata
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.metadata["error"] = exc_type.__name__
        self.trace.spans.append(
            {
                "name": self.name,
                "duration_ms": round(duration * 1000, 2),
                "metadata": self.metadata,
            }
        )


TRACE_VIEW_FIELDS = {
    "chunk_count",
    "embedding_model",
    "chunks_retrieved",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "error",
}


def read_trace(trace_id: str) -> dict[str, Any] | None:
    try:
        with TRACE_FILE.open(encoding="utf-8") as log:
            for line in log:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict) or record.get("trace_id") != trace_id:
                    continue

                spans = []
                for span in record.get("spans") or []:
                    if not isinstance(span, dict):
                        continue
                    metadata = span.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    spans.append(
                        {
                            "name": span["name"],
                            "duration_ms": span["duration_ms"],
                            "metadata": {
                                key: value
                                for key, value in metadata.items()
                                if key in TRACE_VIEW_FIELDS
                            },
                        }
                    )

                return redact_trace(
                    {
                        "trace_id": trace_id,
                        "name": record["name"],
                        "duration_ms": record["duration_ms"],
                        "spans": spans,
                    }
                )
    except FileNotFoundError:
        pass

    return None
