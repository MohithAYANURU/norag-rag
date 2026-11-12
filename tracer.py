import json
import time
import uuid
from pathlib import Path
from typing import Any

TRACE_FILE = Path(__file__).resolve().parent / "traces.jsonl"

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
            "spans": self.spans
        }
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data) + "\n")
        return self.trace_id

class Span:
    def __init__(self, trace: Trace, name: str, metadata: dict):
        self.trace = trace
        self.name = name
        self.metadata = metadata
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.metadata["error"] = str(exc_val)
        self.trace.spans.append({
            "name": self.name,
            "duration_ms": round(duration * 1000, 2),
            "metadata": self.metadata
        })