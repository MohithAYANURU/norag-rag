import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str


def get_llm_config() -> LLMConfig:
    names = ("LLM_URL", "LLM_API_KEY", "LLM_MODEL")

    if any(name in os.environ for name in names):
        url, key, model = ((os.getenv(name) or "").strip() for name in names)
        if not (url and key and model):
            raise ValueError("Set LLM_URL, LLM_API_KEY, and LLM_MODEL together.")
        return LLMConfig(url, key, model)

    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError("Set the three LLM settings or GOOGLE_API_KEY.")

    return LLMConfig(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        key,
        (os.getenv("RAG_MODEL") or "").strip() or "gemini-3.5-flash-lite",
    )