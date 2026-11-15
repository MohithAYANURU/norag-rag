import re

EMAIL_RE = re.compile(
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+",
    re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:\+33[ .-]?|0)[1-9](?:[ .-]?\d{2}){4}"
    r"|(?:\+1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}"
    r")(?!\w)"
)


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED]", text)
    return PHONE_RE.sub("[REDACTED]", text)