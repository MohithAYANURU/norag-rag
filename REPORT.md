# RAG V2 report

## Team

Mohith AYANURU (individual submission).

## Scope and decisions

- The application is implemented in `v2_app.py` with FastAPI, Pydantic response
  models, ChromaDB, and local `all-MiniLM-L6-v2` embeddings.
- The provided Kestrel corpus is loaded from `..\kestrel-kb\docs`.
- The own corpus is the Vellora Duo espresso-machine manual.
- The own question set contains 10 fixed JSONL questions in
  `eval/questions.own.jsonl`.
- The baseline configuration is 800-character chunks, 10% overlap, and
  `all-MiniLM-L6-v2`. The comparison configuration is 400-character chunks,
  20% overlap, with the same embedding model.
- Retrieval relevance is measured by whether a retrieved chunk has the
  question's evidence source and contains its exact evidence quote.
- Unanswerable questions are excluded from HitRate calculations.

## How to run

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn v2_app:app --reload
```

Open `http://127.0.0.1:8000/`. The Words tab supports vocabulary, word-to-chunk,
and question-to-word analysis. `/about` is intentionally unlinked.

Evaluation:

```powershell
.venv\Scripts\python.exe evaluate.py --questions ..\kestrel-kb\eval\questions.starter.jsonl --document-id kestrel:product-guide.md --document-id kestrel:billing-policy-2025.md --document-id kestrel:billing-policy-2026.md --document-id kestrel:service-catalog.md --document-id kestrel:support-runbook-errors.md --output eval\kestrel-results.json
.venv\Scripts\python.exe evaluate.py --questions eval\questions.own.jsonl --document-id own:vellora-duo-espresso-machine.txt --output eval\own-results.json
```

## Retrieval results

The results below are generated from the saved JSON files. Values are HitRate,
and are reported only over answerable questions.

| Corpus | Configuration | @1 | @3 | @5 | @10 |
|---|---|---:|---:|---:|---:|
| Kestrel | 800 / 10% | 0.625 | 0.750 | 0.875 | 1.000 |
| Kestrel | 400 / 20% | 0.500 | 0.750 | 0.750 | 0.750 |
| Vellora | 800 / 10% | 0.375 | 0.750 | 0.875 | 0.875 |
| Vellora | 400 / 20% | 0.750 | 0.750 | 0.875 | 0.875 |

Per-type results are in:

- `eval/kestrel-results.json`
- `eval/kestrel-results-400-20.json`
- `eval/own-results.json`
- `eval/own-results-400-20.json`

The comparison shows that smaller chunks improved Vellora @1 but reduced
Kestrel @10, especially for two-hop and exact-identifier questions.

## No-context baseline

The no-context command was attempted for both fixed question sets. Gemini
returned HTTP 429 rate-limit responses before the Kestrel baseline could be
completed. The Vellora output was also started but should not be treated as a
valid baseline until rerun after the quota resets. The incomplete files are
preserved under `eval/` where present.

## Answer grading

Required grading labels are `correct`, `partial`, `wrong`, and `unsupported`.
For unanswerable questions, only an explicit “not in the documents” answer is
correct. When grading an incorrect answer, compare whether the evidence chunk
was retrieved (generation failure) or not retrieved (retrieval failure).

Hand grading is intentionally left for the student to perform after rerunning
the answer-generation evaluation with an available Gemini quota. No grades are
invented from retrieval scores.

## Two diagnosed misses

### Kestrel: two-hop error question

With 400-character chunks and 20% overlap, the `two_hop` group reached 0.0 even
at @10. The answer requires joining the runbook's `KST-5031` → `postroom`
mapping with the service catalog's `postroom` → Messaging/channel mapping.
Shorter chunks separate the two pieces and the question does not contain the
intermediate service name. The Words tab should be used to inspect `postroom`,
`messaging`, and the error-code tokens.

### Vellora: setup/maintenance questions

With 800-character chunks and 10% overlap, the `other` group reached only
0.6667 at @10. Several questions use ordinary wording while the manual uses
specific section language, and fixed character windows can split setup or
maintenance instructions. The Words tab can distinguish absent terms from
terms that occur in many chunks; this is especially useful for questions using
“app”, “price”, or other words absent from the manual.

## Known failures and next action

- Gemini quota/rate limiting prevented a complete no-context baseline and
  answer-generation run.
- Hand answer grades must be added after the quota resets.
- The application reports controlled API errors for missing keys, unknown
  documents, invalid parameters, and empty LLM responses.
