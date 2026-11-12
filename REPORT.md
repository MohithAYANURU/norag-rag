# RAG V2 report

## Team

Mohith AYANURU (individual submission).

## Scope and decisions

- The application is implemented in `v2_app.py` with FastAPI, Pydantic response
  models, ChromaDB, and local `all-MiniLM-L6-v2` embeddings.
- The provided Kestrel corpus is loaded from `..\kestrel-kb\docs`; the submitted
question copy is `eval/questions.kestrel.jsonl`.
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
.venv\Scripts\python.exe evaluate.py --questions eval\questions.kestrel.jsonl --document-id kestrel:product-guide.md --document-id kestrel:billing-policy-2025.md --document-id kestrel:billing-policy-2026.md --document-id kestrel:service-catalog.md --document-id kestrel:support-runbook-errors.md --output eval\kestrel-results.json
.venv\Scripts\python.exe evaluate.py --questions eval\questions.own.jsonl --document-id own:vellora-duo-espresso-machine.txt --output eval\own-results.json
.venv\Scripts\python.exe evaluate.py --questions eval\questions.kestrel.jsonl --no-context --no-retrieval --output eval\kestrel-no-context.json
.venv\Scripts\python.exe evaluate.py --questions eval\questions.own.jsonl --no-context --no-retrieval --output eval\own-no-context.json
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

Per-type movement (800/10% → 400/20%) was:

- Kestrel: `exact_identifier` stayed 0.5 at @1/@3/@5 but fell from 1.0 to
  0.5 at @10; `versions` improved @3 from 0.5 to 1.0; `two_hop` fell from
  1.0 to 0.0 at every cutoff; `split_answer` and `vocabulary` were unchanged.
- Vellora: `exact_identifier` stayed perfect; `split_answer` improved @1
  from 0.3333 to 1.0; `other` improved @1 from 0.0 to 0.3333 and was
  unchanged at @3/@5/@10.

Thus smaller chunks improved Vellora @1, but reduced Kestrel @10, driven by
the two-hop and exact-identifier misses.

## No-context baseline

The no-context command was run for both fixed question sets with
`--no-context --no-retrieval`. Gemini returned HTTP 429 rate-limit errors; the
per-question status and error text are preserved in
`eval/kestrel-no-context.json` and `eval/own-no-context.json`. These are
reproducible test results, not successful answer baselines.


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

- No-context calls completed, but the returned answers were ungrounded and
  frequently asked for product context; they are retained as the baseline.
- Retrieval answer-generation runs with `--answers` were not rerun after the
  final no-context pass, so no generated-answer comparison is claimed.
- The application reports controlled API errors for missing keys, unknown
  documents, invalid parameters, and empty LLM responses.


## V4 Decision Log

### Part A: Observability (Completed by BUI Minh Nhat)

**What do you trace? Where are the traces stored, and for how long?**
We trace both user queries (`POST /api/search`) and document indexing (`POST /api/documents`). 
*   **Search traces** record spans for retrieval (capturing retriever flags and `chunk_ids` found) and generation (capturing total tokens, latency, and estimated USD cost). 
*   **Indexing traces** record spans for saving the file, chunking it (capturing chunk counts), and vector indexing (capturing the embedding model and ChromaDB collection). 
All traces are stored locally in a zero-dependency `traces.jsonl` file in the root directory. They persist indefinitely on the local disk unless manually deleted.

**Who can read them? Do they leave your machine?**
Traces are strictly local and never leave the host machine. They can only be read by someone with direct file system access to `traces.jsonl` or someone accessing the local `/api/traces/` web endpoint. 

**What data is excluded from traces?**
To protect data privacy and keep log sizes manageable, we strictly forbid API keys, raw document text, and full LLM-generated answers from entering the trace logs. We log `chunk_ids` as references instead of the text itself. *(Note: Personal data redaction is handled prior to tracing via our Part C implementation).*

**Which other options did you consider, and why did you reject them?**
We considered using dedicated observability platforms like Langfuse (cloud-hosted) and Phoenix (self-hosted). We rejected them to strictly adhere to the V4 requirement that the application must be "Easy to install and run." Requiring the grader to configure external cloud API keys or spin up Docker containers introduces unnecessary friction. A local JSONL file approach is bulletproof, zero-dependency, and works instantly out of the box.

---
### Part B: New Features (To be completed by Teammate 2)
*(Teammate 2: Document your feature's cost, latency, failure handling, and testing edge cases here)*

---
### Part C: Responsible AI (To be completed by Teammate 3)
*(Teammate 3: Document your injection scanner methodology and personal data redaction policy here)*