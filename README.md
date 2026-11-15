# RAG V4

A three-person course project by BUI Minh Nhat, Mohith AYANURU, and Maxim Beridze.
The web app answers questions using selected documents and shows the passages
it retrieved. It uses FastAPI, local embeddings and ChromaDB for dense search,
BM25 for keyword search, and Google Gemini for generation.

## Run the web app

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If you already have a working `.venv`, skip creating it. Set `GOOGLE_API_KEY`
in `.env` for answers and document uploads. Get a key from
[Google AI Studio](https://aistudio.google.com/app/apikey). Set `RAG_MODEL`
there if you want to use a different Gemini model.

```powershell
.\.venv\Scripts\python.exe -m uvicorn v2_app:app --reload --port 8000
```

Open [localhost:8000](http://127.0.0.1:8000/). The module is still called
`v2_app.py` and the page heading still reads **RAG V2** on the V4 branch.

## Use the app

- **Ask:** Select one or more documents and ask a question. Expand **Advanced
  Retrieval Settings** to adjust chunk size, overlap, top-k, or the retrievers.
  Dense and BM25 are enabled by default; the optional concept graph uses
  `graph_kestrel.json`.
- **Answer style:** Choose balanced, concise, detailed, or bullet points.
  Successful exchanges are kept in the page's memory for follow-up questions.
  Reload the page to start a fresh conversation.
- **Document Management & Evaluation:** Add a document by entering a filename
  and its text. Use a `.txt` or `.md` extension so it appears in the source list.
- **Words Explorer:** Look up a word or see which words in a question appear
  in the selected documents and chunks.

The bundled sources are in `docs/`. To use the course Kestrel corpus, unpack
`kestrel-kb` alongside this repository so that `../kestrel-kb/docs/` contains
the source files and `../kestrel-kb/eval/questions.starter.jsonl` exists for
the browser's **Evaluate Kestrel** button.

## Upload checks, redaction, and traces

Gemini scans uploaded text for prompt injection before the app saves or indexes
it. Flagged uploads return `400 Quarantine`. If the scan cannot produce a
valid verdict, the upload is rejected with `503`. The scanner is a model-based
check and can miss attacks or flag harmless text.

The app masks matching email addresses and common French and North American
phone numbers in the current search question and conversation history.
Redaction applies to those search inputs; uploaded document text is sent to
Gemini for scanning, and retrieved excerpts are sent for answer generation.

Generation and indexing spans are written locally to `traces.jsonl`. The file
is ignored by Git.

## Run retrieval evaluation

Keep the web app running. In a second PowerShell terminal at the repository
root, run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --questions eval/questions.own.jsonl --document-id own:vellora-duo-espresso-machine.txt --output eval/local-own-results.json
```

For the Kestrel CLI evaluation, make the Kestrel corpus available as described
above, then run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --questions eval/questions.kestrel.jsonl `
  --document-id kestrel:product-guide.md --document-id kestrel:billing-policy-2025.md `
  --document-id kestrel:billing-policy-2026.md --document-id kestrel:service-catalog.md `
  --document-id kestrel:support-runbook-errors.md --output eval/local-kestrel-results.json
```

The evaluator reports HitRate@1, @3, @5, and @10 overall and by question type.
Add `--answers` to generate grounded answers, or use
`--no-context --no-retrieval` to run the Gemini-only baseline. See the
[project report](REPORT.md) for decisions and evaluation results.

## Earlier experiments

- [NoRAG](ask_norag_google.py) and [dense RAG](ask_rag_google.py) are standalone
  Gemini demos. [The first RAG version](rag_first_version.py) takes a document
  and question on the command line.
- See the [BM25 exercise](bm25/README.md) and
  [concept graph exercise](concepts/concept_graph_challenge_part1.md) for the
  retrieval experiments. To rebuild the Kestrel graph, run
  `.\.venv\Scripts\python.exe build_graph.py` with the Kestrel corpus and a
  Gemini key configured.
