# RAG V4

A three-person course project by BUI Minh Nhat, Mohith AYANURU, and Maxim Beridze.
The web app answers questions using selected documents and shows the passages
it retrieved. It uses FastAPI, local embeddings and ChromaDB for dense search,
BM25 for keyword search, and a configurable OpenAI-compatible LLM for generation.

## Run the web app

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If you already have a working `.venv`, skip creating it.

### Configure the LLM provider

Set all three values in `.env`. The following example uses Gemini; replace
`your_api_key_here` with a key from [Google AI Studio](https://aistudio.google.com/app/apikey):

```dotenv
LLM_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your_api_key_here
LLM_MODEL=gemini-3.5-flash-lite
```

For another OpenAI-compatible provider, use its base URL, API key, and model
ID. `LLM_URL` is the API base, for example `https://provider.example/v1`,
without `/chat/completions`. Upload scanning requires JSON Schema responses;
the graph builder requires JSON object mode.

An existing Gemini `.env` still works. Leave all three `LLM_*` settings unset
to use `GOOGLE_API_KEY` and optional `RAG_MODEL`. A complete new configuration
takes precedence over the old variables; incomplete or blank new values
are rejected.

### Start the server

```powershell
.\.venv\Scripts\python.exe -m uvicorn v2_app:app --reload --port 8000
```

Open [localhost:8000](http://127.0.0.1:8000/). The module is still called
`v2_app.py` and the page heading still reads **RAG V2** on the V4 branch.

## Use the app

- **Ask:** Select one or more documents and ask a question. Expand **Advanced
  Retrieval Settings** to adjust chunk size, overlap, top-k, or the retrievers.
  Dense and BM25 are enabled by default.
- **Answer style and conversation:** Choose balanced, concise, detailed, or
  bullet points. Successful exchanges stay in the page's memory so follow-ups
  can be rewritten as standalone search queries. Suggested questions fill the
  question box so you can edit them before asking. Reload to start a fresh
  conversation.
- **Source glossary:** Open the collapsible glossary to see terms and their
  definitions from the selected documents.
- **Document Management & Evaluation:** Add a document by entering a filename
  and its text. Use a `.txt` or `.md` extension so it appears in the source list.
- **Words Explorer:** Look up a word or see which words in a question appear
  in the selected documents and chunks.

The bundled sources are in `docs/`. To use the course Kestrel corpus, unpack
`kestrel-kb` alongside this repository so that `../kestrel-kb/docs/` contains
the source files and `../kestrel-kb/eval/questions.starter.jsonl` exists for
the browser's **Evaluate Kestrel** button.

## Build and use a concept graph

1. Select a small document or upload a short excerpt. Uncheck any other documents.
2. Open **Advanced Retrieval Settings** and click **Build graph**.
3. When **Graph ready** appears, open the concept explorer in **List** or
   **Graph** view. Select a concept and expand a link to read its source passage.
   **Ask about this concept** drafts a question you can edit before submitting.
4. To use the graph for search, enable **Concept graph** and ask a question
   with the same documents, chunk size, and overlap.

For a quick test, upload `graph-demo.txt` with the text "Paris is the capital of
France. France is a country in Europe." Select only that file, build the graph,
and ask "What is the capital of France?"

A new build is limited to 50 chunks and makes one model call per chunk. If the
selection is too large, select fewer documents or use a shorter excerpt.
Graphs are saved locally in `graph_store/` and reused when the selected
documents, their contents, chunk size, and overlap match. If any of those
change, build the matching graph or turn off **Concept graph** before asking.

## Upload checks, redaction, and traces

The configured model scans uploaded text for prompt injection before the app
saves or indexes it. Flagged uploads are held for review and can be approved
or discarded. If the scan cannot produce a valid verdict, the upload is
rejected with `503`. The scanner can miss attacks or flag harmless text.

The app masks matching email addresses and common French and North American
phone numbers in questions, conversation history, and document text sent to
the provider for scanning or answer generation. Original source files are
not rewritten.

Traces are saved locally in `traces.jsonl`, which is ignored by Git.

- After an answer, click **Trace Details** to see query rewriting, retrieval,
  and generation timings. Model calls include token counts when the provider
  reports them. The first question skips the rewrite call.
- After adding a document or choosing **Index anyway**, open **Document
  Management & Evaluation** and click **Indexing Trace** to see the save,
  chunking, and embedding steps for the latest successful upload.

To show an estimated cost in search traces, set `LLM_INPUT_USD_PER_1M` and
`LLM_OUTPUT_USD_PER_1M` in `.env` to your model's input and output rates in USD
per million tokens. Restart the app after changing the rates. If either rate
or token count is missing, the estimate is shown as **Unavailable**.

## Quick verification

1. Add the short demo document above and open **Indexing Trace** to check the
   save, chunking, and embedding steps.
2. Select only that document, build its graph, and confirm that a link opens
   the same source passage in both **List** and **Graph** view.
3. Ask the demo question and follow with "Which continent is that country in?" Open
   **Trace Details** after each answer. Check the timings and token counts;
   with rates configured, check that the model calls show an estimated cost.

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
`--no-context --no-retrieval` to ask the configured model without excerpts. See the
[project report](REPORT.md) for decisions and evaluation results.

## Earlier experiments

- [NoRAG](ask_norag_google.py) and [dense RAG](ask_rag_google.py) are standalone
  Gemini demos. [The first RAG version](rag_first_version.py) takes a document
  and question on the command line.
- See the [BM25 exercise](bm25/README.md) and
  [concept graph exercise](concepts/concept_graph_challenge_part1.md) for the
  retrieval experiments. To rebuild the Kestrel graph, run
  `.\.venv\Scripts\python.exe build_graph.py` with the Kestrel corpus and
  an LLM provider configured.
