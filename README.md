# norag-rag project

## RAG V2 web application

The V2 application is implemented in `v2_app.py`. It exposes document management,
retrieval and grounded answering through a FastAPI API, with a browser UI at `/`.
The Words tab is available in the UI.

From this directory:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn v2_app:app --reload
```

The Kestrel corpus is discovered from `..\kestrel-kb\docs`. The own corpus uses
the files in `docs`. The Kestrel question set is copied in Kestrel JSONL format at
`eval\questions.kestrel.jsonl`; the own question set is
`eval\questions.own.jsonl`.

Run retrieval evaluation while the API is running:

```powershell
.venv\Scripts\python.exe evaluate.py --questions eval\questions.kestrel.jsonl --document-id kestrel:product-guide.md --document-id kestrel:billing-policy-2025.md --document-id kestrel:billing-policy-2026.md --document-id kestrel:service-catalog.md --document-id kestrel:support-runbook-errors.md --output eval\kestrel-results.json
.venv\Scripts\python.exe evaluate.py --questions eval\questions.own.jsonl --document-id own:vellora-duo-espresso-machine.txt --output eval\own-results.json
```

The evaluator reports HitRate@1, @3, @5 and @10 overall and per question type.
Use `--no-context --no-retrieval` for the no-context baseline and `--answers`
to request and save LLM answers. API errors are saved per question instead of
discarding the result file. Keep result files and `eval\hand-grades.json` with
the submission.

To open the Words tab, start the API, browse to `http://127.0.0.1:8000/`, and
select **Words** in the top navigation. Choose a document and enter a word or
question to inspect vocabulary, word-to-chunk, and question-to-word results.

This workspace contains 2 Python applications to compare **NoRAG** (full-context prompting) versus **RAG** (retrieval-augmented generation) using **Google Gemini** via the **OpenAI SDK**.

## Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini via OpenAI SDK (`gemini-3.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- a baseline question about Marmeladov (present in the original book)
	- a needle question about Denis (the injected passage)

2. `ask_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Default retrieval uses `n_results=5` for the question: `What does this say about Denis?`

3. `bm25/ask_bm25_rag_google.py`
- RAG with Google Gemini using in-memory BM25 lexical retrieval.
- See [`bm25/README.md`](bm25/README.md) for the comparison with dense retrieval and the BM25 challenge.

The two-hop retrieval exercise is documented in
[`concepts/concept_graph_challenge_part1.md`](concepts/concept_graph_challenge_part1.md).

## Project Structure

- `docs/CrimeAndPunishment.txt`: main source text used by the scripts.
- `docs/CrimeAndPunishment-shorter.txt`: shortened version of the main source text for quicker testing and rapid experimentation in Gemini online at https://gemini.google.com/app (See note below)
- `chroma_db/`: persistent local vector database (used by RAG scripts).
- `bm25/`: BM25 retrieval example and challenges.
- `concepts/`: multi-hop retrieval challenge.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+ (recommended)
- API key:
   - `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)


## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and set your real keys.

## Run Each Application

With your virtual environment activated:

### 1) NoRAG + Google

```bash
python ask_norag_google.py
```

### 2) RAG + Google

```bash
python ask_rag_google.py
```

## Notes

- Both applications use the OpenAI SDK to interact with Google Gemini via Google's OpenAI-compatible endpoint.
- The RAG app splits the text into fixed-size chunks with an overlap (`CHUNK_SIZE`, `OVERLAP_PERCENT` and `MIN_PARAGRAPH_LENGTH` at the top of the script), indexes them into `chroma_db/` and reuses them across runs.
- There is one index per document (`DOCUMENT_FILE`) and chunking configuration. When no index exists for the current document and settings, the script asks before building one (this takes a little while).
- Debug prompt printing can be enabled by passing `debug=True` in the NoRAG app.
- RAG app prints retrieved context for verification.
- Streaming output is enabled by default in both applications (controlled by `STREAM_ON` variable).

### First-version RAG app

The repository now also includes `rag_first_version.py`, a cleaner first pass at a
real RAG application. It reads a text file, chunks it, indexes the chunks in a
persistent local ChromaDB collection, retrieves the most relevant passages, and
then asks Gemini to answer the user’s question using only those passages.

```powershell
.venv\Scripts\python.exe rag_first_version.py --document docs/CrimeAndPunishment.txt --question "What does this say about Denis Amselemovitch?"
```

This is the simplest working version of RAG for the course: retrieval + context
injection + generation.

The RAG script also accepts environment overrides for experiments without
editing the source:

```powershell
$env:RAG_DOCUMENT="docs/vellora-duo-espresso-machine.txt"
$env:RAG_QUESTION="How do I clean and descale the machine?"
$env:RAG_CHUNK_SIZE="400"
$env:RAG_OVERLAP_PERCENT="20"
$env:RAG_RESULTS="5"
$env:RAG_STREAM="false"
python ask_rag_google.py
```

The persistent collection name includes the document and chunking settings,
so changing these values creates an independent index.

- Using the shortened version of the text (`docs/CrimeAndPunishment-shorter.txt`)
	- Go online at https://gemini.google.com/app
	- At the prompt, start writing "Here is a story: ",
	- Then paste the content of docs/CrimeAndPunishment-shorter.txt into the Gemini chat app and then, at the end of the prompt, ask "What can you tell me about Denis Amselemovitch?"
	- Observe the response from Gemini.

# Free books from Project Gutenberg
- https://www.gutenberg.org/
# Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
# Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76
