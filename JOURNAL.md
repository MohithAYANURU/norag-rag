# RAG lab journal

## 2026-10-28 - NoRAG versus RAG needle experiment

### Setup

- Repository: `norag-rag`
- Python: 3.13
- Model: `gemini-3.5-flash-lite`
- Provider: Google Gemini OpenAI-compatible API
- Haystack: `docs/CrimeAndPunishment.txt`
- Needle: the injected Denis Amselemovitch passage
- RAG chunk size: 800 characters
- RAG overlap: 10%
- Minimum paragraph length: 20 characters
- Retrieved chunks: 5
- Embeddings: local `all-MiniLM-L6-v2`
- Vector store: persistent local ChromaDB

### NoRAG run

Command:

```powershell
.venv\Scripts\python.exe ask_norag_google.py
```

The script sent a large full-context prompt containing the truncated book
text and the injected passage. The response to
`What does this say about Denis Amselemovitch?` stated that Denis was not
mentioned in the provided excerpt and gave no useful description.

### RAG run

Command:

```powershell
"y" | .venv\Scripts\python.exe ask_rag_google.py
```

ChromaDB indexed 1,665 chunks locally. Retrieval returned relevant excerpts
containing the injected Denis passage. The generated answer correctly
identified Denis as a brief character, a friend of Raskolnikov, comic relief,
and a possible future/AI artifact.

### Observation

The comparison demonstrates the needle-in-a-haystack failure mode. Supplying
the whole context does not guarantee that the model will use a small,
middle-of-document fact. RAG first narrows the context with local semantic
retrieval, making the relevant passage prominent before generation. The RAG
answer was grounded in the retrieved excerpts, while the NoRAG answer was
not.

### Reproduction

```powershell
cd "C:\Users\mohit\Desktop\Bsc\S5\AI Intregation\norag-rag"
.venv\Scripts\python.exe ask_norag_google.py
"y" | .venv\Scripts\python.exe ask_rag_google.py
```

The API key is loaded from the ignored `.env` file and is not stored in the
repository.

## Follow-up experiments

### Second literary question: Marmeladov

Using the original 800-character chunks and 10% overlap, the RAG pipeline
retrieved passages describing Marmeladov's work history, alcoholism, family,
poverty, and relationship with his wife and daughter. Gemini produced a
multi-point answer grounded in those passages.

### Chunking experiment: 400 characters / 20% overlap

The Denis question was repeated with:

- `RAG_CHUNK_SIZE=400`
- `RAG_OVERLAP_PERCENT=20`
- `RAG_RESULTS=5`

This configuration indexed 3,599 chunks. It still retrieved the injected
Denis passage and generated the same correct classification. Smaller chunks
increased the index size and returned more narrowly focused excerpts, but
also made some retrieved passages end mid-sentence.

### Second document: Vellora Duo espresso machine

The RAG pipeline was switched to
`docs/vellora-duo-espresso-machine.txt` using 400-character chunks and 20%
overlap. It indexed 97 chunks and answered:
`How do I clean and descale the Vellora Duo, and what happens if I ignore
limescale?`

The retrieved context supported the following answer: use the cleaning tablet
programme for cleaning; remove the water filter and use the recommended
descaling liquid for the descaling programme; never use vinegar; descaling
takes about 30 minutes and includes rinsing; ignoring limescale reduces
heating efficiency, lowers coffee temperature, can block the machine, and
can result in warranty-excluded damage.

### Follow-up conclusion

The implementation now covers both the literary needle-in-a-haystack example
and a practical support-document use case. The chunking settings are
configurable through environment variables, so experiments do not require
source edits or accidental mixing of indexes.
