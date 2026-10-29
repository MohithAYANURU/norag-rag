import argparse
import os
from typing import List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = os.getenv("RAG_MODEL", "gemini-3.5-flash-lite")
STREAM_ON = os.getenv("RAG_STREAM", "true").lower() == "true"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
OVERLAP_PERCENT = 10
MIN_PARAGRAPH_LENGTH = 20
RETRIEVAL_RESULTS = 5

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
embeddings = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap_percent: int = OVERLAP_PERCENT) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_PARAGRAPH_LENGTH]
    text = "\n\n".join(paragraphs)
    overlap = max(0, int(chunk_size * overlap_percent / 100))
    step = max(1, chunk_size - overlap)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]
    return [chunk for chunk in chunks if len(chunk.strip()) >= 10]


def get_collection_name(document_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(document_path))[0]
    return f"{base_name}_{CHUNK_SIZE}_{OVERLAP_PERCENT}_{MIN_PARAGRAPH_LENGTH}"


def ensure_index(document_path: str):
    collection_name = get_collection_name(document_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embeddings,
    )

    if collection.count() > 0:
        print(f"Using existing index: {collection_name} ({collection.count()} chunks)")
        return collection

    print(f"Indexing {document_path} into {collection_name} ...")
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text_into_chunks(text)
    collection.add(documents=chunks, ids=[f"id_{i}" for i in range(len(chunks))])
    print(f"Indexed {len(chunks)} chunks")
    return collection


def retrieve_context(collection, question: str, n_results: int = RETRIEVAL_RESULTS) -> str:
    results = collection.query(query_texts=[question], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    return "\n---\n".join(documents)


def generate_answer(question: str, retrieved_context: str) -> str:
    prompt = f"""
You are a helpful assistant. Answer the user question using only the retrieved context below.
If the answer is not in the retrieved context, say that the information is not available in the provided documents.

RETRIEVED CONTEXT:
{retrieved_context}

QUESTION:
{question}
"""

    if STREAM_ON:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        lines = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                lines.append(delta)
        print()
        return "".join(lines)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    print(answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Simple first-version RAG app")
    parser.add_argument("--document", default="docs/CrimeAndPunishment.txt", help="Path to the document to index")
    parser.add_argument("--question", default="What does this say about Denis Amselemovitch?", help="Question to answer")
    parser.add_argument("--results", type=int, default=RETRIEVAL_RESULTS, help="Number of retrieved chunks")
    args = parser.parse_args()

    if not os.path.exists(args.document):
        raise FileNotFoundError(f"Document not found: {args.document}")

    collection = ensure_index(args.document)
    context = retrieve_context(collection, args.question, n_results=args.results)
    print("\n--- RETRIEVED CONTEXT ---")
    print(context)
    print("\n--- ANSWER ---")
    generate_answer(args.question, context)


if __name__ == "__main__":
    main()
