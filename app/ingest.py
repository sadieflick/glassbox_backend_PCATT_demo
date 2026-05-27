"""
Ingest PCATT courses into ChromaDB for Mode B (Vector RAG).

Run this once before starting the server:
  venv/bin/python -m app.ingest

What this script does:
  1. Reads pcatt_courses.csv
  2. Builds one Document per course — combining title, description, and learner outcomes
  3. Embeds each document using nomic-embed-text (via Ollama)
  4. Persists the vector store to disk at ./chroma_db

On query time, Mode B will load this store and search it without re-embedding.
"""

import csv
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

COURSES_CSV = Path(__file__).parent.parent.parent.parent / "data" / "pcatt_courses.csv"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


def build_documents(csv_path: Path) -> list[Document]:
    """
    Turn each CSV row into a LangChain Document.

    A Document has two parts:
      - page_content: the text that gets embedded and searched
      - metadata: structured fields attached to the result (not embedded, just returned)

    We combine title + description + learner outcomes into page_content so the
    embedding captures the full meaning of the course, not just its title.
    """
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            course_id = f"{row['course_prefix']}-{row['course_number']}"
            title = row["course_title"].strip()
            desc = row["course_desc"].strip()
            outcomes = row["metadata"].strip()

            # This is the text the embedding model will "read."
            # Putting the title first ensures the embedding is anchored to the course identity.
            page_content = f"Course: {title}\n\nDescription: {desc}\n\nDetails: {outcomes}"

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "course_id": course_id,
                        "title": title,
                        "dept": row["dept_name"],
                        "institution": row["inst_ipeds"],
                    },
                )
            )
    return docs


def main():
    print(f"Loading courses from: {COURSES_CSV}")
    docs = build_documents(COURSES_CSV)
    print(f"  Built {len(docs)} documents")

    # OllamaEmbeddings calls your local Ollama instance to embed text.
    # nomic-embed-text produces 768-dimension vectors and is fast on M-series chips.
    print("Connecting to Ollama embedding model (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Chroma.from_documents embeds every document and writes the vectors to disk.
    # This will take 1-3 minutes for 138 courses on an M3.
    print(f"Embedding and storing to: {CHROMA_DIR}")
    print("  This will take a few minutes...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="pcatt_courses",
    )

    count = vectorstore._collection.count()
    print(f"\nDone. {count} vectors stored in ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
