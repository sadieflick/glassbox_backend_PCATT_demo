"""
Ingest NICE Framework work roles into ChromaDB for Mode B (Vector RAG).

Run this once (after ingest.py) before starting the server:
  venv/bin/python -m app.ingest_nice

What this script does:
  1. Reads nice_framework.csv — one row per knowledge/skill/task element
  2. Groups all elements by work role → 41 documents (instead of 5,334 tiny fragments)
  3. Reads C3_nice_mapping.xlsx — cert provider alignment per work role
  4. Merges cert data into each work role document
  5. Embeds and stores in a separate 'nice_work_roles' collection in the same chroma_db

Two collections, same disk location:
  pcatt_courses   — 16 PCATT course documents (from ingest.py)
  nice_work_roles — 41 NICE work role documents (this script)

Mode B queries both and merges results.
"""

import csv
from pathlib import Path
from collections import defaultdict

import openpyxl
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

NICE_CSV    = Path(__file__).parent.parent.parent.parent / "data" / "nice_framework.csv"
C3_XLSX     = Path(__file__).parent.parent.parent.parent / "data" / "raw_data" / "C3_nice_mapping_April_15_2025.xlsx"
CHROMA_DIR  = Path(__file__).parent.parent / "chroma_db"
COLLECTION  = "nice_work_roles"

# Cert provider columns in the C3 sheet (by index, 0-based)
CERT_PROVIDERS = {
    3: "CertNexus",
    4: "CompTIA",
    5: "FITSI",
    6: "IAPP",
    7: "(ISC)²",
    8: "ISACA",
    9: "SANS | GIAC",
}


def load_cert_mapping(xlsx_path: Path) -> dict[str, dict[str, str]]:
    """
    Returns {work_role_id: {provider_name: cert_string, ...}}.
    Skips rows where work_role_id is missing or not in expected format.
    """
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    ws = wb["Work Roles (Update v.1.0.0"]

    mapping: dict[str, dict[str, str]] = {}
    header_found = False

    for row in ws.iter_rows(values_only=True):
        # The header row contains 'Work Role ID' in column index 2
        if not header_found:
            if row[2] and "Work Role ID" in str(row[2]):
                header_found = True
            continue

        role_id = row[2]
        if not role_id or not str(role_id).strip():
            continue

        role_id = str(role_id).strip()
        certs: dict[str, str] = {}
        for col_idx, provider in CERT_PROVIDERS.items():
            val = row[col_idx]
            if val and str(val).strip() and str(val).strip().upper() != "N/A":
                # Normalize newlines to commas for readability in the document
                certs[provider] = str(val).replace("\n", ", ").strip()

        mapping[role_id] = certs

    return mapping


# nomic-embed-text via Ollama has a 2048 token context window.
# At ~4 chars/token, that's ~8000 chars. We target 7000 to stay safe.
# Strategy: always keep header + certs, then fill remaining space with
# knowledge > skills > tasks (in order of semantic value for retrieval).
_MAX_CHARS = 7000


def _truncate_list(items: list[str], budget: int) -> list[str]:
    """Keep items from the front until we'd exceed `budget` characters."""
    kept, total = [], 0
    for item in items:
        cost = len(item) + 6  # 6 chars for "  - \n"
        if total + cost > budget:
            break
        kept.append(item)
        total += cost
    return kept


def build_documents(nice_csv: Path, cert_mapping: dict) -> list[Document]:
    """
    Group NICE CSV rows by work role and build one Document per role.

    Embedding tiny atomic fragments ("Knowledge of encryption algorithms") one-by-one
    would make similarity search nearly useless — each fragment is too short and generic.
    Grouping everything about a work role into one document lets the embedding capture
    the full semantic profile of that career path.
    """
    # Accumulate elements per work role
    roles: dict[str, dict] = {}

    with open(nice_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row["work_role_id"].strip()
            if rid not in roles:
                roles[rid] = {
                    "id":          rid,
                    "title":       row["work_role_title"].strip(),
                    "category_id": row["category_id"].strip(),
                    "category":    row["category_title"].strip(),
                    "description": row["work_role_description"].strip(),
                    "knowledge":   [],
                    "skills":      [],
                    "tasks":       [],
                }
            etype = row["element_type"].strip().lower()
            text  = row["element_text"].strip()
            if etype == "knowledge":
                roles[rid]["knowledge"].append(text)
            elif etype == "skill":
                roles[rid]["skills"].append(text)
            elif etype == "task":
                roles[rid]["tasks"].append(text)

    docs = []
    for rid, r in roles.items():
        # Build the cert block from C3 mapping
        certs = cert_mapping.get(rid, {})
        if certs:
            cert_lines = "\n".join(f"  {provider}: {names}" for provider, names in certs.items())
            cert_block = f"Aligned Industry Certifications:\n{cert_lines}"
        else:
            cert_block = "Aligned Industry Certifications: (none mapped)"

        header = "\n\n".join([
            f"Work Role: {r['title']} ({rid})",
            f"Category: {r['category']} ({r['category_id']})",
            f"Description: {r['description']}",
            cert_block,
        ])

        # Budget remaining chars for K/S/T after the fixed header
        remaining = _MAX_CHARS - len(header) - 10  # 10 chars padding

        # Allocate: knowledge gets 50%, skills 30%, tasks 20%
        k_items = _truncate_list(r["knowledge"], int(remaining * 0.50))
        s_items = _truncate_list(r["skills"],    int(remaining * 0.30))
        t_items = _truncate_list(r["tasks"],     int(remaining * 0.20))

        k_block = "Knowledge:\n" + "\n".join(f"  - {k}" for k in k_items) if k_items else ""
        s_block = "Skills:\n"    + "\n".join(f"  - {s}" for s in s_items) if s_items else ""
        t_block = "Tasks:\n"     + "\n".join(f"  - {t}" for t in t_items) if t_items else ""

        sections = [header, k_block, s_block, t_block]
        page_content = "\n\n".join(s for s in sections if s)

        # Flatten all certs into a single string for the metadata field
        all_certs = ", ".join(v for v in certs.values()) if certs else ""

        docs.append(Document(
            page_content=page_content,
            metadata={
                "work_role_id": rid,
                "title":        r["title"],
                "category":     r["category"],
                "certifications": all_certs,
            },
        ))

    return docs


def main():
    print(f"Loading NICE framework from: {NICE_CSV}")
    print(f"Loading C3 cert mapping from: {C3_XLSX}")

    cert_mapping = load_cert_mapping(C3_XLSX)
    print(f"  Loaded cert mappings for {len(cert_mapping)} work roles")

    docs = build_documents(NICE_CSV, cert_mapping)
    print(f"  Built {len(docs)} work role documents")

    print("Connecting to Ollama embedding model (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Embedding and storing to: {CHROMA_DIR}  (collection: {COLLECTION})")
    print("  This will take a minute or two...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )

    count = vectorstore._collection.count()
    print(f"\nDone. {count} work role vectors stored in ChromaDB.")
    print(f"Collections now on disk: pcatt_courses + {COLLECTION}")


if __name__ == "__main__":
    main()
