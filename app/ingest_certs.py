"""
Ingest NICE/C3 certifications into ChromaDB for Mode C entity search.

Run once (after ingest.py and ingest_nice.py):
  venv/bin/python -m app.ingest_certs

What this script does:
  1. Fetches all 132 Certification nodes from AuraDB
  2. For each cert, enriches with: work roles that recommend it + PCATT courses that prepare for it
  3. Builds one Document per cert — enough semantic context for fuzzy matching
  4. Embeds and stores in a new 'certifications' collection in chroma_db

Three collections after this runs:
  pcatt_courses   — 16 PCATT course documents
  nice_work_roles — 41 NICE work role documents
  certifications  — 132 certification documents  ← this script
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase

load_dotenv()

CHROMA_DIR  = Path(__file__).parent.parent / "chroma_db"
COLLECTION  = "certifications"

URI      = os.getenv("NEO4J_URI")
AUTH     = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
DATABASE = os.getenv("NEO4J_DATABASE")

# Certs with no work role or course connections are just acronym + name —
# still worth embedding so Q5 can surface them with an honest "no PCATT prep" answer.
ROLE_CAP = 12  # max work roles to list per cert (keeps doc size reasonable)


def fetch_cert_data() -> list[dict]:
    """
    Pull every Certification node enriched with its graph neighborhood:
      - Work roles that recommend it (via RECOMMENDS)
      - PCATT courses that prepare for it (via PREPARES_FOR)
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=DATABASE) as s:
            rows = list(s.run("""
                MATCH (cert:Certification)
                OPTIONAL MATCH (w:WorkRole)-[:RECOMMENDS]->(cert)
                OPTIONAL MATCH (c:Course)-[:PREPARES_FOR]->(cert)
                RETURN
                    cert.id        AS cert_id,
                    cert.acronym   AS acronym,
                    cert.full_name AS full_name,
                    collect(DISTINCT {id: w.id, title: w.title, category: w.category}) AS roles,
                    collect(DISTINCT {id: c.id, title: c.course_title}) AS courses
                ORDER BY cert.acronym
            """))
    return [dict(r) for r in rows]


def build_document(cert: dict) -> Document:
    """
    Build a cert-centric document whose embedding stays close to the cert's identity.

    The failure mode to avoid: including many work role titles makes certs with
    overlapping recommenders look identical to the embedding model. Instead, the
    page_content repeats and amplifies the cert name/acronym so vector search
    reliably discriminates between e.g. Security+ vs CISSP vs Network+.

    Relational context (role count, categories, course names) is stored in metadata
    only — it's returned with search results but does not dilute the embedding.
    """
    acronym   = cert["acronym"]   or cert["cert_id"] or "Unknown"
    full_name = cert["full_name"] or acronym

    roles   = [r for r in cert["roles"]   if r.get("id")]
    courses = [c for c in cert["courses"] if c.get("id")]

    # Derive a rough domain label from the cert acronym/name for additional signal
    name_lower = (acronym + " " + full_name).lower()
    if any(t in name_lower for t in ("pentest", "pen test", "gpen", "gwapt", "gxpn", "offensive")):
        domain = "penetration testing, ethical hacking, offensive security"
    elif any(t in name_lower for t in ("cissp", "cism", "cgeit", "crisc", "cisa", "audit", "governance")):
        domain = "advanced security, governance, risk management, compliance"
    elif any(t in name_lower for t in ("cysa", "gcih", "gcia", "gmon", "soc", "defense", "analyst")):
        domain = "security operations, cyber defense, incident response, threat analysis"
    elif any(t in name_lower for t in ("cloud", "gcld", "ccsp")):
        domain = "cloud security, cloud computing, cloud infrastructure"
    elif any(t in name_lower for t in ("network+", "ccna", "gnfa", "network")):
        domain = "networking, network administration, network infrastructure"
    elif any(t in name_lower for t in ("security+", "gsec", "sscp", "cfr")):
        domain = "security fundamentals, entry-level cybersecurity, security baseline"
    elif any(t in name_lower for t in ("linux", "server+", "a+", "tech+", "it support", "helpdesk")):
        domain = "IT support, systems administration, hardware, technical fundamentals"
    elif any(t in name_lower for t in ("data", "python", "gpyc", "gmle")):
        domain = "data science, programming, automation, scripting"
    elif any(t in name_lower for t in ("privacy", "iapp", "cipm", "cipp")):
        domain = "privacy, data protection, compliance, regulatory"
    elif any(t in name_lower for t in ("forensic", "gcfa", "gcfe", "gcfr", "gbfa")):
        domain = "digital forensics, incident investigation, evidence analysis"
    else:
        domain = "cybersecurity, information security"

    # PCATT course prep line — useful for Q5 signal
    if courses:
        prep_line = "PCATT courses: " + ", ".join(c["title"] for c in courses if c.get("title"))
    else:
        prep_line = "PCATT courses: no direct preparation course in PCATT catalog"

    role_count  = len(roles)
    role_cats   = sorted({r.get("category", "") for r in roles if r.get("category")})
    cat_str     = ", ".join(role_cats) if role_cats else "various"

    # Page content: cert identity emphasized, domain keywords, minimal relational noise
    page_content = (
        f"Certification: {acronym}\n"
        f"Full name: {full_name}\n"
        f"Domain: {domain}\n"
        f"Recognized in {role_count} NICE work role{'s' if role_count != 1 else ''}"
        + (f" across {cat_str}" if role_cats else "") + ".\n"
        f"{prep_line}"
    )

    # Metadata — relational detail for Cypher lookups, not embedded
    role_titles   = ", ".join(r["title"] for r in roles[:ROLE_CAP] if r.get("title"))
    course_titles = ", ".join(c["title"] for c in courses if c.get("title"))

    return Document(
        page_content=page_content,
        metadata={
            "cert_id":        cert["cert_id"] or "",
            "acronym":        acronym,
            "full_name":      full_name,
            "role_titles":    role_titles,
            "course_titles":  course_titles,
            "has_pcatt_prep": str(bool(courses)),
        },
    )


def main():
    print(f"Connecting to AuraDB at {URI}...")
    certs = fetch_cert_data()
    print(f"  Fetched {len(certs)} certifications")

    docs = [build_document(c) for c in certs]

    with_roles   = sum(1 for c in certs if any(r.get("id") for r in c["roles"]))
    with_courses = sum(1 for c in certs if any(r.get("id") for r in c["courses"]))
    print(f"  {with_roles} certs have work role connections")
    print(f"  {with_courses} certs have PCATT course prep connections")

    print("\nConnecting to Ollama embedding model (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Embedding and storing to: {CHROMA_DIR}  (collection: {COLLECTION})")
    print("  Embedding 132 documents — should take about 2–3 minutes...")

    # Delete existing collection first so re-runs are idempotent
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
        print("  (Deleted existing collection for clean re-run)")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )

    count = vectorstore._collection.count()
    print(f"\nDone. {count} certification vectors stored.")
    print("Collections now on disk: pcatt_courses + nice_work_roles + certifications")


if __name__ == "__main__":
    main()
