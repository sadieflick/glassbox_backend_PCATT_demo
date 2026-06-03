"""
Block 1c — Create ALIGNS_TO edges: Course → WorkRole

Uses stored ChromaDB embeddings (from ingest.py + ingest_nice.py) to compute
cosine similarity between each PCATT course and each NICE work role. Creates
ALIGNS_TO relationships in AuraDB for pairs above the similarity threshold.

This is the "embedding similarity pass" in the FEA methodology:
  - No LLM involved — pure cosine math on existing vectors
  - Same math Mode B uses at query time, applied at graph-build time
  - method="embedding" on the relationship makes the provenance auditable

Run from the backend directory:
  venv/bin/python -m app.build_aligns_to            # preview only (dry run)
  venv/bin/python -m app.build_aligns_to --write    # write edges to AuraDB

Requires:
  - ChromaDB built (ingest.py + ingest_nice.py already run)
  - Course nodes have id property (fix_course_ids.py already run)
"""

import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
AUTH     = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
DATABASE = os.getenv("NEO4J_DATABASE")

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Tuning knobs
SIMILARITY_THRESHOLD = 65.0   # minimum similarity % to create an edge
TOP_K_PER_COURSE     = 4      # max work roles per course

# Work roles excluded from ALIGNS_TO.
# OG-WRL-005 (Cybersecurity Instruction) and OG-WRL-004 (Cybersecurity Curriculum
# Development) have broad, comprehensive TKS descriptions because instructors need
# to know everything — this makes them attract every course above threshold.
# They represent roles for experienced teachers, not entry-level student pathways.
EXCLUDED_ROLE_IDS = {"OG-WRL-005", "OG-WRL-004"}
SKIP_COURSES = {"Com-2036", "Com-2124", "Com-2179"}


def distance_to_pct(d: float) -> float:
    return round(max(0.0, min(100.0, (1.0 - d / 2.0) * 100)), 1)


def build_edges(dry_run: bool):
    client      = chromadb.PersistentClient(path=str(CHROMA_DIR))
    courses_col = client.get_collection("pcatt_courses")
    roles_col   = client.get_collection("nice_work_roles")

    # Pull all course embeddings + metadata in one call — no re-embedding needed
    courses_data = courses_col.get(include=["embeddings", "metadatas"])
    n_courses    = len(courses_data["ids"])
    print(f"Loaded {n_courses} course embeddings from ChromaDB")

    n_roles = roles_col.count()
    print(f"Loaded {n_roles} work role embeddings from ChromaDB")
    print(f"Threshold: {SIMILARITY_THRESHOLD}% similarity, top {TOP_K_PER_COURSE} per course\n")

    edges: list[dict] = []

    for i in range(n_courses):
        meta      = courses_data["metadatas"][i]
        embedding = courses_data["embeddings"][i]
        course_id = meta.get("course_id", "unknown")
        title     = meta.get("title", "")

        if course_id in SKIP_COURSES:
            print(f"  ⊘  {course_id:<15} skipped (cert crosswalk coverage)")
            continue

        results = roles_col.query(
            query_embeddings=[embedding],
            n_results=TOP_K_PER_COURSE,
            include=["metadatas", "distances"],
        )

        matches = []
        for meta_r, dist in zip(results["metadatas"][0], results["distances"][0]):
            pct     = distance_to_pct(dist)
            role_id = meta_r.get("work_role_id", "")
            if role_id in EXCLUDED_ROLE_IDS:
                continue
            if pct >= SIMILARITY_THRESHOLD:
                matches.append({
                    "course_id":    course_id,
                    "course_title": title,
                    "role_id":      role_id,
                    "role_title":   meta_r.get("title", ""),
                    "similarity":   pct,
                    "distance":     round(dist, 4),
                })

        if matches:
            for m in matches:
                flag = "  ✓" if not dry_run else "  →"
                print(f"{flag} {m['course_id']:<15} ({m['similarity']}%)  →  {m['role_id']}  {m['role_title']}")
            edges.extend(matches)
        else:
            print(f"  –  {course_id:<15} no matches above threshold")

    print(f"\n{'Would create' if dry_run else 'Creating'} {len(edges)} ALIGNS_TO edge(s)...\n")

    if dry_run:
        print("Dry run — pass --write to commit to AuraDB.")
        return

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        created = 0
        skipped = 0

        with driver.session(database=DATABASE) as s:
            for e in edges:
                result = s.run("""
                    MATCH (c:Course   {id: $course_id})
                    MATCH (w:WorkRole {id: $role_id})
                    MERGE (c)-[r:ALIGNS_TO]->(w)
                      ON CREATE SET r.similarity = $similarity,
                                    r.distance   = $distance,
                                    r.method     = 'embedding'
                    RETURN c.id AS course, w.id AS role
                """, course_id=e["course_id"], role_id=e["role_id"],
                     similarity=e["similarity"], distance=e["distance"])

                if result.single():
                    created += 1
                else:
                    print(f"  ✗  {e['course_id']} → {e['role_id']}  (node not found)")
                    skipped += 1

        print(f"Done. {created} ALIGNS_TO edge(s) written, {skipped} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write edges to AuraDB (default: dry run / preview only)")
    args = parser.parse_args()
    build_edges(dry_run=not args.write)
