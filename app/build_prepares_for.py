"""
Block 1b — Create PREPARES_FOR edges: Course → Certification

Scans each PCATT course title + description for certification name mentions and
writes a deterministic PREPARES_FOR relationship to the matching Certification node.

This is the "keyword / NER pass" described in the FEA methodology:
confidence = 1.0 because the course literally names the cert as its objective.
LLM involvement: none. Pure text matching.

Run from the backend directory:
  venv/bin/python -m app.build_prepares_for

Requires: Course nodes must already have an id property (run fix_course_ids.py first).
"""

import csv
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
AUTH     = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
DATABASE = os.getenv("NEO4J_DATABASE")

COURSES_CSV = Path(__file__).parent.parent.parent.parent / "data" / "pcatt_courses.csv"

# Map: regex pattern (searched in title + desc) → Certification node id in AuraDB
# Order matters for overlapping patterns — more specific first.
CERT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # CompTIA — specific variants first to avoid A+ matching inside CASP+/CySA+
    (re.compile(r"\bCASP\+|\bSecurityX\+", re.I),          "comptia:SecurityX+"),
    (re.compile(r"\bCySA\+",               re.I),          "comptia:CySA+"),
    (re.compile(r"\bPenTest\+",            re.I),          "comptia:PenTest+"),
    (re.compile(r"\bSecurity\+",           re.I),          "comptia:Security+"),
    (re.compile(r"\bNetwork\+",            re.I),          "comptia:Network+"),
    (re.compile(r"\bTech\+",               re.I),          "comptia:Tech+"),
    (re.compile(r"\bCloud\+",              re.I),          "comptia:Cloud+"),
    (re.compile(r"\bLinux\+",              re.I),          "comptia:Linux+"),
    # A+ needs word boundary on both sides — avoids matching inside "CASP+" etc.
    (re.compile(r"\bA\+\b",               re.I),          "comptia:A+"),
    # GIAC / SANS
    (re.compile(r"\bGPYC\b|GIAC Python",  re.I),          "giac:GPYC"),
]


def scan_course(title: str, desc: str, metadata: str) -> list[str]:
    """Return list of Certification node ids matched in this course's text."""
    text = f"{title} {desc} {metadata}"
    matched = []
    seen = set()
    for pattern, cert_id in CERT_PATTERNS:
        if cert_id not in seen and pattern.search(text):
            matched.append(cert_id)
            seen.add(cert_id)
    return matched


def run():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Connected to AuraDB\n")

        with open(COURSES_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        created = 0
        skipped = 0

        with driver.session(database=DATABASE) as s:
            for row in rows:
                course_id = f"{row['course_prefix']}-{row['course_number']}"
                title     = row.get("course_title", "").strip()
                desc      = row.get("course_desc", "").strip()
                meta      = row.get("metadata", "").strip()

                cert_ids = scan_course(title, desc, meta)

                for cert_id in cert_ids:
                    result = s.run("""
                        MATCH (c:Course {id: $course_id})
                        MATCH (cert:Certification {id: $cert_id})
                        MERGE (c)-[r:PREPARES_FOR]->(cert)
                          ON CREATE SET r.confidence = 1.0,
                                        r.method     = 'keyword'
                        RETURN c.id AS course, cert.acronym AS cert,
                               cert.full_name AS full_name
                    """, course_id=course_id, cert_id=cert_id)

                    record = result.single()
                    if record:
                        print(f"  ✓  {record['course']:<15} → {record['cert']:<12}  {record['full_name']}")
                        created += 1
                    else:
                        print(f"  ✗  {course_id} → {cert_id}  (node not found — check IDs)")
                        skipped += 1

        print(f"\nDone. {created} PREPARES_FOR edge(s) created, {skipped} skipped.")


if __name__ == "__main__":
    run()
