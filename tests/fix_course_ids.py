"""
One-time fix: set id = course_prefix + '-' + course_number on all Course nodes.

  venv/bin/python tests/fix_course_ids.py

Required before build_prepares_for.py or build_aligns_to.py — those scripts
use MERGE on Course.id, which will silently create duplicate nodes if id is missing.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
AUTH     = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
DATABASE = os.getenv("NEO4J_DATABASE")


def run():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Connected to AuraDB\n")

        with driver.session(database=DATABASE) as s:
            result = s.run("""
                MATCH (c:Course)
                WHERE c.course_prefix IS NOT NULL AND c.course_number IS NOT NULL
                SET c.id = c.course_prefix + '-' + toString(c.course_number)
                RETURN c.id AS id, c.title AS title
                ORDER BY c.id
            """)
            rows = list(result)

        if not rows:
            print("No Course nodes found — check AuraDB connection or node labels.")
            return

        print(f"Set id on {len(rows)} Course nodes:\n")
        for r in rows:
            print(f"  {r['id']:<15}  {r['title']}")

        print("\nDone. Course nodes now have an id property.")


if __name__ == "__main__":
    run()
