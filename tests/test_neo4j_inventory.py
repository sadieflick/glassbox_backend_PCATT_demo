"""
Quick AuraDB inventory — run once to confirm what's loaded.

  venv/bin/python test_neo4j_inventory.py

Reads credentials from .env — never prints them.
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

            print("=== NODE COUNTS ===")
            for r in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label"):
                print(f"  {r['label']:<20} {r['count']}")

            print("\n=== RELATIONSHIP COUNTS ===")
            for r in s.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY rel_type"):
                print(f"  {r['rel_type']:<25} {r['count']}")

            print("\n=== COURSE NODES (all) ===")
            for r in s.run("MATCH (c:Course) RETURN c ORDER BY c.id"):
                node = dict(r['c'])
                print(f"  id:    {node.get('id', '—')}")
                print(f"  title: {node.get('title', node.get('course_title', '—'))}")
                keys = [k for k in node if k not in ('id', 'title', 'course_title')]
                for k in keys:
                    val = str(node[k])
                    print(f"  {k}: {val[:80]}{'…' if len(val) > 80 else ''}")
                print()

            print("=== COURSE RELATIONSHIP CHECK ===")
            result = s.run("""
                MATCH (c:Course)
                OPTIONAL MATCH (c)-[r]-()
                RETURN c.id AS id, count(r) AS rels
                ORDER BY rels DESC
            """)
            for r in result:
                status = "no relationships" if r['rels'] == 0 else f"{r['rels']} relationship(s)"
                print(f"  {r['id']}: {status}")


if __name__ == "__main__":
    run()
