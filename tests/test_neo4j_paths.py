"""
Test traversal queries — confirm the NICE/C3 spine is intact and explorable.

  venv/bin/python test_neo4j_paths.py

Run AFTER test_neo4j_inventory.py confirms the DB is connected.
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
        with driver.session(database=DATABASE) as s:

            print("=== ALL WORK ROLES ===")
            for r in s.run("""
                MATCH (cat:Category)-[:HAS_ROLE]->(w:WorkRole)
                RETURN cat.title AS category, w.id AS id, w.title AS title
                ORDER BY cat.title, w.title
            """):
                print(f"  [{r['category']}]  {r['id']}  {r['title']}")

            print("\n=== SAMPLE PATH: WorkRole → Certifications ===")
            for r in s.run("""
                MATCH (w:WorkRole)-[:RECOMMENDS]->(c:Certification)
                RETURN w.title AS role, collect(c.acronym) AS certs
                ORDER BY w.title LIMIT 8
            """):
                print(f"  {r['role']}")
                print(f"    certs: {', '.join(r['certs'])}")

            print("\n=== SAMPLE PATH: Certification → WorkRoles (CISSP) ===")
            for r in s.run("""
                MATCH (w:WorkRole)-[:RECOMMENDS]->(c:Certification {acronym: 'CISSP'})
                RETURN w.title AS role, w.id AS id ORDER BY w.title
            """):
                print(f"  {r['id']}  {r['role']}")

            print("\n=== SAMPLE PATH: SANS Course → Cert → WorkRole ===")
            for r in s.run("""
                MATCH (s:SANSCourse)-[:AWARDS]->(cert:Certification)<-[:RECOMMENDS]-(w:WorkRole)
                RETURN s.code AS code, s.title AS course, cert.acronym AS cert, w.title AS role
                LIMIT 8
            """):
                print(f"  {r['code']} → {r['cert']} → {r['role']}")


if __name__ == "__main__":
    run()
