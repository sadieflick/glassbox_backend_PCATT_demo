"""
Mode C: GraphRAG — vector anchor + semantic routing + graph traversal + LLM synthesis.

Flow:
  1. Embed question → ChromaDB (nice_work_roles + pcatt_courses + certifications)
  2. LLM router: classify question intent → pick 1 of 8 query templates (closed set)
  3. Emit {"type":"route", template, label, reason} SSE event  ← decision trace
  4. Execute selected Cypher template with verified entity IDs  ← deterministic
  5. Emit {"type":"path", nodes, edges, roles} SSE event       ← graph lights up
  6. LLM synthesizes answer strictly from retrieved subgraph context
  7. Stream tokens → [DONE]

Templates:
  Q1 role_anchor    — WorkRole → aligned Courses + recommended Certs
  Q2 course_path    — Course → WorkRoles it leads to + Certs it prepares
  Q3 cert_bridge    — WorkRole → Certs recommended + Courses that prepare each cert
  Q4 role_compare   — Two WorkRoles side-by-side (Q1 × 2, structured as comparison)
  Q5 cert_prep      — Cert → PCATT Courses that prepare for it + Roles that need it
  Q6 advanced_path  — Course → Roles → next-level Certs not yet covered → further Courses
  Q7 broad_domain   — High-recall: top 4 WorkRoles + all aligned Courses (catalog view)
  Q8 course_compare — Two Courses side-by-side (Q2 × 2, structured as comparison)
"""
import json
import os
from pathlib import Path
from typing import AsyncGenerator

import chromadb
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j import GraphDatabase
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

CHROMA_DIR = Path(__file__).parent.parent.parent / "chroma_db"

embeddings  = OllamaEmbeddings(model="nomic-embed-text")
llm         = ChatOllama(model="llama3.2:latest", temperature=0.3, num_predict=500)
llm_router  = ChatOllama(model="llama3.2:latest", temperature=0, num_predict=80, format="json")

URI      = os.getenv("NEO4J_URI")
AUTH     = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
DATABASE = os.getenv("NEO4J_DATABASE")

CERT_CAP        = 10
COURSE_CAP      = 14
BROAD_ROLE_CAP  = 6   # Q7 broad catalog view

TEMPLATE_META = {
    "Q1": {"label": "role_anchor",    "desc": "Role-centric: courses + certs to reach this career"},
    "Q2": {"label": "course_path",    "desc": "Course-centric: where this course leads"},
    "Q3": {"label": "cert_bridge",    "desc": "Cert landscape: which certs connect roles to courses"},
    "Q4": {"label": "role_landscape",  "desc": "Domain view: all roles in the relevant career domain"},
    "Q5": {"label": "cert_prep",      "desc": "Cert-prep: PCATT courses that prepare for a specific cert"},
    "Q6": {"label": "advanced_path",  "desc": "Forward chain: what comes next after this course"},
    "Q7": {"label": "broad_domain",   "desc": "Catalog view: all courses across a whole domain"},
    "Q8": {"label": "course_compare", "desc": "Side-by-side: two courses compared"},
}

ROUTING_PROMPT = """\
You are a query router for a career advisor system about IT and cybersecurity careers in Hawaii.
Given a user question and the entities found by vector search, output JSON with exactly two fields:
  "template": one of "Q1" through "Q8"
  "reason": one sentence explaining the choice (max 20 words)

Templates:
Q1 role_anchor    — user wants to reach a specific work role; needs courses + certs to get there
Q2 course_path    — user wants to know where a specific course leads (roles, certs)
Q3 cert_bridge    — user asks WHICH certifications are needed/recommended for a career area
Q4 role_landscape — user is exploring or comparing career domains; show all roles in those domains
Q5 cert_prep      — user names a specific cert and asks WHICH COURSE prepares for it
Q6 advanced_path  — user finished or is past a specific course and wants what comes NEXT
Q7 broad_domain   — user asks broadly what courses exist in a domain with no specific cert/role target
Q8 course_compare — user is comparing two specific courses side by side

PRIORITY RULES (apply before looking at entities):
- If the question names a specific certification (Security+, CISSP, CySA+, etc.) AND asks which course/training prepares for it → Q5
- If the question asks "what certifications" or "which certs" for a career/job → Q3 (NOT Q7)
- If the question asks "what courses do you offer" or "show me everything" with no specific cert → Q7
- Q7 is a last resort for genuinely open catalog browsing — most cert or role questions fit Q1/Q3/Q5

Examples:

Q: "I want to become a cybersecurity analyst. Where do I start?"
Entities: WorkRole: Cyber Defense Analyst (91%), WorkRole: Vulnerability Assessment Analyst (82%)
-> {{"template": "Q1", "reason": "User wants to reach a specific role; needs courses and certs to get there."}}

Q: "I just finished Com-2036. What happens after I take it?"
Entities: Course: CompTIA Security+ Prep (94%), WorkRole: Cyber Defense Analyst (71%)
-> {{"template": "Q2", "reason": "User asking where a specific course leads; course is the anchor."}}

Q: "What careers are there in networking and what certs matter?"
Entities: WorkRole: Network Operations Specialist (85%), WorkRole: Systems Administrator (79%)
-> {{"template": "Q3", "reason": "Question explicitly asks about certifications across career paths."}}

Q: "Should I aim for cyber defense or network operations as a career?"
Entities: WorkRole: Cyber Defense Analyst (88%), WorkRole: Network Operations Specialist (84%)
-> {{"template": "Q4", "reason": "User weighing two career domains; domain landscape shows all relevant roles."}}

Q: "Which course in your catalog prepares me for the Security+ exam?"
Entities: Certification: Security+ (82%), Course: CompTIA Security+ Prep (78%)
-> {{"template": "Q5", "reason": "User asking which course prepares for a specific named cert."}}

Q: "I already finished the Security+ prep. What advanced courses or certs should I pursue next?"
Entities: Course: CompTIA Security+ Prep (91%), Certification: Security+ (74%)
-> {{"template": "Q6", "reason": "User completed a course and wants the path beyond what it already covers."}}

Q: "What IT and cybersecurity courses do you offer in your catalog overall?"
Entities: WorkRole: Cyber Defense Analyst (80%), WorkRole: Vulnerability Assessment Analyst (77%), Course: CompTIA Security+ Prep (71%)
-> {{"template": "Q7", "reason": "User wants a broad catalog overview; no specific role or cert target."}}

Q: "What is the difference between the Security+ prep course and the CySA+ prep course?"
Entities: Course: CompTIA Security+ Prep (90%), Course: CompTIA CySA+ Prep (87%)
-> {{"template": "Q8", "reason": "User explicitly comparing two specific courses side by side."}}

Q: "What certifications do I need for a cybersecurity job?"
Entities: WorkRole: Vulnerability Assessment Analyst (87%), WorkRole: Cyber Defense Analyst (85%)
-> {{"template": "Q3", "reason": "User asking which certs are needed for a career area — cert landscape question."}}

Q: "Which certs matter most for networking careers?"
Entities: WorkRole: Network Operations Specialist (86%), WorkRole: Systems Administrator (80%)
-> {{"template": "Q3", "reason": "Explicit cert-centric question for a career area — Q3 not Q7."}}

Q: "Do you have a prep course for the Security+ exam?"
Entities: WorkRole: Cyber Defense Analyst (78%), Course: CompTIA Security+ Prep (75%), Certification: Security+ (71%)
-> {{"template": "Q5", "reason": "User names Security+ and asks which course prepares for it."}}

Q: "What courses help me get into IT security in Hawaii?"
Entities: WorkRole: Cyber Defense Analyst (89%), WorkRole: Information Systems Security Officer (80%)
-> {{"template": "Q1", "reason": "Asking for courses to reach a role; role anchor is correct."}}

Q: "What is a realistic path to a cybersecurity job in Hawaii?"
Entities: WorkRole: Cyber Defense Analyst (87%), WorkRole: Vulnerability Assessment Analyst (82%)
-> {{"template": "Q1", "reason": "Career path question; user wants courses and steps to reach a specific role."}}

---
Q: "{question}"
Entities: {entity_summary}

Output ONLY valid JSON with "template" and "reason" fields:\
"""

SYSTEM_PROMPT = """\
You are a helpful career advisor for students in Hawaii interested in IT and cybersecurity careers.
You have been given VERIFIED graph data from the PCATT course catalog and the NICE Cybersecurity Workforce Framework.

Answer the user's question using ONLY this verified data. Be specific — name the actual courses and work roles.
If the data doesn't fully answer the question, say so honestly. Do not invent courses, roles, or certifications.\
"""


class QueryRequest(BaseModel):
    question: str


# ─── Entity search ────────────────────────────────────────────────────────────

def _query_entities(question: str) -> dict:
    """
    Search all three ChromaDB collections in one embedding pass.
    Returns top candidates per entity type for the router and templates.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    q_vec  = embeddings.embed_query(question)

    def _search(col_name: str, id_field: str, label: str,
                n: int, extra_fields: list[str] | None = None) -> list[dict]:
        try:
            col     = client.get_collection(col_name)
            results = col.query(query_embeddings=[q_vec], n_results=n,
                                include=["metadatas", "distances"])
        except Exception:
            return []
        out = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            sim = round(max(0.0, min(100.0, (1.0 - dist / 2.0) * 100)), 1)
            eid = meta.get(id_field, "")
            if eid:
                entry = {
                    "id":         eid,
                    "title":      meta.get("title", eid),
                    "label":      label,
                    "category":   meta.get("category", ""),
                    "similarity": sim,
                }
                for f in (extra_fields or []):
                    entry[f] = meta.get(f, "")
                out.append(entry)
        return out

    roles   = _search("nice_work_roles", "work_role_id", "WorkRole",      n=8)
    courses = _search("pcatt_courses",   "course_id",    "Course",        n=6)
    certs   = _search("certifications",  "cert_id",      "Certification", n=5,
                      extra_fields=["acronym", "full_name", "has_pcatt_prep"])
    return {"roles": roles, "courses": courses, "certs": certs}


# ─── Router ───────────────────────────────────────────────────────────────────

_CERT_NAMES = {
    "security+", "cissp", "cysa+", "network+", "pentest+", "securityx+", "securityx",
    "tech+", "a+", "linux+", "cloud+", "server+", "ccna", "ccnp", "cism", "cisa",
    "crisc", "cgeit", "sscp", "gsec", "gpen", "gcia", "gcih", "gcfa",
}
_COMPARE_SIGNALS    = {"compare", "difference between", "vs ", "versus", "which is better",
                       "what's the difference", "what is the difference"}
_ADV_PATH_SIGNALS   = {"what's next after", "what comes next after", "what should i do next",
                       "what to do next", "next step after", "what after", "where do i go from",
                       "already have", "already finished", "already completed", "already passed",
                       "now that i finished", "now that i have", "now that i passed",
                       "i already have", "i finished", "i completed", "i passed",
                       "what comes after", "after finishing", "after completing",
                       "if i complete", "if i finish", "once i complete", "once i finish",
                       "after i complete", "after i finish", "when i complete", "when i finish",
                       "what advanced"}
_CERT_PREP_SIGNALS  = {"prep", "prepare", "pass the", "study for", "which course",
                       "what course", "which class", "what class", "get ready for"}
_CERT_QUERY_SIGNALS = {"what cert", "which cert", "what certifications", "which certifications",
                       "certs do i", "certs are", "certifications do i", "certifications are",
                       "certs needed", "cert do i", "cert should"}


async def _route(question: str, entities: dict) -> tuple[str, str]:
    """
    Route to one of Q1–Q8. Applies keyword pre-filters in priority order for
    high-confidence cases, then falls back to the LLM classifier.

    Pre-filter order matters: Q8/Q6 must run before Q5 because comparison and
    completion phrases also contain cert names and prep-related words.
    """
    q_lower = question.lower()

    # Pre-filter Q8: explicit comparison of two courses (checked before Q5/Q3)
    if any(s in q_lower for s in _COMPARE_SIGNALS):
        return "Q8", "Question contains a comparison signal — course_compare template."

    # Pre-filter Q6: user signals they've completed something and wants the next path
    if any(s in q_lower for s in _ADV_PATH_SIGNALS):
        return "Q6", "Question signals course completion and asks for next steps."

    # Pre-filter Q5: question names a cert + asks how to prepare for it
    if (any(c in q_lower for c in _CERT_NAMES)
            and any(s in q_lower for s in _CERT_PREP_SIGNALS)):
        return "Q5", "Question names a specific certification with a preparation signal."

    # Pre-filter Q3: question explicitly asks which certs are needed/recommended
    if any(s in q_lower for s in _CERT_QUERY_SIGNALS):
        return "Q3", "Question asks which certifications are needed or recommended."

    # LLM classifier for everything else
    parts = []
    for r in entities["roles"]:
        parts.append(f"WorkRole: {r['title']} ({r['similarity']}%)")
    for c in entities["courses"]:
        parts.append(f"Course: {c['title']} ({c['similarity']}%)")
    for cert in entities.get("certs", []):
        parts.append(f"Certification: {cert.get('acronym') or cert['title']} ({cert['similarity']}%)")
    entity_summary = ", ".join(parts) if parts else "no entities found"

    prompt = ROUTING_PROMPT.format(question=question, entity_summary=entity_summary)
    try:
        result = await llm_router.ainvoke([HumanMessage(content=prompt)])
        data   = json.loads(result.content)
        tpl    = str(data.get("template", "Q1")).strip().upper()
        reason = str(data.get("reason", "")).strip()
        if tpl not in TEMPLATE_META:
            tpl = "Q1"
        return tpl, reason
    except Exception:
        return "Q1", "Routing fallback — defaulted to role anchor."


# ─── Cypher template helpers ──────────────────────────────────────────────────

def _role_subgraph(s, role: dict, add_node, add_edge) -> str:
    """
    Q1/Q4/Q7 core: for one work role, pull aligned courses + recommended certs.
    Mutates shared graph state via closures. Returns a context block string.
    """
    role_id = role["id"]
    add_node(role_id, "WorkRole", role["title"], category=role.get("category", ""))

    rec_rows = list(s.run(
        "MATCH (w:WorkRole {id: $rid})-[:RECOMMENDS]->(cert:Certification) "
        "RETURN cert.id AS id, cert.acronym AS acronym, cert.full_name AS full_name "
        "LIMIT $cap",
        rid=role_id, cap=CERT_CAP,
    ))
    course_rows = list(s.run(
        "MATCH (c:Course)-[:ALIGNS_TO]->(w:WorkRole {id: $rid}) "
        "OPTIONAL MATCH (c)-[:PREPARES_FOR]->(prep:Certification) "
        "RETURN c.id AS cid, c.course_title AS title, "
        "       collect(DISTINCT {id: prep.id, acronym: prep.acronym}) AS prep_certs "
        "LIMIT $cap",
        rid=role_id, cap=COURSE_CAP,
    ))

    for r in rec_rows:
        cid = r["id"] or r["acronym"]
        if cid:
            add_node(cid, "Certification", r["full_name"] or r["acronym"])
            add_edge(role_id, cid, "RECOMMENDS")

    for row in course_rows:
        cid = row["cid"]
        if not cid:
            continue
        add_node(cid, "Course", row["title"] or cid)
        add_edge(cid, role_id, "ALIGNS_TO")
        for p in row["prep_certs"]:
            if p.get("acronym"):
                pid = p["id"] or p["acronym"]
                add_node(pid, "Certification", p["acronym"])
                add_edge(cid, pid, "PREPARES_FOR")

    course_names  = [r["title"] or r["cid"] for r in course_rows if r["cid"]]
    prep_acronyms = sorted({p["acronym"] for r in course_rows
                            for p in r["prep_certs"] if p.get("acronym")})
    rec_lines     = [f"{r['acronym']} ({r['full_name']})"
                     for r in rec_rows if r.get("acronym")]

    return (
        f"WORK ROLE: {role['title']} ({role_id}) — {role.get('category', '')}\n"
        "Aligned PCATT courses:\n" +
        ("\n".join(f"  - {n}" for n in course_names) or "  (none)") + "\n"
        "Certifications those courses prepare for:\n" +
        ("\n".join(f"  - {a}" for a in prep_acronyms) or "  (none)") + "\n"
        "Certifications recommended by NICE/C3:\n" +
        ("\n".join(f"  - {l}" for l in rec_lines[:8]) or "  (none)")
    )


def _course_subgraph(s, course: dict, add_node, add_edge) -> str:
    """
    Q2/Q8 core: for one course, pull work roles it leads to + certs it prepares.
    Returns a context block string.
    """
    cid = course["id"]
    add_node(cid, "Course", course["title"])

    rows = list(s.run(
        "MATCH (c:Course {id: $cid}) "
        "OPTIONAL MATCH (c)-[:ALIGNS_TO]->(w:WorkRole) "
        "OPTIONAL MATCH (c)-[:PREPARES_FOR]->(cert:Certification) "
        "RETURN collect(DISTINCT {id: w.id, title: w.title, category: w.category}) AS roles, "
        "       collect(DISTINCT {id: cert.id, acronym: cert.acronym}) AS prep_certs",
        cid=cid,
    ))
    if not rows:
        return f"COURSE: {course['title']} ({cid})\n  (no graph data found)"

    row = rows[0]
    for w in row["roles"]:
        if w.get("id"):
            add_node(w["id"], "WorkRole", w["title"], category=w.get("category", ""))
            add_edge(cid, w["id"], "ALIGNS_TO")
    for cert in row["prep_certs"]:
        pid = cert.get("id") or cert.get("acronym")
        if pid:
            add_node(pid, "Certification", cert.get("acronym", pid))
            add_edge(cid, pid, "PREPARES_FOR")

    role_names = [w["title"] for w in row["roles"] if w.get("title")]
    prep_names = [c["acronym"] for c in row["prep_certs"] if c.get("acronym")]

    return (
        f"COURSE: {course['title']} ({cid})\n"
        "Leads to work roles:\n" +
        ("\n".join(f"  - {n}" for n in role_names) or "  (none)") + "\n"
        "Prepares students for certifications:\n" +
        ("\n".join(f"  - {n}" for n in prep_names) or "  (none)")
    )


def _roles_from_courses(s, courses: list[dict]) -> list[dict]:
    """
    Follow Course -[:ALIGNS_TO]-> WorkRole for the given courses.
    Course descriptions match student questions more reliably than role title
    embeddings do, so the top-matched courses are a better expansion signal
    than NICE category buckets.
    """
    cids = [c["id"] for c in courses if c.get("id")]
    if not cids:
        return []
    rows = list(s.run(
        "UNWIND $cids AS cid "
        "MATCH (c:Course {id: cid})-[:ALIGNS_TO]->(w:WorkRole) "
        "RETURN DISTINCT w.id AS id, w.title AS title, w.category AS category",
        cids=cids,
    ))
    return [
        {"id": r["id"], "title": r["title"] or r["id"], "category": r["category"] or ""}
        for r in rows if r["id"]
    ]


# ─── Traversal dispatcher ─────────────────────────────────────────────────────

def _traverse_graph(template_id: str, entities: dict) -> dict:
    nodes: list[dict]     = []
    edges: list[dict]     = []
    seen_n: set           = set()
    seen_e: set           = set()
    ctx_blocks: list[str] = []

    def add_node(nid: str, label: str, title: str, **props):
        if nid and nid not in seen_n:
            seen_n.add(nid)
            nodes.append({"id": nid, "label": label, "title": title, **props})

    def add_edge(src: str, tgt: str, etype: str):
        key = (src, tgt, etype)
        if key not in seen_e and src and tgt:
            seen_e.add(key)
            edges.append({"source": src, "target": tgt, "type": etype})

    roles   = entities["roles"]
    courses = entities["courses"]
    certs   = entities.get("certs", [])

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=DATABASE) as s:

            # ── Q2: course_path ───────────────────────────────────────────────
            if template_id == "Q2" and courses:
                for course in courses[:2]:
                    ctx_blocks.append(_course_subgraph(s, course, add_node, add_edge))

            # ── Q3: cert_bridge ───────────────────────────────────────────────
            elif template_id == "Q3" and roles:
                seen_ids = {r["id"] for r in roles[:3]}
                q3_roles = list(roles[:3])
                for cr in _roles_from_courses(s, courses[:3]):
                    if cr["id"] not in seen_ids and len(q3_roles) < 5:
                        seen_ids.add(cr["id"])
                        q3_roles.append(cr)
                for role in q3_roles:
                    role_id = role["id"]
                    add_node(role_id, "WorkRole", role["title"],
                             category=role.get("category", ""))
                    cert_rows = list(s.run(
                        "MATCH (w:WorkRole {id: $rid})-[:RECOMMENDS]->(cert:Certification) "
                        "OPTIONAL MATCH (c:Course)-[:PREPARES_FOR]->(cert) "
                        "RETURN cert.id AS cert_id, cert.acronym AS acronym, "
                        "       cert.full_name AS full_name, "
                        "       collect(DISTINCT {id: c.id, title: c.course_title}) AS courses "
                        "LIMIT $cap",
                        rid=role_id, cap=CERT_CAP,
                    ))
                    cert_lines = []
                    for row in cert_rows:
                        cid = row["cert_id"] or row["acronym"]
                        if not cid:
                            continue
                        add_node(cid, "Certification", row["full_name"] or row["acronym"])
                        add_edge(role_id, cid, "RECOMMENDS")
                        for c in row["courses"]:
                            if c.get("id"):
                                add_node(c["id"], "Course", c["title"] or c["id"])
                                add_edge(c["id"], cid, "PREPARES_FOR")
                        course_list = [c["title"] for c in row["courses"] if c.get("title")]
                        cert_lines.append(
                            f"  {row['acronym']} ({row['full_name'] or ''}): "
                            + (", ".join(course_list) if course_list
                               else "no PCATT courses prepare for this cert")
                        )
                    ctx_blocks.append(
                        f"WORK ROLE: {role['title']} ({role_id})\n"
                        "Certifications recommended (and PCATT courses that prepare each):\n" +
                        ("\n".join(cert_lines) or "  (none)")
                    )

            # ── Q4: role_landscape ────────────────────────────────────────────
            elif template_id == "Q4" and roles:
                seen_ids     = {r["id"] for r in roles[:3]}
                domain_roles = list(roles[:3])
                for cr in _roles_from_courses(s, courses[:4]):
                    if cr["id"] not in seen_ids and len(domain_roles) < 8:
                        seen_ids.add(cr["id"])
                        domain_roles.append(cr)
                cats        = list(dict.fromkeys(r.get("category", "") for r in domain_roles if r.get("category")))
                cat_display = ", ".join(cats) if cats else "IT/Cybersecurity"
                ctx_blocks.append(f"DOMAIN LANDSCAPE — career areas: {cat_display}\n")
                for role in domain_roles:
                    ctx_blocks.append(_role_subgraph(s, role, add_node, add_edge))

            # ── Q5: cert_prep ─────────────────────────────────────────────────
            elif template_id == "Q5" and certs:
                cert   = certs[0]
                cert_id   = cert["id"]
                acronym   = cert.get("acronym") or cert_id
                full_name = cert.get("full_name") or acronym

                rows = list(s.run(
                    "MATCH (cert:Certification) "
                    "WHERE cert.id = $cid OR cert.acronym = $acro "
                    "WITH cert LIMIT 1 "
                    "OPTIONAL MATCH (c:Course)-[:PREPARES_FOR]->(cert) "
                    "OPTIONAL MATCH (w:WorkRole)-[:RECOMMENDS]->(cert) "
                    "RETURN cert.id AS cid, cert.acronym AS cacro, cert.full_name AS cfull, "
                    "       collect(DISTINCT {id: c.id, title: c.course_title}) AS courses, "
                    "       collect(DISTINCT {id: w.id, title: w.title}) AS roles",
                    cid=cert_id, acro=acronym,
                ))
                if rows:
                    row       = rows[0]
                    resolved_id = row["cid"] or cert_id
                    add_node(resolved_id, "Certification", row["cfull"] or row["cacro"] or acronym)

                    prep_courses = [c for c in row["courses"] if c.get("id")]
                    rec_roles    = [r for r in row["roles"]   if r.get("id")]

                    for c in prep_courses:
                        add_node(c["id"], "Course", c["title"] or c["id"])
                        add_edge(c["id"], resolved_id, "PREPARES_FOR")
                    for r in rec_roles[:6]:
                        add_node(r["id"], "WorkRole", r["title"])
                        add_edge(r["id"], resolved_id, "RECOMMENDS")

                    course_names = [c["title"] for c in prep_courses if c.get("title")]
                    role_names   = [r["title"] for r in rec_roles   if r.get("title")]

                    ctx_blocks.append(
                        f"CERTIFICATION: {row['cacro'] or acronym} ({row['cfull'] or full_name})\n"
                        "PCATT courses that prepare for this certification:\n" +
                        ("\n".join(f"  - {n}" for n in course_names)
                         if course_names else
                         "  (none — no PCATT course currently prepares for this cert)") + "\n"
                        "Work roles that recommend this certification (per NICE/C3):\n" +
                        ("\n".join(f"  - {n}" for n in role_names[:8]) or "  (none)")
                    )

            # ── Q6: advanced_path ─────────────────────────────────────────────
            elif template_id == "Q6" and courses:
                course = courses[0]
                cid    = course["id"]
                add_node(cid, "Course", course["title"])

                # Step 1: what roles does this course lead to, and what cert does it cover?
                step1 = list(s.run(
                    "MATCH (c:Course {id: $cid}) "
                    "OPTIONAL MATCH (c)-[:ALIGNS_TO]->(w:WorkRole) "
                    "OPTIONAL MATCH (c)-[:PREPARES_FOR]->(covered:Certification) "
                    "RETURN collect(DISTINCT {id: w.id, title: w.title, category: w.category}) AS roles, "
                    "       collect(DISTINCT covered.id) AS covered_ids, "
                    "       collect(DISTINCT covered.acronym) AS covered_acronyms",
                    cid=cid,
                ))
                if not step1:
                    ctx_blocks.append(f"COURSE: {course['title']} ({cid})\n  (no graph data found)")
                else:
                    r1           = step1[0]
                    anchor_roles = [r for r in r1["roles"]   if r.get("id")]
                    covered_ids  = [i for i in r1["covered_ids"] if i]
                    covered_acro = [a for a in r1["covered_acronyms"] if a]

                    for r in anchor_roles:
                        add_node(r["id"], "WorkRole", r["title"], category=r.get("category", ""))
                        add_edge(cid, r["id"], "ALIGNS_TO")

                    # Step 2: per role, find next-level certs not yet covered + courses for them
                    next_ctx_lines = []
                    for role in anchor_roles[:2]:
                        step2 = list(s.run(
                            "MATCH (w:WorkRole {id: $rid})-[:RECOMMENDS]->(cert:Certification) "
                            "WHERE NOT cert.id IN $covered "
                            "OPTIONAL MATCH (next:Course)-[:PREPARES_FOR]->(cert) "
                            "RETURN cert.id AS ncid, cert.acronym AS nacro, cert.full_name AS nfull, "
                            "       collect(DISTINCT {id: next.id, title: next.course_title}) AS next_courses "
                            "LIMIT $cap",
                            rid=role["id"], covered=covered_ids, cap=CERT_CAP,
                        ))
                        for row in step2:
                            ncid = row["ncid"] or row["nacro"]
                            if not ncid:
                                continue
                            add_node(ncid, "Certification", row["nfull"] or row["nacro"])
                            add_edge(role["id"], ncid, "RECOMMENDS")
                            next_courses = [c for c in row["next_courses"] if c.get("id")]
                            for nc in next_courses:
                                add_node(nc["id"], "Course", nc["title"] or nc["id"])
                                add_edge(nc["id"], ncid, "PREPARES_FOR")
                            course_list = [nc["title"] for nc in next_courses if nc.get("title")]
                            next_ctx_lines.append(
                                f"  {row['nacro']} ({row['nfull'] or ''}): "
                                + (", ".join(course_list) if course_list
                                   else "no PCATT courses prepare for this cert yet")
                            )

                    role_names    = [r["title"] for r in anchor_roles if r.get("title")]
                    covered_str   = ", ".join(covered_acro) if covered_acro else "(none)"

                    ctx_blocks.append(
                        f"STARTING POINT: {course['title']} ({cid})\n"
                        "This course aligns to work roles:\n" +
                        ("\n".join(f"  - {n}" for n in role_names) or "  (none)") + "\n"
                        f"Certifications already covered by this course: {covered_str}\n"
                        "Next-level certifications to pursue (and PCATT courses that prepare each):\n" +
                        ("\n".join(next_ctx_lines) or "  (none found in catalog)")
                    )

            # ── Q7: broad_domain ─────────────────────────────────────────────
            elif template_id == "Q7" and roles:
                ctx_blocks.append("CATALOG VIEW — all PCATT courses across relevant work roles:\n")
                for role in roles[:BROAD_ROLE_CAP]:
                    ctx_blocks.append(_role_subgraph(s, role, add_node, add_edge))

            # ── Q8: course_compare ────────────────────────────────────────────
            elif template_id == "Q8" and len(courses) >= 2:
                for i, course in enumerate(courses[:5], start=1):
                    ctx_blocks.append(
                        f"COURSE {i}:\n{_course_subgraph(s, course, add_node, add_edge)}"
                    )

            # ── Q1 (default): role_anchor ─────────────────────────────────────
            else:
                seen_ids = {r["id"] for r in roles[:3]}
                q1_roles = list(roles[:3])
                for cr in _roles_from_courses(s, courses[:3]):
                    if cr["id"] not in seen_ids and len(q1_roles) < 5:
                        seen_ids.add(cr["id"])
                        q1_roles.append(cr)
                for role in q1_roles:
                    ctx_blocks.append(_role_subgraph(s, role, add_node, add_edge))

    # Anchor entities for the path SSE event (drives active path highlighting)
    if template_id == "Q5" and certs:
        anchors = [{"id": c["id"], "title": c.get("full_name") or c.get("acronym") or c["id"],
                    "category": "Certification", "similarity": c["similarity"]}
                   for c in certs[:1]]
    elif template_id in ("Q2", "Q6", "Q8") and courses:
        anchors = [{"id": c["id"], "title": c["title"],
                    "category": "Course", "similarity": c["similarity"]}
                   for c in courses[:5]]
    elif template_id in ("Q4", "Q7"):
        anchors = [{"id": r["id"], "title": r["title"],
                    "category": r["category"], "similarity": r["similarity"]}
                   for r in roles[:5]]
    else:
        anchors = [{"id": r["id"], "title": r["title"],
                    "category": r["category"], "similarity": r["similarity"]}
                   for r in roles[:5]]

    return {
        "nodes":   nodes,
        "edges":   edges,
        "context": "\n\n---\n\n".join(ctx_blocks),
        "anchors": anchors,
    }


# ─── Stream ───────────────────────────────────────────────────────────────────

async def _graph_rag_stream(question: str) -> AsyncGenerator[str, None]:
    # 1. Search all three entity collections
    entities = _query_entities(question)
    if not entities["roles"] and not entities["courses"] and not entities["certs"]:
        yield f"data: {json.dumps({'type': 'token', 'token': 'No matching entities found for this question.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. Route — LLM picks from closed set Q1–Q8
    template_id, reason = await _route(question, entities)

    # 3. Emit route decision — frontend decision trace card
    yield f"data: {json.dumps({'type': 'route', 'template': template_id, 'label': TEMPLATE_META[template_id]['label'], 'reason': reason})}\n\n"

    # 4. Deterministic graph traversal anchored on verified entity IDs
    graph = _traverse_graph(template_id, entities)

    # 5. Emit path — frontend lights up the graph panel
    yield f"data: {json.dumps({'type': 'path', 'nodes': graph['nodes'], 'edges': graph['edges'], 'roles': graph['anchors']})}\n\n"

    if not graph["context"]:
        yield f"data: {json.dumps({'type': 'token', 'token': 'Graph traversal returned no data for this question.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 6. LLM synthesis grounded strictly to the verified subgraph
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Graph data:\n\n{graph['context']}\n\nQuestion: {question}"),
    ]
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/query")
async def graph_rag_query(request: QueryRequest):
    """
    Mode C: GraphRAG with semantic routing.
    Vector search anchors entity IDs → LLM picks one of 8 query templates →
    deterministic Cypher traversal → LLM synthesis from verified subgraph.
    """
    return StreamingResponse(
        _graph_rag_stream(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
