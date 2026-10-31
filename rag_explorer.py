"""RAG explorer: one question through five ways of searching the lab's data, showing every request and its time.

Solutions:
  1. TiDB, one SQL statement: TiDB embeds the question itself (EMBED_TEXT) and runs the whole hybrid search.
  2. TiDB, several statements: TiDB embeds, the vector comes back to the app, rank fusion runs in Python.
  3. TiDB, the app embeds the question (Bedrock, as the lab apps do), then one SQL statement.
  4. pytidb, as tidb_rag.py: the SQL that pytidb 0.0.6 sends, captured.
  5. MySQL + Elasticsearch + Milvus, as improved_rag.py.
  6. All five, one after another, to compare their parts and total time.
Times are wall clock in this app, so they include the network. Each solution uses as few round trips as it can.

Run after loading the playground data (playground/load_tidb.py --table employees, playground/load_multidb.py):
  cd /root/tidb-lab-unify-storage && poetry run streamlit run rag_explorer.py --server.port 8503
Configuration: the lab's SERVERLESS_CLUSTER_*, MYSQL_*, ELASTICSEARCH_* and MILVUS_* variables, and optionally
RAG_TIDB_TABLE (default employees), RAG_MYSQL_TABLE (employee), RAG_SEARCH_INDEX (employee_id_mapping).
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import pymysql
import pymysql.cursors
import streamlit as st

TIDB_MODEL = "tidbcloud_free/amazon/titan-embed-text-v2"
BEDROCK_MODEL = "bedrock/amazon.titan-embed-text-v2:0"  # the lab apps' model; same vectors as TIDB_MODEL
TIDB_TABLE = os.getenv("RAG_TIDB_TABLE", "employees")
MYSQL_TABLE = os.getenv("RAG_MYSQL_TABLE", "employee")
SEARCH_INDEX = os.getenv("RAG_SEARCH_INDEX", "employee_id_mapping")  # Elasticsearch index and Milvus collection

SOLUTIONS = {
    "tidb_one": "1. TiDB, one SQL statement (TiDB embeds the question)",
    "tidb_several": "2. TiDB, several statements (TiDB embeds the question)",
    "tidb_app": "3. TiDB, the app embeds the question, then one SQL statement",
    "pytidb": "4. pytidb, as tidb_rag.py",
    "multidb": "5. MySQL + Elasticsearch + Milvus, as improved_rag.py",
    "all": "All five, one after another",
}
NOTES = {
    "tidb_one": "One round trip: TiDB embeds the question inside the statement.",
    "tidb_several": "The vector comes back to the app and goes to TiDB again; rank fusion runs in Python.",
    "tidb_app": "The app embeds the question, then one statement does the whole search.",
    "pytidb": "pytidb 0.0.6 embeds in the app and sends two statements; it always uses the vector index.",
    "multidb": "Three databases: Milvus (always its index) for vectors, Elasticsearch for words, MySQL for the rows. "
               "Fused like the others; improved_rag.py itself keeps only the top row of each.",
}
EMBEDDERS = {"bedrock": "Bedrock, as the lab apps", "tidb": "TiDB's hosted model (SELECT EMBED_TEXT)"}
EXAMPLES = [
    "Who is a beekeeper?", "Who has beehives?", "Who keeps bees?", "Who bakes cinnamon buns on Fridays?",
    "ストックホルムのパン職人", "Is there a Banksy in Gothenburg?", "Is there an anonymous street artist in Gothenburg?",
    "What does Björn Borg do for a living?", "INC-2041", "Who fixes leaking pipes in Mölndal?",
    "Who saw the northern lights in July?", "Finns det någon rörmokare?", "Is there a plumber?",
    "Who is Cheese Wang at PingCAP?",
]


@dataclass
class Step:
    system: str  # TiDB, Bedrock, Milvus, Elasticsearch, MySQL or Python
    label: str
    request: str
    language: str
    ms: float
    extra: bool = False  # only fills in values for the rows shown


@dataclass
class Run:
    solution: str
    steps: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    total_ms: float = 0.0
    error: Optional[str] = None

    def call(self, system, label, request, language, fn, extra=False):
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.steps.append(Step(system, label, request, language, (time.perf_counter() - start) * 1000, extra))


def lit(text: str) -> str:
    return "'" + pymysql.converters.escape_string(text) + "'"


def vector_text(vector) -> str:
    return vector if isinstance(vector, str) else "[" + ",".join(f"{x:.9g}" for x in vector) + "]"


def short(text: str) -> str:
    """Vectors are 1,024 numbers; show the first three."""
    def cut(m):
        values = m.group(1).split(",")
        return "[" + ",".join(values[:3]) + f", ... {len(values) - 3} more]"
    return re.sub(r"\[([-0-9.eE+, ]{200,})\]", cut, text)


def kind(row_id: int) -> str:
    if row_id < 100:
        return "lab sample" if row_id < 4 else "event"
    return "real person" if row_id < 1000 else ("fictional" if row_id < 100000 else "filler")


def rrf(vector_ids, fulltext_ids, limit):
    """Reciprocal rank fusion with k = 60, as pytidb does; ties by best rank, then id (as the SQL version)."""
    fused, best, found = {}, {}, {}
    for side, ids in (("vector", vector_ids), ("full-text", fulltext_ids)):
        for rank, row_id in enumerate(ids, start=1):
            fused[row_id] = fused.get(row_id, 0) + 1 / (60 + rank)
            best[row_id] = min(best.get(row_id, 99), rank)
            found.setdefault(row_id, []).append(side)
    order = sorted(fused, key=lambda i: (-fused[i], best[i], i))[:limit]
    return [(i, fused[i], " and ".join(found[i])) for i in order]


# Connections, kept between reruns so that every solution is timed on a warm connection.

@st.cache_resource
def tidb():
    return pymysql.connect(
        host=os.environ["SERVERLESS_CLUSTER_HOST"], port=int(os.environ.get("SERVERLESS_CLUSTER_PORT", "4000")),
        user=os.environ["SERVERLESS_CLUSTER_USERNAME"], password=os.environ["SERVERLESS_CLUSTER_PASSWORD"],
        database=os.environ["SERVERLESS_CLUSTER_DATABASE_NAME"], ssl={"ca": None}, autocommit=True,
        cursorclass=pymysql.cursors.DictCursor)


@st.cache_resource
def mysql():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"], port=int(os.environ.get("MYSQL_PORT", "3306")), user="root",
        password=os.environ["MYSQL_ROOT_PASSWORD"], database=os.environ["MYSQL_DATABASE"], autocommit=True,
        cursorclass=pymysql.cursors.DictCursor)


@st.cache_resource
def elasticsearch():
    from elasticsearch import Elasticsearch
    return Elasticsearch(f"http://{os.environ['ELASTICSEARCH_HOST']}:{os.environ['ELASTICSEARCH_PORT']}", request_timeout=30)


@st.cache_resource
def milvus():
    from pymilvus import MilvusClient
    client = MilvusClient(uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ['MILVUS_GRPC_PORT']}", token="root:Milvus")
    client.load_collection(SEARCH_INDEX)
    return client


def sql(connection, statement: str) -> list:
    connection.ping(reconnect=True)
    with connection.cursor() as cur:
        cur.execute(statement)
        return list(cur.fetchall())


@st.cache_resource
def pytidb_table():
    """pytidb 0.0.6 set up as tidb_rag.py sets it up, with hooks that record its SQL and its embedding call.

    Returns the table and a capture state. The state lives here, in the cached resource, because Streamlit runs the
    script again for every search, while these hooks are installed only once.
    """
    from types import SimpleNamespace
    from typing import Any as AnyType
    from sqlalchemy import event
    from pytidb import TiDBClient
    from pytidb.embeddings import EmbeddingFunction
    from pytidb.schema import Field, TableModel
    import pytidb.embeddings.litellm as pytidb_litellm

    original = pytidb_litellm.get_embeddings
    capture = SimpleNamespace(run=None, embedder="bedrock", vector=None)

    def get_embeddings(model_name, input, **kwargs):
        run, text = capture.run, input[0]
        if capture.embedder == "tidb":
            statement = f"-- pytidb asks for the question's vector; here TiDB's hosted model computes it\nSELECT EMBED_TEXT({lit(TIDB_MODEL)}, {lit(text)}) AS v"
            result = [json.loads(run.call("TiDB", "embed the question for pytidb", statement, "sql",
                                          lambda: sql(tidb(), statement))[0]["v"])]
        else:
            request = f'litellm.embedding(model="{model_name}", input=[{text!r}], dimensions={kwargs.get("dimensions")})'
            result = run.call("Bedrock", "pytidb embeds the question (as tidb_rag.py does)", request, "python",
                              lambda: original(model_name, input, **kwargs))
        capture.vector = result[0]
        return result

    pytidb_litellm.get_embeddings = get_embeddings
    db = TiDBClient.connect(
        host=os.environ["SERVERLESS_CLUSTER_HOST"], port=int(os.environ.get("SERVERLESS_CLUSTER_PORT", "4000")),
        username=os.environ["SERVERLESS_CLUSTER_USERNAME"], password=os.environ["SERVERLESS_CLUSTER_PASSWORD"],
        database=os.environ["SERVERLESS_CLUSTER_DATABASE_NAME"], enable_ssl=True)
    # dimensions=1024 skips the extra embedding call that EmbeddingFunction otherwise makes at startup.
    embed_fn = EmbeddingFunction(BEDROCK_MODEL, dimensions=1024, timeout=60)

    class Employee(TableModel, table=True):
        __tablename__ = TIDB_TABLE
        __table_args__ = {"extend_existing": True}
        id: int | None = Field(default=None, primary_key=True)
        name: str
        intro: str
        intro_vec: Optional[AnyType] = embed_fn.VectorField(source_field="intro")

    @event.listens_for(db.db_engine, "before_cursor_execute")
    def before(conn, cursor, statement, parameters, context, executemany):
        conn.info["rag_start"] = time.perf_counter()

    @event.listens_for(db.db_engine, "after_cursor_execute")
    def after(conn, cursor, statement, parameters, context, executemany):
        if capture.run is not None:
            capture.run.steps.append(Step("TiDB", "SQL sent by pytidb", render(statement, parameters), "sql",
                                          (time.perf_counter() - conn.info.pop("rag_start")) * 1000))

    return db.open_table(TIDB_TABLE), capture


def render(statement: str, parameters) -> str:
    """pytidb's statement with its parameters written in, as TiDB receives it."""
    def value(v):
        if isinstance(v, (list, tuple)):
            return lit(vector_text(v))
        if isinstance(v, str):
            return lit(v)
        return str(v)
    if isinstance(parameters, dict):
        return re.sub(r"%\((\w+)\)s", lambda m: value(parameters[m.group(1)]), statement)
    return statement


# The SQL for the TiDB solutions.

def tidb_statement(mode, exact, limit, question, vector=None) -> str:
    """One statement. Without a vector, TiDB embeds the question; with one, the app's vector is in the SQL."""
    q, table = lit(question), f"`{TIDB_TABLE}`"
    order, how = ("d + 0", 'compares every row exactly ("+ 0" skips the vector index)') if exact else ("d", "uses the vector index (fast, approximate)")
    match = f"fts_match_word({q}, intro)"
    if mode == "full-text":
        return (f"-- Full-text search: the {limit} people whose intro best matches the question's words (BM25 score)\n"
                f"SELECT id, name, intro, {match} AS score\nFROM {table}\nWHERE {match}\nORDER BY {match} DESC\nLIMIT {limit}")
    if vector is None:
        embed = f"EMBED_TEXT({lit(TIDB_MODEL)}, {q})"
        question_step = f"question AS (\n  SELECT {embed} AS v\n)"
        reference, final = "(SELECT v FROM question)", embed
    else:
        question_step, reference = None, lit(vector_text(vector))
        final = reference
    if mode == "vector":
        head = (f"WITH\n-- Embed the question inside TiDB, with its free hosted model\n{question_step}\n" if question_step else
                "-- The question's vector was computed by the app\n")
        return (f"-- Vector search: the {limit} people whose intro is closest in meaning to the question\n{head}"
                f"SELECT id, name, intro, VEC_COSINE_DISTANCE(intro_vec, {reference}) AS d\nFROM {table}\n"
                f"ORDER BY {order}  -- {how}\nLIMIT {limit}")
    n = iter(range(1, 10))
    parts = []
    if question_step:
        parts.append(f"-- {next(n)}. Embed the question inside TiDB, with its free hosted model\n{question_step}")
    parts.append(f"""-- {next(n)}. Vector search: the {limit} people closest in meaning; {how}
vec AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY d) AS r
  FROM (
    SELECT id, VEC_COSINE_DISTANCE(intro_vec, {reference}) AS d
    FROM {table}
    ORDER BY {order}
    LIMIT {limit}
  ) AS t
)""")
    parts.append(f"""-- {next(n)}. Full-text search: the {limit} people whose intro best matches the question's words
fts AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY s DESC) AS r
  FROM (
    SELECT id, {match} AS s
    FROM {table}
    WHERE {match}
    ORDER BY {match} DESC
    LIMIT {limit}
  ) AS t
)""")
    parts.append(f"""-- {next(n)}. Reciprocal rank fusion, as pytidb does: each side adds 1 / (60 + rank)
fused AS (
  SELECT id, SUM(1 / (60 + r)) AS fused, MIN(r) AS best,
         GROUP_CONCAT(side ORDER BY side DESC SEPARATOR ' and ') AS found
  FROM (
    SELECT id, r, 'vector' AS side FROM vec
    UNION ALL
    SELECT id, r, 'full-text' FROM fts
  ) AS sides
  GROUP BY id
)""")
    parts.append(f"""-- {next(n)}. Full-text scores of all matching people, also for those found only by vector search
--    (TiDB needs the bare match in WHERE; this LIMIT keeps the join condition out of it)
scores AS (
  SELECT id, {match} AS score
  FROM {table}
  WHERE {match}
  ORDER BY {match} DESC
  LIMIT 10000
)""")
    inline_note = ("The question is embedded inline here:\n--    a second reference to \"question\" makes TiDB compute it "
                   "separately, and then the vector search skips the index" if question_step else "")
    final_step = f"-- {next(n)}. The best {limit} after fusion, with both values for each. {inline_note}".rstrip() + f"""
SELECT p.id, p.name, p.intro,
       VEC_COSINE_DISTANCE(
         p.intro_vec,
         {final}
       ) AS distance,
       scores.score, f.fused, f.found
FROM fused AS f
JOIN {table} AS p ON p.id = f.id
LEFT JOIN scores ON scores.id = f.id
ORDER BY f.fused DESC, f.best, f.id
LIMIT {limit}"""
    return "-- Hybrid search in one statement: vector search and full-text search, combined by rank\nWITH\n" + ",\n".join(parts) + "\n" + final_step


def tidb_rows(rows) -> list:
    return [{"id": int(r["id"]), "name": r["name"], "intro": r["intro"],
             "distance": None if r.get("distance", r.get("d")) is None else float(r.get("distance", r.get("d"))),
             "score": None if r.get("score") is None else float(r["score"]),
             "fused": None if r.get("fused") is None else float(r["fused"]), "found": r.get("found")} for r in rows]


def fill_in_tidb(run, rows, vector, question):
    """Display only: the vector distance and full-text score of each row shown, where one side did not provide it."""
    ids = ", ".join(str(r["id"]) for r in rows)
    if not ids:
        return
    table, match = f"`{TIDB_TABLE}`", f"fts_match_word({lit(question)}, intro)"
    statement = (f"-- Display only: the vector distances of the people shown\n"
                 f"SELECT id, VEC_COSINE_DISTANCE(intro_vec, {lit(vector_text(vector))}) AS distance\nFROM {table}\nWHERE id IN ({ids})")
    distances = {int(r["id"]): float(r["distance"]) for r in run.call("TiDB", "vector distances of the rows shown", statement, "sql",
                                                                       lambda: sql(tidb(), statement), extra=True)}
    statement = (f"-- Display only: their full-text scores\nSELECT id, score\nFROM (\n  SELECT id, {match} AS score\n  FROM {table}\n"
                 f"  WHERE {match}\n  ORDER BY {match} DESC\n  LIMIT 10000\n) AS matches\nWHERE id IN ({ids})")
    scores = {int(r["id"]): float(r["score"]) for r in run.call("TiDB", "full-text scores of the rows shown", statement, "sql",
                                                                lambda: sql(tidb(), statement), extra=True)}
    for r in rows:
        r["distance"] = r["distance"] if r.get("distance") is not None else distances.get(r["id"])
        r["score"] = r["score"] if r.get("score") is not None else scores.get(r["id"])


def app_embed(run, question, embedder):
    """The app gets the question's vector: from Bedrock (as the lab apps) or from TiDB's hosted model."""
    if embedder == "tidb":
        statement = f"-- The app asks TiDB's hosted model for the question's vector\nSELECT EMBED_TEXT({lit(TIDB_MODEL)}, {lit(question)}) AS v"
        return run.call("TiDB", "embed the question", statement, "sql", lambda: sql(tidb(), statement))[0]["v"]
    from litellm import embedding
    request = f'litellm.embedding(model="{BEDROCK_MODEL}", input=[{question!r}], dimensions=1024)'
    response = run.call("Bedrock", "embed the question (as the lab apps do)", request, "python",
                        lambda: embedding(model=BEDROCK_MODEL, input=[question], dimensions=1024))
    return response.data[0]["embedding"]


# The five solutions.

def solve_tidb_one(run, question, mode, exact, limit, embedder):
    statement = tidb_statement(mode, exact, limit, question)
    run.rows = tidb_rows(run.call("TiDB", "the whole search", statement, "sql", lambda: sql(tidb(), statement)))


def solve_tidb_app(run, question, mode, exact, limit, embedder):
    vector = app_embed(run, question, embedder) if mode != "full-text" else None
    statement = tidb_statement(mode, exact, limit, question, vector)
    run.rows = tidb_rows(run.call("TiDB", "the whole search, with the app's vector", statement, "sql", lambda: sql(tidb(), statement)))


def solve_tidb_several(run, question, mode, exact, limit, embedder):
    table, match = f"`{TIDB_TABLE}`", f"fts_match_word({lit(question)}, intro)"
    vector_rows = fulltext_rows = []
    vector = None
    if mode != "full-text":
        statement = f"-- Embed the question inside TiDB; its 1,024-number vector comes back to the app\nSELECT EMBED_TEXT({lit(TIDB_MODEL)}, {lit(question)}) AS v"
        vector = run.call("TiDB", "embed the question", statement, "sql", lambda: sql(tidb(), statement))[0]["v"]
        how = 'compares every row exactly ("+ 0" skips the vector index)' if exact else "uses the vector index (fast, approximate)"
        statement = (f"-- Vector search with that vector, sent back to TiDB; {how}\nSELECT id, name, intro, "
                     f"VEC_COSINE_DISTANCE(intro_vec, {lit(vector)}) AS d\nFROM {table}\nORDER BY {'d + 0' if exact else 'd'}\nLIMIT {limit}")
        vector_rows = tidb_rows(run.call("TiDB", "vector search", statement, "sql", lambda: sql(tidb(), statement)))
    if mode != "vector":
        statement = (f"-- Full-text search: the {limit} people whose intro best matches the question's words\n"
                     f"SELECT id, name, intro, {match} AS score\nFROM {table}\nWHERE {match}\nORDER BY {match} DESC\nLIMIT {limit}")
        fulltext_rows = tidb_rows(run.call("TiDB", "full-text search", statement, "sql", lambda: sql(tidb(), statement)))
    if mode != "hybrid":
        run.rows = vector_rows or fulltext_rows
        return
    by_id = {r["id"]: r for r in fulltext_rows + vector_rows}
    fused = run.call("Python", "reciprocal rank fusion in the app", "rrf(vector_ids, fulltext_ids, k=60)", "python",
                     lambda: rrf([r["id"] for r in vector_rows], [r["id"] for r in fulltext_rows], limit))
    run.rows = [{**by_id[i], "fused": f, "found": found} for i, f, found in fused]
    fill_in_tidb(run, run.rows, vector, question)


def solve_pytidb(run, question, mode, exact, limit, embedder):
    table, capture = pytidb_table()
    capture.run, capture.embedder, capture.vector = run, embedder, None
    try:
        search = table.search(question, search_type={"full-text": "fulltext"}.get(mode, mode))
        if mode != "vector":
            search = search.text_column("intro")
        results = search.limit(limit).to_list()
    finally:
        capture.run = None
    run.rows = [{"id": int(r["id"]), "name": r["name"], "intro": r["intro"],
                 "distance": r.get("_distance"), "score": r.get("_match_score"),
                 "fused": r.get("_score") if mode == "hybrid" else None,
                 "found": " and ".join(s for s, k in (("vector", "_distance"), ("full-text", "_match_score")) if r.get(k) is not None)}
                for r in results]
    if mode == "hybrid":
        fill_in_tidb(run, run.rows, capture.vector, question)


def solve_multidb(run, question, mode, exact, limit, embedder):
    milvus_ids, es_ids, similarity, es_score = [], [], {}, {}
    vector = None
    if mode != "full-text":
        vector = app_embed(run, question, embedder)
        vector = json.loads(vector) if isinstance(vector, str) else vector
        request = (f'milvus.search(\n  collection_name="{SEARCH_INDEX}",\n  data=[{vector_text(vector)}],\n  anns_field="embedding",\n'
                   f'  limit={limit},\n  search_params={{"metric_type": "COSINE"}},\n  output_fields=["id"],\n)')
        hits = run.call("Milvus", "vector search (its index)", request, "python", lambda: milvus().search(
            collection_name=SEARCH_INDEX, data=[vector], anns_field="embedding", limit=limit,
            search_params={"metric_type": "COSINE"}, output_fields=["id"]))[0]
        milvus_ids = [int(h["id"]) for h in hits]
        similarity = {int(h["id"]): float(h["distance"]) for h in hits}  # COSINE in Milvus: similarity, higher is closer
    if mode != "vector":
        body = {"query": {"match": {"intro": question}}, "size": limit}
        request = f'es.search(\n  index="{SEARCH_INDEX}",\n  body={json.dumps(body, ensure_ascii=False, indent=2)},\n)'
        hits = run.call("Elasticsearch", "full-text search", request, "python",
                        lambda: elasticsearch().search(index=SEARCH_INDEX, body=body))["hits"]["hits"]
        es_ids = [int(h["_id"]) for h in hits]
        es_score = {int(h["_id"]): float(h["_score"]) for h in hits}
    if mode == "hybrid":
        fused = run.call("Python", "reciprocal rank fusion in the app", "rrf(milvus_ids, es_ids, k=60)", "python",
                         lambda: rrf(milvus_ids, es_ids, limit))
    else:
        fused = [(i, None, "vector" if mode == "vector" else "full-text") for i in (milvus_ids or es_ids)]
    ids = [i for i, _, _ in fused]
    if ids:
        statement = f"-- The rows themselves live in MySQL\nSELECT id, name, intro\nFROM `{MYSQL_TABLE}`\nWHERE id IN ({', '.join(map(str, ids))})"
        records = {int(r["id"]): r for r in run.call("MySQL", "fetch the rows", statement, "sql", lambda: sql(mysql(), statement))}
    if mode == "hybrid" and ids:
        missing_vector = [i for i in ids if i not in similarity]
        if missing_vector:
            request = (f'milvus.search(\n  collection_name="{SEARCH_INDEX}",\n  data=[{vector_text(vector)}],\n  anns_field="embedding",\n'
                       f'  filter="id in {missing_vector}",\n  limit={len(missing_vector)},\n  search_params={{"metric_type": "COSINE"}},\n)')
            hits = run.call("Milvus", "display only: distances of the rows shown", request, "python", lambda: milvus().search(
                collection_name=SEARCH_INDEX, data=[vector], anns_field="embedding", filter=f"id in {missing_vector}",
                limit=len(missing_vector), search_params={"metric_type": "COSINE"}, output_fields=["id"]), extra=True)[0]
            similarity.update({int(h["id"]): float(h["distance"]) for h in hits})
        missing_score = [i for i in ids if i not in es_score]
        if missing_score:
            body = {"query": {"bool": {"must": {"match": {"intro": question}}, "filter": {"ids": {"values": [str(i) for i in missing_score]}}}},
                    "size": len(missing_score)}
            request = f'es.search(\n  index="{SEARCH_INDEX}",\n  body={json.dumps(body, ensure_ascii=False, indent=2)},\n)'
            hits = run.call("Elasticsearch", "display only: scores of the rows shown", request, "python",
                            lambda: elasticsearch().search(index=SEARCH_INDEX, body=body), extra=True)["hits"]["hits"]
            es_score.update({int(h["_id"]): float(h["_score"]) for h in hits})
    run.rows = [{"id": i, "name": records[i]["name"], "intro": records[i]["intro"],
                 "distance": None if i not in similarity else 1 - similarity[i], "score": es_score.get(i),
                 "fused": f, "found": found} for i, f, found in fused if i in records]


SOLVERS = {"tidb_one": solve_tidb_one, "tidb_several": solve_tidb_several, "tidb_app": solve_tidb_app,
           "pytidb": solve_pytidb, "multidb": solve_multidb}


def prepare(key, embedder):
    """Open connections and import client libraries before the clock starts, so the times show the searches."""
    tidb()
    if key == "pytidb":
        pytidb_table()
    if key == "multidb":
        mysql(), elasticsearch(), milvus()
    if embedder == "bedrock" and key in ("tidb_app", "pytidb", "multidb"):
        import litellm  # noqa: F401  (the first import takes seconds)


def run_solution(key, question, mode, exact, limit, embedder) -> Run:
    run = Run(key)
    try:
        prepare(key, embedder)
    except Exception as e:
        run.error = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        return run
    start = time.perf_counter()
    try:
        SOLVERS[key](run, question, mode, exact, limit, embedder)
    except Exception as e:  # shown with the steps that ran, so a missing service does not hide the other solutions
        run.error = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    run.total_ms = (time.perf_counter() - start) * 1000
    return run


# The page.

def show_steps(run: Run):
    for i, step in enumerate(run.steps, start=1):
        tag = " · display only" if step.extra else ""
        st.markdown(f"**{i}. {step.system}**: {step.label} · {step.ms:.0f} ms wall clock{tag}")
        try:
            st.code(short(step.request), language=step.language, wrap_lines=True)
        except TypeError:  # older Streamlit without wrap_lines
            st.code(short(step.request), language=step.language)


def show_rows(run: Run, mode: str):
    if not run.rows:
        st.info("No rows found.")
        return
    columns = {"#": [], "Person": [], "Kind": [], "Intro": [], "Vector distance": [], "Full-text score": []}
    if mode == "hybrid":
        columns.update({"Fused": [], "Found by": []})
    for n, r in enumerate(run.rows, start=1):
        columns["#"].append(n)
        columns["Person"].append(f"{r['name']} ({r['id']})")
        columns["Kind"].append(kind(r["id"]))
        columns["Intro"].append(r["intro"])
        columns["Vector distance"].append(None if r.get("distance") is None else round(float(r["distance"]), 4))
        columns["Full-text score"].append(None if r.get("score") is None else round(float(r["score"]), 2))
        if mode == "hybrid":
            columns["Fused"].append(None if r.get("fused") is None else round(float(r["fused"]), 4))
            columns["Found by"].append(r.get("found") or "")
    st.dataframe(pd.DataFrame(columns), hide_index=True, use_container_width=True)


def summary(run: Run) -> dict:
    trips = [s for s in run.steps if s.system != "Python"]
    extra_ms = sum(s.ms for s in run.steps if s.extra)
    return {"Solution": SOLUTIONS[run.solution], "Round trips": len(trips), "Total ms": round(run.total_ms),
            "Display-only ms": round(extra_ms), "Search ms": round(run.total_ms - extra_ms),
            "Top rows": ", ".join(f"{r['name']} ({r['id']})" for r in run.rows[:3]) if not run.error else f"ERROR {run.error}"}


st.set_page_config(page_title="RAG Explorer", layout="wide")
st.title("RAG explorer")
st.caption("One question, five ways to search the lab's data. Every request is shown with its wall-clock time, measured in "
           "this app on open connections. TiDB caches the embedding of a question it has seen, so the first request that embeds "
           "a new question pays for computing it (often 0.5 to 1 second); search again to compare warm times. Bedrock does not cache.")


def use_example():
    if st.session_state.example:
        st.session_state.question = st.session_state.example
        st.session_state.go = True


st.selectbox("Example questions", [""] + EXAMPLES, key="example", on_change=use_example,
             format_func=lambda x: x or "Pick an example question")
with st.form("search"):
    question = st.text_input("Question", key="question")
    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 2])
    solution = c1.selectbox("Solution", list(SOLUTIONS), format_func=SOLUTIONS.get, key="solution")
    mode = c2.selectbox("Search", ["hybrid", "vector", "full-text"], key="mode")
    vectors = c3.selectbox("Vectors (TiDB)", ["exact", "index"], key="vectors")
    limit = c4.selectbox("Rows", [1, 3, 5, 10], index=1, key="limit")
    embedder = c5.selectbox("App embeds with", list(EMBEDDERS), format_func=EMBEDDERS.get, key="embedder")
    submitted = st.form_submit_button("Search")

if (submitted or st.session_state.pop("go", False)) and question.strip():
    question = question.strip()[:500]
    keys = [k for k in SOLVERS] if solution == "all" else [solution]
    runs = [run_solution(k, question, mode, vectors == "exact", limit, embedder) for k in keys]
    if solution == "all":
        st.subheader("Comparison")
        st.dataframe(pd.DataFrame([summary(r) for r in runs]), hide_index=True, use_container_width=True)
        st.caption("Run one after another on warm connections. \"Display-only\" requests fill in the other side's value "
                   "for the rows shown; the single TiDB statement does that inside the same round trip.")
        for run in runs:
            with st.expander(f"{SOLUTIONS[run.solution]}: {run.total_ms:.0f} ms, {summary(run)['Round trips']} round trips"):
                st.caption(NOTES[run.solution])
                if run.error:
                    st.error(run.error)
                show_rows(run, mode)
                show_steps(run)
    else:
        run = runs[0]
        info = summary(run)
        m1, m2, m3 = st.columns(3)
        m1.metric("Wall clock, total", f"{info['Total ms']} ms")
        m2.metric("Round trips", info["Round trips"])
        m3.metric("Of which display only", f"{info['Display-only ms']} ms")
        st.caption(NOTES[run.solution])
        if run.error:
            st.error(run.error)
        show_rows(run, mode)
        st.subheader("Requests, in order")
        show_steps(run)
