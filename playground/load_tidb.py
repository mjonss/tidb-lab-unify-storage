"""Load the playground data with precomputed vectors into TiDB, so no embedding model is called.

Usage: poetry run python playground/load_tidb.py [DATA.parquet] [--table persons] [--map SOURCE=TARGET] [--threads 8] [--env-file FILE]
DATA defaults to data/persons_playground.parquet. Connects with the SERVERLESS_CLUSTER_* variables from .env (or the ones in --env-file, e.g. for your own Starter cluster).

Creates the table if it is missing: id, name, intro, one VECTOR column per vector column in the file, a full-text
index on intro and a cosine vector index per vector column. If the table exists, it inserts into its columns and
skips ids that are already there, so an interrupted load can be rerun.
--map SOURCE=TARGET loads only the mapped vector columns, under the target name. For the lab's tidb_rag.py table,
whose vector column is also called intro_vec:
  poetry run python playground/load_tidb.py --table employees
Query a column filled from TiDB's hosted model without any client-side model:
  ORDER BY VEC_COSINE_DISTANCE(intro_vec, EMBED_TEXT('tidbcloud_free/amazon/titan-embed-text-v2', 'question'))
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pymysql

parser = argparse.ArgumentParser()
parser.add_argument("data", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "persons_playground.parquet"))
parser.add_argument("--table", default="persons")
parser.add_argument("--map", action="append", default=[], help="SOURCE=TARGET vector column")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--batch", type=int, default=500)
parser.add_argument("--env-file")
args = parser.parse_args()

if args.env_file:
    for line in open(args.env_file, encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ[key.removeprefix("export ").strip()] = value.strip().strip("'\"")

data = pq.read_table(args.data)
vector_sources = [f.name for f in data.schema if pa.types.is_list(f.type) or pa.types.is_fixed_size_list(f.type)]
mapping = dict(m.split("=", 1) for m in args.map) if args.map else {c: c for c in vector_sources}
missing = [c for c in mapping if c not in data.column_names]
if missing:
    raise SystemExit(f"Not in {args.data}: {missing}. Vector columns there: {vector_sources}")


def connect():
    return pymysql.connect(
        host=os.environ["SERVERLESS_CLUSTER_HOST"], port=int(os.environ.get("SERVERLESS_CLUSTER_PORT", "4000")),
        user=os.environ["SERVERLESS_CLUSTER_USERNAME"], password=os.environ["SERVERLESS_CLUSTER_PASSWORD"],
        database=os.environ["SERVERLESS_CLUSTER_DATABASE_NAME"], ssl={"ca": None}, autocommit=True,
    )


conn = connect()
cur = conn.cursor()
cur.execute("SHOW TABLES LIKE %s", (args.table,))
if not cur.fetchone():
    dims = {target: len(data[source][0].as_py()) for source, target in mapping.items()}
    columns = ["id INT PRIMARY KEY", "name VARCHAR(255) NOT NULL", "intro TEXT NOT NULL"]
    columns += [f"`{t}` VECTOR({d})" for t, d in dims.items()]
    columns += ["FULLTEXT INDEX fts_idx_intro (intro) WITH PARSER MULTILINGUAL"]
    columns += [f"VECTOR INDEX vec_idx_{t}_cosine ((VEC_COSINE_DISTANCE(`{t}`)))" for t in dims]
    cur.execute(f"CREATE TABLE `{args.table}` (\n  " + ",\n  ".join(columns) + "\n)")
    print(f"Created table {args.table}")
cur.execute(f"SHOW COLUMNS FROM `{args.table}`")
existing_columns = {r[0] for r in cur.fetchall()}
absent = [t for t in ["id", "name", "intro", *mapping.values()] if t not in existing_columns]
if absent:
    raise SystemExit(f"Table {args.table} has no column(s) {absent}; use --map or another --table")
cur.execute(f"SELECT id FROM `{args.table}`")
existing_ids = {r[0] for r in cur.fetchall()}

ids = data["id"].to_pylist()
keep = [i for i, row_id in enumerate(ids) if row_id not in existing_ids]
targets = ["id", "name", "intro", *mapping.values()]
sql = (f"INSERT INTO `{args.table}` ({', '.join(f'`{t}`' for t in targets)}) VALUES ({', '.join(['%s'] * len(targets))})")
print(f"Inserting {len(keep)} rows into {args.table} ({len(existing_ids)} already there), vector columns {mapping}")


def vector_text(v):
    return None if v is None else "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def insert(chunk):
    c = connect()
    try:
        rows = data.take(chunk).to_pylist()
        values = [(r["id"], r["name"], r["intro"], *(vector_text(r[s]) for s in mapping)) for r in rows]
        for attempt in range(5):
            try:
                c.cursor().executemany(sql, values)
                return len(values)
            except pymysql.err.OperationalError as e:
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(2**attempt)
        raise RuntimeError(f"batch starting at id {rows[0]['id']} failed 5 times")
    finally:
        c.close()


start, done = time.time(), 0
with ThreadPoolExecutor(args.threads) as pool:
    for n in pool.map(insert, [keep[i : i + args.batch] for i in range(0, len(keep), args.batch)]):
        done += n
        if done % 5000 < args.batch or done == len(keep):
            print(f"{done}/{len(keep)} rows, {done / (time.time() - start):.0f} rows/s", flush=True)
cur.execute(f"ALTER TABLE `{args.table}` COMPACT")
cur.execute(f"SELECT COUNT(*) FROM `{args.table}`")
print(f"Done in {time.time() - start:.1f}s; {args.table} now has {cur.fetchone()[0]} rows")
print("Full-text ranking settles a few minutes after a load: wait about 3 minutes before trying questions.")
