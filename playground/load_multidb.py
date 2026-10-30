"""Load the playground data with precomputed vectors into MySQL, Elasticsearch and Milvus, for improved_rag.py.

Usage: poetry run python playground/load_multidb.py [DATA.parquet] [--vector-column intro_vec] [--reset]
DATA defaults to data/persons_playground.parquet. Uses the names and settings from improved_rag.py: MySQL table
`employee` (id, name, intro), Elasticsearch index and Milvus collection `employee_id_mapping` (id + 1024-dimension
embedding, AUTOINDEX). Connects with the MYSQL_*, ELASTICSEARCH_* and MILVUS_* variables that docker-compose.yml sets.
MySQL skips ids it already has; Elasticsearch and Milvus overwrite by id, so a rerun completes an interrupted load.
Rows are converted 1,000 at a time, so memory stays small however large the file is.
"""

import argparse
import os
import time

import pyarrow.parquet as pq
import pymysql
from elasticsearch import Elasticsearch, helpers
from pymilvus import DataType, MilvusClient

NAME = "employee_id_mapping"
BATCH = 1000
KIT = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("data", nargs="?", default=os.path.join(KIT, "data", "persons_playground.parquet"))
parser.add_argument("--vector-column", default="intro_vec")
parser.add_argument("--reset", action="store_true", help="empty all three stores first")
args = parser.parse_args()

data = pq.read_table(args.data, columns=["id", "name", "intro", args.vector_column])
dim = len(data[args.vector_column][0].as_py())
mysql = pymysql.connect(host=os.environ["MYSQL_HOST"], port=int(os.environ.get("MYSQL_PORT", "4000")), user="root",
                        password=os.environ["MYSQL_ROOT_PASSWORD"], database=os.environ["MYSQL_DATABASE"],
                        autocommit=True)
es = Elasticsearch(f"http://{os.environ['ELASTICSEARCH_HOST']}:{os.environ['ELASTICSEARCH_PORT']}", request_timeout=60)
milvus = MilvusClient(uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ['MILVUS_GRPC_PORT']}", token="root:Milvus")

cur = mysql.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS employee (id INT NOT NULL AUTO_INCREMENT, name VARCHAR(255) NOT NULL,"
            " intro VARCHAR(255) NOT NULL, PRIMARY KEY (id))")  # what SQLModel creates for improved_rag.Employee
if args.reset:
    cur.execute("TRUNCATE TABLE employee")
    if es.indices.exists(index=NAME):
        es.indices.delete(index=NAME)
    if milvus.has_collection(NAME):
        milvus.drop_collection(NAME)
if not es.indices.exists(index=NAME):
    es.indices.create(index=NAME, body={"mappings": {"properties": {
        "id": {"type": "integer"}, "name": {"type": "text"}, "intro": {"type": "text"}}}}, wait_for_active_shards="0")
if es.options(ignore_status=408).cluster.health(index=NAME, wait_for_status="yellow", timeout="30s")["timed_out"]:
    raise SystemExit(f"Elasticsearch cannot place index {NAME}. The usual cause is a disk over 90% full;"
                     " check GET /_cat/allocation?v on Elasticsearch.")
if not milvus.has_collection(NAME):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    index_params = milvus.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX")
    milvus.create_collection(collection_name=NAME, schema=schema, index_params=index_params)

cur.execute("SELECT id FROM employee")
existing = {r[0] for r in cur.fetchall()}
print(f"Loading {data.num_rows} rows ({len(existing)} already in MySQL), vectors from {args.vector_column}")
batches = [data.slice(i, BATCH) for i in range(0, data.num_rows, BATCH)]

start = time.time()
for batch in batches:
    rows = [r for r in batch.select(["id", "name", "intro"]).to_pylist() if r["id"] not in existing]
    if rows:
        cur.executemany("INSERT INTO employee (id, name, intro) VALUES (%s, %s, %s)",
                        [(r["id"], r["name"], r["intro"]) for r in rows])
print(f"MySQL: {time.time() - start:.1f}s")

start = time.time()
helpers.bulk(es, ({"_index": NAME, "_id": r["id"], "_source": r}
                  for batch in batches for r in batch.select(["id", "name", "intro"]).to_pylist()), chunk_size=BATCH)
es.indices.refresh(index=NAME)
print(f"Elasticsearch: {time.time() - start:.1f}s, {es.count(index=NAME)['count']} documents")

start = time.time()
for batch in batches:
    ids, vectors = batch["id"].to_pylist(), batch[args.vector_column].to_pylist()
    milvus.upsert(collection_name=NAME, data=[{"id": i, "embedding": v} for i, v in zip(ids, vectors)])
milvus.flush(NAME)
milvus.load_collection(NAME)
# get_collection_stats() also counts rows replaced by upsert until Milvus compacts them, so count with a query.
count = milvus.query(NAME, filter="", output_fields=["count(*)"])[0]["count(*)"]
print(f"Milvus: {time.time() - start:.1f}s, {count} entities")
