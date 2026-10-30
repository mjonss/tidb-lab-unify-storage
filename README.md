# TiDB Lab: RAG with unified storage

A fork of [Icemap/tidb-lab-unify-storage](https://github.com/Icemap/tidb-lab-unify-storage), the code behind the TiDB
Labs RAG lab, for trying the lab on your own machine after it ends. It adds a 5,105-person playground dataset with
precomputed vectors and loaders for it, and fixes problems found while running the lab (listed below).

The original repository has no license file, so its code comes with no license beyond GitHub's terms of service; ask
its author before redistributing it. The playground data is described in
[playground/data/ATTRIBUTION.md](playground/data/ATTRIBUTION.md).

## The apps

| App | Stores | Retrieval |
|---|---|---|
| `vanilla_ai_app.py` | none | none: the model answers from what it already knows |
| `rag.py` | MySQL and Milvus | vector search, top row |
| `improved_rag.py` | MySQL, Elasticsearch and Milvus | vector search and full-text search, top row of each |
| `tidb_rag.py` | TiDB | hybrid search (vector and full-text in one query), 3 rows (`RAG_LIMIT`) |

All four use Amazon Bedrock: Titan Text Embeddings V2 for vectors and Nova Pro for answers.

## What you need

- Docker with Compose.
- A TiDB Cloud Starter cluster (the free tier is enough). Full-text search is not available in self-managed TiDB.
- An AWS account with Bedrock access to `amazon.titan-embed-text-v2:0` and Nova Pro. The apps call
  `us.amazon.nova-pro-v1:0`, a US cross-region inference profile, so use a US region.

## Run it

```sh
cp .env.example .env          # then fill in the TiDB and AWS values
docker compose up -d --build  # the first build takes a few minutes
```

Open VS Code at http://localhost:3000 (no login; it listens on localhost only) and, in its terminal:

```sh
cd /root/tidb-lab-unify-storage && poetry install
poetry run streamlit run tidb_rag.py --server.port 8501
```

Then open http://localhost:8501. The other apps can use ports 8502 to 8504.

## Playground data

`playground/data/persons_playground.parquet`: 69 real public people, 36 fictional people with planted details that no
model can know, and 5,000 synthetic filler people that make generic questions harder, as in real data. The vectors
(column `intro_vec`) are Titan Text Embeddings V2; Bedrock's `amazon.titan-embed-text-v2:0` and TiDB Cloud's hosted
`tidbcloud_free/amazon/titan-embed-text-v2` gave bit-for-bit identical vectors for every row.

Start an app and open it once, so it creates its tables, then load the data:

```sh
poetry run python playground/load_tidb.py --table employees   # for tidb_rag.py
poetry run python playground/load_multidb.py                  # for rag.py and improved_rag.py
```

Wait about 3 minutes before asking questions: right after a load, TiDB's full-text ranking differs from the settled
one. `playground/data/questions.txt` has 58 questions. Pairs that show what retrieval can and cannot do:

| Misses (generic) | Hits (specific) |
|---|---|
| Who is a beekeeper? | Who has beehives? |
| Who bakes cinnamon buns on Fridays? | ストックホルムのパン職人 (Japanese for "Stockholm's bread baker"; her intro is in Japanese) |
| Is there a Banksy in Gothenburg? | Is there an anonymous street artist in Gothenburg? |

These behave the same with exact vector search, with TiDB's vector index, and in `tidb_rag.py` (checked 2026-10-30).

Also try "What does Björn Borg do for a living?" (two people share the name), "INC-2041" (an identifier only full-text
search finds), "Who fixes leaking pipes in Mölndal?" and "Who saw the northern lights in July?".

TiDB's vector index is approximate. Once fully built, it missed the nearest row for 9 of the 58 questions on this data
(recall at 3: 0.88); for example "Who keeps bees?" loses Elin Rosengren from the vector results. With 5,105 rows an exact
search is cheap (about 60 ms instead of 40): add `+ 0` to the distance in `ORDER BY` so that TiDB compares every row.

Without AWS, TiDB can embed questions itself with its hosted model, on the same vectors:

```sql
SELECT id, name FROM employees
ORDER BY VEC_COSINE_DISTANCE(intro_vec, EMBED_TEXT('tidbcloud_free/amazon/titan-embed-text-v2', 'Who has beehives?'))
LIMIT 3;
```

## RAG explorer

`rag_explorer.py` sends one question through five ways of searching the same data and shows every request (SQL, Bedrock,
Milvus, Elasticsearch, MySQL) with its wall-clock time and the rows it found:

1. TiDB, one SQL statement: TiDB embeds the question itself (`EMBED_TEXT`) and runs the whole hybrid search.
2. TiDB, several statements: the vector comes back to the app, rank fusion runs in Python.
3. TiDB, the app embeds the question (Bedrock, as the lab apps), then one SQL statement.
4. pytidb, as `tidb_rag.py`: the SQL that pytidb 0.0.6 sends, captured.
5. MySQL + Elasticsearch + Milvus, as `improved_rag.py`.

"All five" runs them one after another for a side-by-side comparison of results, round trips and time. Load the
playground data first (both loaders), then:

```sh
cd /root/tidb-lab-unify-storage && poetry run streamlit run rag_explorer.py --server.port 8503
```

TiDB caches the embedding of a question it has seen, so the first request that embeds a new question pays for computing it;
search again to compare warm times.

## Changes from the original

- `tidb_rag.py` checks for its full-text index with `SHOW INDEXES`: pytidb's `has_fts_index()` always returns False on
  TiDB Cloud Starter ([pytidb#264](https://github.com/pingcap/pytidb/issues/264)), so every rerun failed with
  "Duplicate key name 'fts_idx_intro'".
- `tidb_rag.py` returns 3 rows instead of 1. With 1, the vector and full-text top rows tie and one is dropped.
- The "All Employees" listings show the first 100 rows. They fetched the whole table on every rerun, which failed at
  50,105 rows (218 MB, over Streamlit's 200 MB limit).
- The Bedrock inference profile comes from `BEDROCK_MODEL_ID` instead of a fixed ARN from the lab's AWS account.
- `docker-compose.yml` runs Milvus without MinIO, whose image can no longer be pulled, and builds the VS Code image from
  `docker/vscode.Dockerfile` (the original image is amd64 only and based on Debian 11, which cannot be rebuilt).
