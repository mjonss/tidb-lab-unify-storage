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
| Who bakes cinnamon buns on Fridays? | Is there a Japanese baker in Stockholm? |
| I have a leaking pipe, who can help? | Who fixes leaking pipes in Mölndal? |

Also try "What does Björn Borg do for a living?" (two people share the name), "INC-2041" (an identifier only full-text
search finds) and "Who saw the northern lights in July?".

Without AWS, TiDB can embed questions itself with its hosted model, on the same vectors:

```sql
SELECT id, name FROM employees
ORDER BY VEC_COSINE_DISTANCE(intro_vec, EMBED_TEXT('tidbcloud_free/amazon/titan-embed-text-v2', 'Who has beehives?'))
LIMIT 3;
```

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
