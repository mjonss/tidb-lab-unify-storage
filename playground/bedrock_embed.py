"""Compute Amazon Bedrock Titan Text Embeddings V2 vectors for the playground data, called the way the lab apps call it.

Needs AWS access to Bedrock: the lab machine's role in TiDB Labs, otherwise AWS_* variables in .env.
  poetry run python playground/bedrock_embed.py --check   compare Bedrock with TiDB's hosted Titan v2 on 39 samples
  poetry run python playground/bedrock_embed.py           every row of data/persons_playground.csv, and data/questions.txt
Options: --threads N (default 8), --limit N (first N rows only, for a trial), --out DIR (default out)
An interrupted run resumes where it stopped: finished rows are kept in OUT/bedrock_parts/.
Writes OUT/persons_bedrock.parquet (id, name, intro, intro_vec) and OUT/questions_bedrock.json.
Not needed for loading: data/persons_playground.parquet already holds these vectors. Use it to show the embedding step.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from litellm import embedding

MODEL = "bedrock/amazon.titan-embed-text-v2:0"  # same model and dimensions as tidb_rag.py and improved_rag.py
DIMENSIONS = 1024
KIT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--limit", type=int)
parser.add_argument("--out", default=os.path.join(KIT, "out"))
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def embed(text):
    """One Bedrock call, as improved_rag.get_embedding() makes it, retried with backoff when throttled."""
    for attempt in range(8):
        try:
            r = embedding(model=MODEL, input=[text], dimensions=DIMENSIONS, timeout=60)
            return r.data[0]["embedding"]
        except Exception as e:
            if attempt == 7:
                raise
            if attempt >= 2:
                print(f"  retry {attempt + 1}: {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)
            time.sleep(min(30, 2**attempt) + random.random())


try:  # one call without retries, so missing AWS access fails at once
    embedding(model=MODEL, input=["test"], dimensions=DIMENSIONS, timeout=60)
except Exception as e:
    sys.exit(f"Bedrock call failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}\n"
             "Check the AWS credentials and AWS_REGION_NAME (in .env), and that the account has Bedrock access to Titan v2.")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


if args.check:
    # Also embed two texts through pytidb's EmbeddingFunction, built as tidb_rag.py builds it.
    from pytidb.embeddings import EmbeddingFunction

    sample = json.load(open(os.path.join(KIT, "data", "hosted_sample.json"), encoding="utf-8"))
    print(f"Embedding {len(sample)} sample texts with {MODEL}")
    results = []
    with ThreadPoolExecutor(args.threads) as pool:
        for item, vec in zip(sample, pool.map(embed, [s["text"] for s in sample])):
            results.append({"id": item["id"], "kind": item["kind"], "cosine_vs_hosted": cosine(vec, item["vec"]),
                            "norm_bedrock": math.sqrt(sum(x * x for x in vec)),
                            "norm_hosted": math.sqrt(sum(x * x for x in item["vec"]))})
            if len(results) <= 3:
                results[-1]["bedrock_vector"] = vec
    pytidb_fn = EmbeddingFunction(MODEL, timeout=60)
    same_path = [cosine(pytidb_fn.get_source_embedding(s["text"]), embed(s["text"])) for s in sample[:2]]
    values = [r["cosine_vs_hosted"] for r in results]
    print(f"Cosine similarity, Bedrock vs TiDB hosted model: min {min(values):.6f}, mean {sum(values) / len(values):.6f}")
    print(f"Vector length: Bedrock {results[0]['norm_bedrock']:.4f}, hosted {results[0]['norm_hosted']:.4f}")
    print(f"pytidb EmbeddingFunction vs litellm call (the two lab apps): {', '.join(f'{c:.6f}' for c in same_path)}")
    if min(values) > 0.9999:
        print("Verdict: same vectors (up to rounding). One vector column would serve both models.")
    elif min(values) > 0.99:
        print("Verdict: nearly the same vectors, small differences. Keep both columns.")
    else:
        print("Verdict: different vectors. Keep both columns, and never mix them in one search.")
    json.dump({"model": MODEL, "results": results, "pytidb_vs_litellm": same_path},
              open(os.path.join(args.out, "check_bedrock_vs_hosted.json"), "w"))
    print(f"Saved {os.path.join(args.out, 'check_bedrock_vs_hosted.json')}")
    sys.exit()

# Questions first: few, and useful on their own.
questions_path = os.path.join(args.out, "questions_bedrock.json")
questions = [q.strip() for q in open(os.path.join(KIT, "data", "questions.txt"), encoding="utf-8") if q.strip()]
done_questions = json.load(open(questions_path, encoding="utf-8")) if os.path.exists(questions_path) else {}
missing = [q for q in questions if q not in done_questions]
if missing:
    with ThreadPoolExecutor(args.threads) as pool:
        done_questions.update(zip(missing, pool.map(embed, missing)))
    json.dump(done_questions, open(questions_path, "w", encoding="utf-8"), ensure_ascii=False)
print(f"Questions: {len(done_questions)} embedded, saved to {questions_path}")

rows = {}
for r in csv.DictReader(open(os.path.join(KIT, "data", "persons_playground.csv"), newline="", encoding="utf-8")):
    rows[int(r["id"])] = r
parts_dir = os.path.join(args.out, "bedrock_parts")
os.makedirs(parts_dir, exist_ok=True)
done = set()
for part in glob.glob(os.path.join(parts_dir, "part-*.parquet")):
    done.update(pq.read_table(part, columns=["id"])["id"].to_pylist())
ids = sorted(rows)[: args.limit] if args.limit else sorted(rows)
todo = [i for i in ids if i not in done]
chars = sum(len(rows[i]["intro"]) for i in todo)
print(f"Rows: {len(ids)} selected, {len(ids) - len(todo)} already done, {len(todo)} to embed"
      f" (about {chars / 4 / 1e6:.1f} million tokens, assuming 4 characters per token)")

vector_type = pa.list_(pa.float32(), DIMENSIONS)
buffer, start, finished = [], time.time(), 0


def flush():
    global buffer
    if buffer:
        table = pa.table({"id": pa.array([i for i, _ in buffer], pa.int64()),
                          "intro_vec": pa.array([v for _, v in buffer], vector_type)})
        pq.write_table(table, os.path.join(parts_dir, f"part-{time.time_ns()}.parquet"), compression="zstd")
        buffer = []


pool = ThreadPoolExecutor(args.threads)
try:
    futures = {pool.submit(embed, rows[i]["intro"]): i for i in todo}
    for future in as_completed(futures):
        buffer.append((futures[future], future.result()))
        finished += 1
        if len(buffer) >= 1000:
            flush()
        if finished % 1000 == 0 or finished == len(todo):
            rate = finished / (time.time() - start)
            print(f"{finished}/{len(todo)} rows, {rate:.0f} rows/s, {(len(todo) - finished) / rate / 60:.1f} min left",
                  flush=True)
finally:
    flush()
    pool.shutdown(wait=False, cancel_futures=True)

vectors = {}
for part in glob.glob(os.path.join(parts_dir, "part-*.parquet")):
    t = pq.read_table(part)
    vectors.update(zip(t["id"].to_pylist(), t["intro_vec"].to_pylist()))
final = [i for i in sorted(rows) if i in vectors]
out_path = os.path.join(args.out, "persons_bedrock.parquet")
pq.write_table(pa.table({
    "id": pa.array(final, pa.int64()),
    "name": [rows[i]["name"] for i in final],
    "intro": [rows[i]["intro"] for i in final],
    "intro_vec": pa.array([vectors[i] for i in final], vector_type),
}), out_path, compression="zstd")
print(f"Saved {len(final)} of {len(rows)} rows to {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)"
      f" in {time.time() - start:.0f}s")
