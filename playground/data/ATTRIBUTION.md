# Data sources

- `persons_playground.csv` and `.parquet`: 5,114 people. Ids 1 to 3 (Cheese Wang, Hao Huo, Humble BGL) are the lab's own
  sample rows, copied verbatim (no "Name: " prefix) from `rag.py`, `improved_rag.py` and `tidb_rag.py` in
  https://github.com/Icemap/tidb-lab-unify-storage (`main` at bb48062, no license stated), so that the playground
  answers as the lab's own data does. Ids 4 to 9 are people from the AWS and TiDB Immersion Day in Stockholm on
  2026-11-03; their texts were written for this event from the speaker bios on the event page
  (https://www.pingcap.com/event/aws-tidb-immersion-day-stockholm/) and the people's own input (final confirmation on
  2026-11-02).
  Ids 101 to 169 are real public people, described from public knowledge, and ids 1001 to 1036 are fictional, with
  planted details (INC-2041, Lingonberry Systems, 1:42.318, ...); both were written for this demo.
  Ids 100001 to 105000 are synthetic people derived from NVIDIA Nemotron-Personas-USA,
  https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA, licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Changes: one persona field per person, prefixed with the person's
  name and shortened at a clause boundary to at most 255 characters; minors and possessive names left out.
- Vectors (`intro_vec`, `questions_playground.json`, `hosted_sample.json`): Amazon Titan Text Embeddings V2, 1024 dimensions.
  Bedrock's `amazon.titan-embed-text-v2:0` and TiDB Cloud's hosted `tidbcloud_free/amazon/titan-embed-text-v2` gave
  bit-for-bit identical vectors for every row (checked 2026-10-30), so they work with either.
  The vectors of ids 1 to 9 (of the intro alone, as the lab apps embed it) and of the seven questions added with them
  were computed with the hosted model on 2026-10-31; the same method reproduced four stored Bedrock vectors bit for bit.
  `bedrock_embed.py --check` compares them with Bedrock.
