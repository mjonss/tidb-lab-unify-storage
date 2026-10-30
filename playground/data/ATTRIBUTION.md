# Data sources

- `persons_playground.csv` and `.parquet`: 5,105 people. Ids 101 to 169 are real public people, described from public
  knowledge, and ids 1001 to 1036 are fictional, with planted details (INC-2041, Lingonberry Systems, 1:42.318, ...);
  both were written for this demo. Ids 100001 to 105000 are synthetic people derived from NVIDIA Nemotron-Personas-USA,
  https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA, licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Changes: one persona field per person, prefixed with the person's
  name and shortened at a clause boundary to at most 255 characters; minors and possessive names left out.
- Vectors (`intro_vec`, `questions_playground.json`, `hosted_sample.json`): Amazon Titan Text Embeddings V2, 1024 dimensions.
  Bedrock's `amazon.titan-embed-text-v2:0` and TiDB Cloud's hosted `tidbcloud_free/amazon/titan-embed-text-v2` gave
  bit-for-bit identical vectors for every row (checked 2026-10-30), so they work with either.
