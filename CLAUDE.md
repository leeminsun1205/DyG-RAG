# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DyG-RAG is an **event-centric dynamic-graph Retrieval-Augmented Generation** framework for temporal reasoning (arXiv 2507.13396, code: github.com/RingBDStack/DyG-RAG). It is built on top of **nano-graphrag** by Gustavo Ye.

This repository is a **research fork used for the VDS / Viettel Digital Services project** (see Research Context). We switched the project's base method from HippoRAG to DyG-RAG because DyG-RAG already ships the temporal machinery the project needs — explicit per-event timestamps, a time-decayed event graph, timestamp-aware vector search, and Time Chain-of-Thought generation — so it is a stronger substrate for temporal-versioning / conflict-aware retrieval research than a static graph RAG.

Core idea: text → **Dynamic Event Units (DEUs)** (a sentence-level event + a normalized timestamp + involved entities) → an **event graph** whose edges link events by shared entities and temporal proximity → time-aware retrieval (timestamp-weighted vector search + weighted random walk) → **Time-CoT** generation.

When touching indexing or graph construction, **preserve the temporal plumbing**: every event carries a `timestamp` (normalized `YYYY[-MM[-DD]]` or `static`), edges carry `weight` / `time_distance` / shared-entity metadata, and the events vector DB may carry a Fourier timestamp encoding. Do not silently drop timestamps or entity metadata from events or edges.

---

## Current Implementation Status

Distinguishes what **exists in the code today** from what is a **research goal**. Verify in code before relying on a feature.

**Implemented (present today):**

- **Full DyG-RAG pipeline** (upstream): `GraphRAG.insert()` (chunk → LLM event extraction → NER → event merge → event-graph edge building → persist) and `GraphRAG.query(mode="dynamic")` (timestamp-weighted vector search → cross-encoder rerank → weighted random walk → Time-CoT generation). `"dynamic"` is the **only** implemented query mode.
- **Single reproduction driver** for all three temporal-QA datasets: `reproduce/run.py --dataset {complextr,timeqa,tempreason}`. It builds the graph, runs the QA loop with concurrency + retries, saves `results_{dataset}_mode-{mode}_topk-{top_k}.json`, then calls `graphrag.evaluate.run_evaluation` and prints Accuracy/Recall/F1/EM. (The three datasets used identical config + pipeline and differed only in dataset path, so the old per-dataset `complextr.py`/`timeqa.py`/`tempreason.py` were unified into this one file.)
- **OpenAI-API adaptation (ours).** `run.py` runs the LLM through the OpenAI API (or any OpenAI-compatible endpoint) instead of a local vLLM server, configurable by env var: `LLM_MODEL` (default `gpt-4.1-nano`), `OPENAI_BASE_URL` (unset = official OpenAI; set = vLLM/proxy), `OPENAI_API_KEY`, `LOCAL_BGE_PATH` (default `BAAI/bge-m3`, auto-downloads). It forces `temperature=0` for reproducibility. Embedding stays **BGE-M3 local** and reranker/NER stay as in the paper — only the LLM backend changed. The legacy env names (`VLLM_BASE_URL`, `QWEN_BEST`) are still accepted as fallbacks, so the local-vLLM path keeps working without edits. A `--max_questions N` flag limits the run for quick smoke tests.
- **Reproduced ComplexTR baseline (ours), matches the paper.** `gpt-4.1-nano` + BGE-M3 on ComplexTR gave **Accuracy 54.41 / Recall 69.43**, against the paper's Qwen2.5-14B **55.62 / 69.88** — within ~1pp despite the different LLM. The paper's headline metrics are **Accuracy (inclusion) and Recall only**; `evaluate.py` also prints F1/precision/EM but those are near-zero here **by design** (the `dynamic_QA` prompt asks for a verbose answer + justification, so token-precision/EM collapse — not a bug, not comparable to the paper).
- **Bug fixes (ours, in repo code — keep them):**
  - `graphrag/_op.py`: event `context` is coerced to `""` when the LLM emits `"context": null` (lines ~586 and the `_merge_events_then_upsert` lists ~835). Without this, `max(contexts, key=len)` does `len(None)` and the whole insert crashes. Surfaced by gpt-4.1-nano (Qwen rarely emitted null); latent for any model.
  - `graphrag/evaluate.py` (~line 166): `str(results_file).replace('.json', '_eval.json')` — `RESULTS_FILE` is a `Path`, and `Path.replace(a, b)` is a filesystem rename, not a string replace.
- **Kaggle install recipe (ours):** `requirements-kaggle.txt` — a minimal, pinned dependency set that works on the Kaggle Python-3.10 / torch-2.0.0 image. Do **not** `pip install -r requirements.txt` on Kaggle (it pins `umap==0.1.1`, broken on py3.10, and pulls vllm/faiss/colbert/llama-index that DyG-RAG never imports). See Runtime Environment.

**Not implemented yet (project research TODO — no code for these here):**

- **Conflict detection / `superseded` marking.** DyG-RAG models events on a timeline but does **not** detect that a newer event contradicts an older one, nor mark anything superseded. This is the central VDS research goal and must be built.
- **Provenance / reliability-weighted scoring.** Events have a `source_id` (chunk provenance) but retrieval scoring uses only semantic + temporal signals; no reliability/source weighting.
- **Knowledge update / version-aware retrieval & evaluation.** No synthetic "versioned fact" benchmark, no version-pick metric. The reproduction datasets test temporal QA accuracy, not knowledge-update behavior.

**Maintenance rule:** When a code change implements, removes, or materially changes any feature above, update this section **in the same task** — move completed TODOs into "Implemented" and record new limitations. A stale status here is worse than none.

---

## Research Context

This repository is used for the VDS / Viettel Digital Services research project.

**Vietnamese title:** Truy xuất tri thức đa bước nhận biết được mâu thuẫn và phiên bản thời gian

**English working title:** Multi-hop Knowledge Retrieval with Conflict Awareness and Temporal Versioning

The project studies how to extend graph-based RAG so it can handle continuously changing enterprise knowledge. In real internal knowledge bases, regulations, project specs, personnel info, and business rules change over time. A normal RAG or graph-based RAG may retrieve outdated information or mix old and new versions without telling the LLM that multiple versions exist.

The main research direction adds two capabilities to graph-based retrieval:

1. **Temporal awareness**: each event / triple / edge preserves temporal information (timestamp, version, validity period). *DyG-RAG already provides per-event timestamps and a time-decayed event graph — this is largely the substrate, not new work.*
2. **Conflict detection**: when new information is inserted, detect whether it contradicts older information and mark the older one as superseded instead of deleting it. *This is the new work — not in DyG-RAG today.*

During retrieval, the system should prefer newer / more reliable information, but still expose older versions to the LLM when relevant for reasoning or comparison.

---

## Research Goals

1. Reproduce an open-source graph-based RAG baseline (now **DyG-RAG**) on standard temporal-QA benchmarks. *(ComplexTR reproduced; TimeQA/TempReason pending OpenAI adaptation.)*
2. Keep / extend the event-graph schema so each event, node, or edge carries `timestamp` and `provenance`.
3. Implement a lightweight conflict-detection mechanism (rule-based, then possibly an NLI model such as DeBERTa-MNLI) to detect contradictions between old and new events.
4. Mark outdated / contradicted events as `superseded` instead of deleting them.
5. Modify graph propagation / retrieval scoring so edge weights can depend on recency, provenance, and reliability. *(DyG-RAG already does recency via time-decay edges + timestamp-weighted search.)*
6. Build or adapt a temporal / versioned-fact benchmark.
7. Evaluate on both standard temporal QA and the versioned/conflict benchmark.

---

## Expected Outputs

1. Reproducible source code.
2. Augmented temporal / versioned dataset, or scripts to build it.
3. Results on the standard temporal-QA benchmarks (TimeQA / TempReason / ComplexTR).
4. Results on the temporal / conflict-aware benchmark.
5. A technical report on the trade-off between temporal accuracy and insertion/update overhead.
6. Error analysis explaining when conflict detection fails.
7. At least 5 qualitative case studies of success and failure.

---

## Running

The package is imported locally as:

```python
from graphrag import GraphRAG, QueryParam
```

Run scripts **from the repository root**. The reproduce scripts use relative paths (`datasets/...`, `./models`) and `sys.path.insert` to the repo root.

### Prerequisites

1. Download the auxiliary models (cross-encoder + NER) once:
   ```sh
   python models/download.py
   # → models/cross-encoder_ms-marco-TinyBERT-L-2-v2/  and  models/dslim_bert_base_ner/
   ```
2. Download the preprocessed datasets (Google Drive link in `datasets/link.txt`) into:
   ```text
   datasets/{TimeQA,TempReason,ComplexTR}/{Corpus,Question}.json
   ```

### Any dataset via OpenAI API (the adapted path)

```sh
export OPENAI_API_KEY="sk-..."          # real key; or "EMPTY" for a local vLLM
export LLM_MODEL="gpt-4.1-nano"         # any OpenAI chat model (or local model name)
# export OPENAI_BASE_URL="https://..."  # optional: vLLM / proxy endpoint; unset = api.openai.com
export LOCAL_BGE_PATH="BAAI/bge-m3"     # BGE-M3 path or HF id (auto-downloads on GPU)
python reproduce/run.py --dataset complextr      # or timeqa / tempreason
python reproduce/run.py --dataset timeqa --max_questions 20   # quick smoke test
```

### Legacy local-vLLM path (still supported via fallback env names)

```sh
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"   # → used as base_url; api_key defaults to "EMPTY"
export QWEN_BEST="qwen-14b"                        # → used as LLM_MODEL
export LOCAL_BGE_PATH="/path/to/bge-m3"
python reproduce/run.py --dataset timeqa
```

### Minimal examples (single query on `demo/Corpus.json`)

```sh
python examples/openai_all.py            # default GraphRAG = OpenAI embeddings + OpenAI LLM
python examples/local_BGE_local_LLM.py   # BGE-M3 embeddings + local vLLM LLM
```

Notes:

- `run.py` prints progress and ends with `[<dataset>] Evaluation completed! Accuracy: .. | Recall: .. | F1: .. | EM: ..` and writes `results_{dataset}_mode-{mode}_topk-{top_k}.json` (+ `..._eval.json`).
- Defaults: `--mode dynamic` (only mode implemented), `--top_k 20`, `--concurrency 5`, `--max_questions 0` (all). Override on the CLI.
- The headline metrics for these datasets are **Accuracy (inclusion) and Recall**; F1/EM are printed but not paper-comparable with the verbose `dynamic_QA` prompt.

---

## Re-running / Cache Invalidation

Indexing is heavily cached via `enable_llm_cache=True` and persisted storage. State lives in the **working directory** per dataset:

```text
complextr_dir/    timeqa_dir/    tempreason_dir/
```

Each contains the LLM response cache (`kv_store_llm_response_cache.json`), full docs, text chunks, the events vector DB, and the event graph (GraphML). LLM-extraction results are cached here, so re-running after a crash is cheap and does not re-call the API for already-processed chunks.

To force a fully clean rebuild of one dataset:

```sh
rm -rf complextr_dir       # then re-run: python reproduce/run.py --dataset complextr
```

When debugging strange retrieval results, suspect stale cache / a half-written graph from an interrupted insert.

---

## Architecture

DyG-RAG funnels everything through **one dataclass that is both config and orchestrator**: `GraphRAG` in `graphrag/graphrag.py`. (Unlike some forks, there is no separate config object — every tunable is a field on `GraphRAG`. Add new knobs as fields there.)

### `graphrag/graphrag.py` — `GraphRAG`

**Insert flow** (`insert` → `ainsert`):

```text
ainsert(docs)
→ _insert_start()                       # prep storages
→ dedup docs by compute_mdhash_id (doc-)
→ get_chunks()                          # chunk_token_size=1200, overlap=64, cl100k_base
→ extract_events(chunks, dyg_inst, events_vdb, global_config)   # _op.py
    → LLM event extraction (dynamic_event_units prompt) + gleaning
    → BatchNERExtractor tags entities
    → _merge_events_then_upsert (dedup events, pick longest sentence/context)
    → compute_event_relationships_batch → event-graph edges (entity + time-decay)
→ upsert full_docs + text_chunks
→ _insert_done()                        # flush all storages
```

**Query flow** (`query` → `aquery` → `dynamic_query`, the only mode):

```text
dynamic_query(q, param)
→ parse_query_time_and_entities()       # LLM extracts time constraints + entities (time_entity_extraction prompt)
→ events_vdb time-weighted / plain vector search (topk1 candidates)
→ rerank seeds: cross-encoder (default) or BM25 or distance  → et_top_k seeds
→ _random_walk_graph_traversal()        # weighted walk: walk_depth=2, walk_n=3, walk_nodes=5
→ get_nodes_batch() for traversed events
→ build_time_CoT()                       # chronological timeline block (if if_timeline_events)
→ text_chunks.get_by_ids() for source chunks (truncated to max_token_for_text_unit)
→ PROMPTS["dynamic_QA"] (or dynamic_QA_wo_timeline ablation) formatted with {question, events_data, chunks_data}
→ best_model_func(context)               # final answer (cached)
```

If reranking yields no events, the flow still proceeds on whatever survived vector search.

**Key DyG config fields (defaults = upstream / paper values; the paper does not publish the exact numbers):**

| Field | Default | Meaning |
|---|---|---|
| `chunk_token_size` / `chunk_overlap_token_size` | 1200 / 64 | sliding-window chunking (paper) |
| `max_links` | 6 | max edges created per event |
| `decay_rate` | 0.02 | time-decay factor for edge weight |
| `ent_ratio` / `time_ratio` | 0.75 / 0.25 | entity vs time share of edge composite score |
| `ent_factor` | 0.3 | shared-entity weight scaling |
| `time_weight` | 0.1 | timestamp weight in vector search |
| `walk_depth` / `walk_n` / `walk_nodes` | 2 / 3 / 5 | random-walk traversal |
| `enable_ce_rerank` / `ce_model` | True / `cross-encoder/ms-marco-TinyBERT-L-2-v2` | cross-encoder reranker |
| `enable_bm25_reranking` | False | BM25 alternative reranker |
| `enable_timestamp_encoding` / `timestamp_dim` | True / 16 | Fourier timestamp encoding in events VDB |
| `event_extract_max_gleaning` | 1 | extra LLM passes to catch missed events |
| `best/cheap_model_max_token_size` | 32768 (reproduce scripts override to 16384) | LLM context budget |
| `random_seed` | 42 | reproducibility |

**`QueryParam`** (`graphrag/base.py`): `mode="dynamic"`, `top_k=20`, `et_top_k=20`, `topk1=500`, `max_token_for_text_unit=12000`, `response_type`, `time_constraints`, `entities`, `only_need_context`.

---

## Node Types and Storage Stores

Three stores plus the event graph, all persisted under the working directory:

| Attribute | Backend (default) | Holds |
|---|---|---|
| `full_docs` | `JsonKVStorage` | original documents (id `doc-…`) |
| `text_chunks` | `JsonKVStorage` | chunked text (id `chunk-…`) |
| `llm_response_cache` | `JsonKVStorage` | cached LLM responses (keyed by `compute_args_hash`) |
| `events_vdb` | `NanoVectorDBStorage` or `TimestampEnhancedVectorStorage` | event embeddings (id `event-…`), meta `{event_id, timestamp, sentence}` |
| `event_dynamic_graph` | `NetworkXStorage` | event nodes + temporal-proximity edges (GraphML) |

IDs are `compute_mdhash_id` content hashes with a namespace prefix (`doc-`, `chunk-`, `event-`). When `enable_timestamp_encoding=True`, the events VDB uses `TimestampEnhancedVectorStorage`, which concatenates a Fourier encoding of the event timestamp (`timestamp_dim` extra dims) to the semantic vector, enabling `time_weighted_query`.

---

## Pluggable Backends

### LLM / embedding (`graphrag/_llm.py`)

`GraphRAG` takes `best_model_func`, `cheap_model_func`, and `embedding_func` callables. Defaults:

- `best_model_func = gpt_4o_complete` → actually calls **`gpt-3.5-turbo`** (note the misleading name).
- `cheap_model_func = gpt_4o_mini_complete` → `gpt-4o-mini`.
- `embedding_func = openai_embedding` → `text-embedding-3-small` (1536-d).
- `using_azure_openai=True` swaps in the Azure variants; `using_amazon_bedrock=True` swaps in Bedrock.

The OpenAI client (`AsyncOpenAI()`) honors `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment, so an OpenAI-compatible endpoint (vLLM, proxy) works by env alone.

**The reproduce scripts and `examples/local_BGE_local_LLM.py` override all three** with their own `EmbeddingFunc` (BGE-M3 via `SentenceTransformer`) and an `AsyncOpenAI`-based `best_model_func` (pointed at vLLM or OpenAI). `examples/openai_all.py` uses the GraphRAG defaults (pure OpenAI).

### Storage backends (`graphrag/_storage/`)

| File | Class | Default? |
|---|---|---|
| `kv_json.py` | `JsonKVStorage` | yes |
| `gdb_networkx.py` | `NetworkXStorage` | yes (graph) |
| `vdb_nanovectordb.py` | `NanoVectorDBStorage` | yes (vector) |
| `vdb_timestamp.py` | `TimestampEnhancedVectorStorage` | when `enable_timestamp_encoding` |
| `gdb_neo4j.py` | `Neo4jStorage` | optional (needs `neo4j`) |
| `vdb_hnswlib.py` | `HNSWVectorStorage` | optional (needs `hnswlib`) |

Note: `_storage/__init__.py` imports the Neo4j and HNSW classes unconditionally, so `neo4j` and `hnswlib` must be **importable** even though the defaults don't use them (relevant on Kaggle — they're in `requirements-kaggle.txt`).

### Cross-encoder / NER

The cross-encoder (`ce_model`) loads from `models/<ce_model with / → _>` if present, else auto-downloads from HuggingFace, else falls back to BM25 (`enable_bm25_reranking`). The NER model (`ner_model_name`, default `dslim_bert_base_ner`) loads from `model_path` (`./models`). `models/download.py` fetches both.

---

## Prompts

`graphrag/prompt.py` holds a `PROMPTS` dict. Key entries:

| Key | Purpose |
|---|---|
| `dynamic_event_units` | LLM instruction to extract Dynamic Event Units (event sentence + time + entities) from a chunk |
| `event_continue_extraction` / `event_if_loop_extraction` | gleaning: extract more events / decide whether to continue |
| `time_entity_extraction` | extract query time constraints + entities (JSON: `start_time`, `end_time`, `entities`) |
| `dynamic_QA` | main answer-generation prompt: reason over events timeline + chunks. **Asks for a direct answer followed by justification** → verbose output (see metrics caveat) |
| `dynamic_QA_wo_timeline` | ablation: answer from chunks only, strict temporal constraint matching |
| `short_answer` | fallback generation prompt (used only if chunk retrieval errors) |
| `fail_response` | fallback when no answer is possible |

When changing a prompt, state whether it affects event extraction, query parsing, answer generation, or the ablation path. To force short, EM-friendly answers you would edit `dynamic_QA` — but that **diverges from the paper**, whose reported Accuracy/Recall are computed on the verbose output; do not change it without a reason.

---

## Datasets

```text
datasets/{TimeQA,TempReason,ComplexTR}/Corpus.json     # documents
datasets/{TimeQA,TempReason,ComplexTR}/Question.json   # questions + gold answers
```

Preprocessed datasets: Google Drive folder in `datasets/link.txt`. Original sources: TimeQA (wenhuchen/Time-Sensitive-QA), TempReason (DAMO-NLP-SG/TempReason), ComplexTR (HF `tonytan48/complex-tr`).

Corpus entries: `{title, context, ...}` (the reproduce scripts build `f"Title: {title}\nDocument ID: {idx}\n\n{context}"` per doc). `demo/Corpus.json` uses `{doc_id, title, context}` for the examples. Question entries carry `question` and `answer` (gold); ComplexTR answers may be multi-value (comma/pipe/quoted) and `evaluate.py` handles that.

---

## Evaluation

`graphrag/evaluate.py` (`Evaluator`, `run_evaluation`) reads a results JSON (e.g. `results_complextr_mode-dynamic_topk-20.json`), evaluates only `status=="success"` rows, writes `..._eval.json`, and returns `{accuracy, f1, precision, recall, em, avg_query_time}`. Run standalone with `python graphrag/evaluate.py --results-file <file>`.

- `normalize_answer`: lowercase, strip punctuation + articles, collapse whitespace.
- **Accuracy** = inclusion (normalized gold is a substring of the prediction). **This is the paper's headline metric**, together with **Recall** (token overlap). For multi-answer items, accuracy requires *all* golds covered.
- F1 / precision / EM are token-level / exact-match — printed for rigor but **near-zero and not paper-comparable** with the verbose `dynamic_QA` answers.

Do not change evaluation metrics without explicit approval (Experiment Rules).

---

## Experiment Rules

- Do not modify original benchmark datasets directly. Save augmented data as a separate derived dataset.
- Keep original splits unchanged.
- Do not change evaluation metrics without explicit approval.
- Do not silently remove temporal / entity / provenance metadata from events or edges.
- For the future conflict work: do **not** delete old events on conflict — mark them `superseded`.
- Preserve reproducibility: every experiment has a clear command, config, dataset path, model name, and output dir.
- Prefer minimal, well-scoped changes over large refactors.
- If a change affects retrieval, indexing, graph construction, or evaluation, explain the expected impact before editing.

---

## Evaluation Principles

Standard temporal-QA evaluation should answer:

- Does the system reproduce the published DyG-RAG baseline (Accuracy / Recall) on TimeQA / TempReason / ComplexTR?
- Does any modification regress the baseline?

Temporal / conflict evaluation (future) should answer:

- Does the system prefer newer information when old and new facts conflict?
- Can it recognize that multiple versions of a fact exist, and expose them?
- Can it avoid mixing incompatible facts from different time periods?
- When does rule-based / NLI conflict detection fail?

Report when available: Accuracy, Recall, EM, F1, indexing time, retrieval time, conflict-detection overhead, number of detected conflicts, number of superseded events. Include ≥5 qualitative case studies.

---

## Runtime Environment

Local use: read code, small debugging, syntax checks, tiny indexing/retrieval. Large runs go on **Kaggle / a GPU box** (BGE-M3 + NER + cross-encoder want a GPU; `ner_device` defaults to `cuda:0`).

**Kaggle setup (verified working):**

1. `pip install -q -r requirements-kaggle.txt` — pinned, py3.10 / torch-2.0.0-compatible. **Restart the kernel afterwards** (transformers/torch must reload).
   - Do **not** install `requirements.txt` on Kaggle (broken `umap==0.1.1`; pulls vllm/faiss/colbert/llama-index that DyG-RAG never imports).
   - Key pins: `transformers==4.44.2`, `sentence-transformers==3.0.1`, `accelerate==0.33.0`, `tokenizers<0.20` (newer transformers demands torch ≥ 2.4, which Kaggle does not have; `accelerate` is required by `SentenceTransformer` for `device_map`/`low_cpu_mem_usage`). Also `neo4j` + `hnswlib` (imported by `_storage/__init__.py`).
2. Load the OpenAI key from a Kaggle Secret into the env (Kaggle does not auto-export secrets):
   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
   ```
3. `python models/download.py` **after** deps are installed (cross-encoder needs `sentence-transformers`; if it 's still missing, `CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2").save("models/cross-encoder_ms-marco-TinyBERT-L-2-v2")`).
4. Run from the repo root: `python reproduce/run.py --dataset complextr`.

Common failure modes (and where they come from): missing `accelerate` → BGE-M3 load fails; newer `transformers` → "PyTorch ≥ 2.4 required" disables torch; broken notebook-kernel imports (`typing_extensions`) → run eval via `!python -c "..."` subprocess instead of the kernel.

Relevant env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`, `LOCAL_BGE_PATH` (OpenAI path); `VLLM_BASE_URL`, `QWEN_BEST`, `LOCAL_BGE_PATH` (legacy vLLM path); `HF_HOME`, `CUDA_VISIBLE_DEVICES`.

---

## Development Preferences

- Explain the root cause before proposing code changes.
- Explain the plan and name affected files / functions before editing.
- Prefer simple, reproducible implementations; avoid unrelated refactoring.
- Do not install new packages unless necessary; if you add one, say why and where it's used.
- Keep code readable for undergraduate thesis work.
- Prioritize correctness and reproducibility before optimization.
- When debugging, identify whether the problem is environment, data format, cache, model backend, or code logic.
- When unsure, inspect the relevant code before making broad claims.
- For large changes, propose a small first step that can be tested quickly.

---

## References

1. **DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning** (arXiv 2507.13396; code: github.com/RingBDStack/DyG-RAG). This repository. Event-centric: Dynamic Event Units + event graph + time-aware traversal + Time Chain-of-Thought.
2. **nano-graphrag** by Gustavo Ye (github.com/gusye1234/nano-graphrag) — the simple GraphRAG implementation DyG-RAG is built on.
3. **IA-RAG: Interval-Algebra–Driven Temporal Reasoning for Dynamic Knowledge Retrieval** (arXiv 2606.06044) — uses the same TimeQA/TempReason/ComplexTR benchmarks; comparable baseline.
4. **HippoRAG / HippoRAG 2** (OSU-NLP-Group) — the project's previous base method.
5. **It's High Time: A Survey of Temporal Information Retrieval and Question Answering** (arXiv 2505.20243) — temporal intent, time normalization, event ordering, recency/conflict.
6. TempLAMA / TimeQA / TempReason — temporal QA benchmarks.
7. Microsoft GraphRAG; MuSiQue / 2WikiMultiHopQA / HotpotQA (static multi-hop references).

Do not overfit the implementation to one paper. Use these to guide design, experiments, and comparison.
