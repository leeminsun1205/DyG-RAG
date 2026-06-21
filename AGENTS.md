# AGENTS.md

This file is the canonical working guide for Codex and other coding agents in this repository. The historical `CLAUDE.md` guidance has been migrated here; do not recreate a separate agent guide unless the user asks.

## Project Overview

DyG-RAG is a Python research repository for Dynamic Graph Retrieval-Augmented Generation with event-centric temporal reasoning. It is built on nano-graphrag and implements the DyG-RAG paper pipeline: text -> Dynamic Event Units (DEUs) -> timestamp/entity-aware event graph -> time-aware retrieval -> Time-CoT answer generation.

This fork is used for the VDS / Viettel Digital Services research project on multi-hop knowledge retrieval with conflict awareness and temporal versioning. Prefer correctness, reproducibility, and small scoped changes over broad cleanup.

Core idea:

```text
documents
-> chunks
-> DEUs: event sentence + normalized timestamp + context + entities + source_id
-> dynamic event graph: shared entities + temporal proximity
-> timestamp-aware vector retrieval + reranking + graph traversal
-> Time-CoT / optional Version-CoT answer generation
```

When touching indexing, retrieval, graph construction, prompts, or evaluation, preserve temporal plumbing: `timestamp`, entities, `source_id`, edge `weight`, `time_distance`, shared-entity metadata, and provenance-like fields.

## Current Implementation Status

Keep this section current when changing feature behavior.

Implemented today:

- Full DyG-RAG insert/query pipeline in `graphrag/graphrag.py`. `mode="dynamic"` is the only implemented query mode.
- Unified benchmark driver: `reproduce/run.py --dataset {complextr,timeqa,tempreason}`. It builds/loads the index, runs QA with concurrency and retries, saves `results_{dataset}_mode-{mode}_topk-{top_k}.json`, then evaluates.
- OpenAI-compatible reproduction path. `run.py` uses `OPENAI_API_KEY`, `LLM_MODEL`, `OPENAI_BASE_URL`, and `LOCAL_BGE_PATH`; legacy `VLLM_BASE_URL` and `QWEN_BEST` still work. It sets temperature to 0 for reproducibility. BGE-M3, cross-encoder, and NER remain local/paper-aligned.
- ComplexTR baseline reproduced with `gpt-4.1-nano` + BGE-M3: Accuracy 54.41 / Recall 69.43, close to the paper's Qwen2.5-14B 55.62 / 69.88. Treat this as historical project context, not a universal guarantee for future environments.
- Bug fixes to keep:
  - `graphrag/_op.py`: coerce LLM-emitted `context: null` to `""`, including merge paths, so `max(..., key=len)` cannot crash on `None`.
  - `graphrag/evaluate.py`: use `str(results_file).replace(".json", "_eval.json")`; `Path.replace()` is filesystem rename, not string replace.
- Kaggle-compatible dependency path: `requirements-kaggle.txt`. Do not assume full `requirements.txt` is safe on Kaggle.
- Conflict-aware Version-CoT: `graphrag/versioning.py`, `GraphRAG.enable_version_cot=False`, and `reproduce/run.py --version-cot`. Default off means baseline behavior should remain unchanged. When on, it makes one extra LLM call over retrieved events, extracts transient `(subject, attribute, value, validity interval)` facts, renders version timelines, and prepends them to the answer context. Output files are suffixed `_vcot`.
- Version-CoT v1 is query-time and additive: it highlights the row valid at the asked time but should not narrow away otherwise useful answer details. It marks `superseded` for single-valued conflicts; currently only `spouse` is treated single-valued. Concurrent positions/employers/etc. are multi-valued.
- Version-CoT seed expansion: `GraphRAG.enable_version_cot_seed_events=False` and `reproduce/run.py --version-cot-seeds`. Default off means both baseline and `--version-cot` v1 stay unchanged. When on, Version-CoT reads the union of graph-traversed events plus reranked seed events, deduplicated by event ID/sentence. This changes only the Version-CoT context input, not seed selection, traversal, source chunk retrieval, or evaluation. It implies `--version-cot` in the runner and output files are suffixed `_vcot-seeds`.
- Interval event metadata: `GraphRAG.enable_interval_events=False` and `reproduce/run.py --interval-events`. Default off means baseline behavior should remain unchanged. When on, extraction stores rule-based `start_time`, `end_time`, `time_fuzzy`, and `time_expression` fields on event nodes/vector metadata while preserving the original `timestamp` and event IDs. Output files are suffixed `_interval-events`. This is schema groundwork for interval-aware retrieval.
- Interval-aware reranking: `GraphRAG.enable_interval_rerank=False` and `reproduce/run.py --interval-rerank`. Default off means baseline behavior should remain unchanged. When on, seed reranking blends the existing cross-encoder/BM25+entity score with query-window interval relevance using `interval_rerank_weight` (default 0.2). It adds no LLM calls and implies `--interval-events` in the runner. Output files are suffixed `_interval-rerank`.
- Allen edge metadata: `GraphRAG.enable_allen_edges=False` and `reproduce/run.py --allen-edges`. Default off means baseline behavior should remain unchanged. When on, graph construction stores Allen interval relation metadata (`allen_relation`, inverse relation, canonical endpoint IDs, gap/overlap days) on event graph edges without changing edge scoring, traversal, prompts, or evaluation. It implies `--interval-events` in the runner and output files are suffixed `_allen-edges`. This is schema groundwork for Allen-guided traversal.
- Allen-guided traversal: `GraphRAG.enable_allen_traversal=False` and `reproduce/run.py --allen-traversal`. Default off means baseline behavior should remain unchanged. When on, random-walk graph traversal keeps the existing edge topology but biases edge weights using Allen relation metadata and neighbor interval relevance to the parsed query time. It implies `--allen-edges` and `--interval-events` in the runner and output files are suffixed `_allen-traversal`.

Recent full ComplexTR ablation with `gemini-2.5-flash-lite`:

| Run | Accuracy | Recall | Notes |
|---|---:|---:|---|
| baseline | 72.04 | 82.28 | no feature flags |
| `--version-cot` | 74.47 | 83.96 | best current run; main gain source |
| `--interval-rerank` | 72.04 | 82.49 | negligible accuracy gain, slight recall gain |
| `--version-cot --interval-rerank` | 73.25 | 83.48 | worse than Version-CoT alone; do not assume feature stacking helps |

Treat `--interval-rerank` as experimental. Do not recommend it as the default best setting unless later ablations overturn this result. `--version-cot-seeds` is newly implemented and still needs ablation results before being treated as a default recommendation.

Not implemented yet:

- Persistent insert-time conflict detection and persistent `superseded` flags on graph nodes/edges.
- NLI-based contradiction detection.
- Provenance/reliability-weighted scoring.
- Synthetic or derived versioned-fact benchmark and version-pick metric.
- IA-RAG-style interval-native IEUs, Sub-graph Time Tightening, or Thematic Forest. The current interval metadata/rerank/Allen-edge/traversal features are lightweight/backward-compatible, not full IEU replacement.

## Research Context and Goals

Vietnamese title: Truy xuat tri thuc da buoc nhan biet duoc mau thuan va phien ban thoi gian.

English working title: Multi-hop Knowledge Retrieval with Conflict Awareness and Temporal Versioning.

The project studies continuously changing enterprise knowledge. Regulations, specs, personnel data, and business rules can change over time; ordinary RAG may retrieve outdated information or mix old/new versions without exposing the conflict.

Research goals:

- Reproduce DyG-RAG on standard temporal-QA benchmarks.
- Preserve and extend event/edge temporal metadata and provenance.
- Add conflict/version awareness without deleting old facts; mark old facts `superseded` where appropriate.
- Prefer newer or more reliable information during retrieval while still exposing older versions when relevant.
- Build or adapt a temporal/versioned-fact benchmark.
- Report accuracy, recall, EM, F1, indexing/retrieval time, conflict overhead, detected conflicts, superseded events, and qualitative cases when available.

Expected outputs:

- Reproducible source code.
- Augmented temporal/versioned dataset or build scripts.
- Results on TimeQA, TempReason, ComplexTR.
- Results on temporal/conflict-aware benchmarks.
- Technical report on temporal accuracy vs insertion/update overhead.
- Error analysis and at least five success/failure case studies.

## Repository Structure

- `graphrag/`: main DyG-RAG package.
- `graphrag/_storage/`: JSON KV, NetworkX GraphML, NanoVectorDB, timestamp vector storage, optional Neo4j/HNSW backends.
- `reproduce/`: benchmark reproduction driver and utilities.
- `examples/`: runnable examples using `demo/Corpus.json`.
- `demo/`: small demo corpus.
- `datasets/`: dataset link plus downloaded benchmark datasets when present.
- `models/`: model download helper plus downloaded auxiliary models when present.
- `docs/`, `figs/`: documentation and figures.
- `experiments/`: experimental/prototype research code.
- `results_*.json`, `*_eval.json`: experiment outputs.

Important files:

- `README.md`: high-level setup and examples.
- `reproduce/README.md`: benchmark reproduction notes.
- `reproduce/run.py`: main benchmark driver.
- `reproduce/diff_results.py`: baseline vs Version-CoT A/B helper.
- `graphrag/graphrag.py`: GraphRAG config, insert flow, query flow.
- `graphrag/base.py`: `QueryParam` and storage base interfaces.
- `graphrag/_op.py`: event extraction, merging, graph construction helpers.
- `graphrag/prompt.py`: extraction, query parsing, QA prompts.
- `graphrag/evaluate.py`: evaluation metrics and `*_eval.json` writer.
- `graphrag/versioning.py`: optional query-time Version-CoT logic.
- `models/download.py`: downloads cross-encoder and NER models.
- `requirements.txt`: full local dependency file.
- `requirements-kaggle.txt`: minimal Kaggle-compatible dependency file.

## Architecture

`GraphRAG` in `graphrag/graphrag.py` is both config and orchestrator. Add new knobs as `GraphRAG` fields unless an existing local pattern says otherwise.

Insert flow:

```text
insert/ainsert(docs)
-> _insert_start()
-> dedup docs by compute_mdhash_id("doc-")
-> get_chunks(): chunk_token_size=1200, overlap=64, cl100k_base
-> extract_events()
   -> LLM event extraction with dynamic_event_units prompt + gleaning
   -> BatchNERExtractor tags entities
   -> _merge_events_then_upsert()
   -> compute_event_relationships_batch(): entity + time-decay edges
   -> events_vdb.upsert()
-> full_docs/text_chunks upsert
-> _insert_done()
```

Query flow:

```text
dynamic_query(q, param)
-> parse_query_time_and_entities()
-> events_vdb time_weighted_query or regular vector search
-> cross-encoder rerank, BM25 fallback, or distance sort
-> et_top_k seed events
-> _random_walk_graph_traversal()
-> get full event nodes
-> source_id frequency -> text_chunks.get_by_ids()
-> build_time_CoT()
-> optional build_version_section() if enable_version_cot
-> PROMPTS["dynamic_QA"] or dynamic_QA_wo_timeline
-> best_model_func()
```

Key config defaults:

| Field | Default | Meaning |
|---|---:|---|
| `chunk_token_size` / `chunk_overlap_token_size` | 1200 / 64 | sliding-window chunking |
| `max_links` | 6 | max event graph edges per event |
| `decay_rate` | 0.02 | time-decay factor |
| `ent_ratio` / `time_ratio` | 0.75 / 0.25 | entity vs time edge score ratio |
| `time_weight` | 0.1 | timestamp vector search weight |
| `walk_depth` / `walk_n` / `walk_nodes` | 2 / 3 / 5 | graph traversal |
| `enable_ce_rerank` | True | cross-encoder reranking |
| `enable_timestamp_encoding` / `timestamp_dim` | True / 16 | Fourier timestamp encoding |
| `event_extract_max_gleaning` | 1 | extra extraction pass count |
| `enable_version_cot` | False | optional Version-CoT; off = baseline |
| `enable_version_cot_seed_events` | False | optional seed-event expansion for Version-CoT; off = Version-CoT v1 |
| `enable_interval_events` | False | optional rule-based interval metadata on events; off = baseline |
| `enable_allen_edges` | False | optional Allen interval relation metadata on event graph edges; off = baseline |
| `enable_allen_traversal` / `allen_traversal_weight` | False / 0.3 | optional Allen/time-biased graph traversal; off = baseline |
| `enable_interval_rerank` / `interval_rerank_weight` | False / 0.2 | optional query-window interval relevance blended into seed reranking |

`QueryParam` lives in `graphrag/base.py`: `mode`, `top_k`, `et_top_k`, `topk1`, `max_token_for_text_unit`, `time_constraints`, `entities`, etc.

## Storage and IDs

Stores under each working directory:

- `full_docs`: `JsonKVStorage`, document IDs `doc-*`.
- `text_chunks`: `JsonKVStorage`, chunk IDs `chunk-*`.
- `llm_response_cache`: `JsonKVStorage`, keyed by `compute_args_hash`.
- `events_vdb`: `NanoVectorDBStorage` or `TimestampEnhancedVectorStorage`, event IDs `event-*`.
- `event_dynamic_graph`: `NetworkXStorage`, GraphML event graph.

When `enable_timestamp_encoding=True`, `TimestampEnhancedVectorStorage` concatenates a Fourier timestamp vector to semantic embeddings and supports `time_weighted_query`.

## Prompts and Metrics

Key prompt names in `graphrag/prompt.py`:

- `dynamic_event_units`: DEU extraction.
- `event_continue_extraction`, `event_if_loop_extraction`: gleaning.
- `time_entity_extraction`: query temporal constraints and entities.
- `dynamic_QA`: main answer generation over timeline + chunks.
- `dynamic_QA_wo_timeline`: ablation path without timeline events.
- `short_answer`: fallback.

Changing prompts can alter reported results. State whether a prompt change affects event extraction, query parsing, answer generation, or an ablation path.

Evaluation in `graphrag/evaluate.py`:

- Headline paper-comparable metrics are Accuracy and Recall.
- Accuracy is normalized inclusion: gold answer appears in prediction. Multi-answer accuracy requires all golds covered.
- F1, precision, and EM are printed for rigor but are often low with verbose `dynamic_QA` answers and are not directly paper-comparable.
- Do not change metrics without explicit user approval.

## Running

Run from the repository root.

Local setup:

```bash
conda create -n dygrag python=3.10
conda activate dygrag
pip install -r requirements.txt
python models/download.py
```

Benchmark datasets should be placed as:

```text
datasets/TimeQA/Corpus.json
datasets/TimeQA/Question.json
datasets/TempReason/Corpus.json
datasets/TempReason/Question.json
datasets/ComplexTR/Corpus.json
datasets/ComplexTR/Question.json
```

OpenAI-compatible benchmark path:

```bash
export OPENAI_API_KEY="sk-..."      # or "EMPTY" for local OpenAI-compatible server
export LLM_MODEL="gpt-4.1-nano"
export LOCAL_BGE_PATH="BAAI/bge-m3"
# optional:
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
python reproduce/run.py --dataset complextr
python reproduce/run.py --dataset timeqa --max_questions 20
# optional interval metadata ablation; requires rebuild to populate graph nodes:
python reproduce/run.py --dataset complextr --interval-events
# optional interval-aware rerank; implies interval metadata and requires rebuild for a fair run:
python reproduce/run.py --dataset complextr --interval-rerank
```

Legacy local-vLLM fallback:

```bash
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export QWEN_BEST="qwen-14b"
export LOCAL_BGE_PATH="/path/to/bge-m3"
python reproduce/run.py --dataset timeqa
```

Examples:

```bash
python examples/openai_all.py
python examples/local_BGE_local_LLM.py
```

Evaluate an existing result:

```bash
python graphrag/evaluate.py --results-file results_complextr_mode-dynamic_topk-20.json
```

Syntax-only check:

```bash
python -m py_compile path/to/file.py
```

## Cache and Working Directories

Benchmark state lives in:

```text
complextr_dir/
timeqa_dir/
tempreason_dir/
work_dir/
dyg_rag_cache_*/
```

These contain docs, chunks, LLM cache, event vector DB, and graph. Re-running after a crash may reuse cached extraction. If retrieval looks strange, suspect stale cache or half-written graph. New index-time features such as `--interval-events`, `--allen-edges`, `--allen-traversal`, and `--interval-rerank` require a clean rebuild, otherwise existing graph nodes/edges will not have interval metadata and rerank/traversal will mostly fall back to timestamps.

Clean rebuild example:

```bash
rm -rf complextr_dir
python reproduce/run.py --dataset complextr
```

Do not remove working/cache dirs unless explicitly asked.

## Runtime Notes

Local CPU use is suitable for reading code, syntax checks, small examples, and tiny debugging. Large benchmark runs generally need GPU resources because BGE-M3, NER, and cross-encoder are heavy. `ner_device` defaults to `cuda:0`.

Kaggle notes:

- Use `pip install -q -r requirements-kaggle.txt`, then restart the kernel.
- Do not use full `requirements.txt` on Kaggle unless explicitly tested.
- Kaggle secrets must be manually exported to `OPENAI_API_KEY`.
- Common failures: missing `accelerate`, too-new `transformers` requiring newer torch, missing `neo4j`/`hnswlib` imports from `_storage/__init__.py`.

## Experiment and Feature Rules

- Baseline behavior must remain unchanged unless the user explicitly asks to change it.
- New research features must be additive, flag-gated, and ablation-friendly. Each feature should have its own CLI/config flag where practical.
- Result filenames for non-baseline features must include a suffix so baseline outputs are not overwritten.
- Do not overfit to one dataset or one observed result file.
- Record command, dataset, model, config/flags, working dir, and output path for benchmark experiments.
- Do not modify original benchmark datasets directly. Save derived/augmented data separately.
- Do not silently remove temporal/entity/provenance metadata.
- For conflict/version work, do not delete old facts on conflict; mark them `superseded`.
- If a change affects retrieval, indexing, graph construction, prompts, or evaluation, explain expected impact before editing.

Planned IA-RAG-inspired integration order:

1. Implemented: add backward-compatible interval metadata (`start_time`, `end_time`, `time_fuzzy`, `time_expression`) to events behind `--interval-events`.
2. Implemented: add interval-aware seed reranking behind `--interval-rerank`.
3. Implemented: add Allen relation edge metadata behind `--allen-edges`.
4. Implemented: add Allen-guided traversal behind `--allen-traversal`.
5. Extend fuzzy interval heuristics behind a flag if benchmark evidence supports it.

Do not implement a full IA-RAG Thematic Forest, LLM-heavy IEU deduplication, or full Sub-graph Time Tightening unless explicitly requested.

## Coding Rules for Agents

- Explain the plan and affected files before editing.
- Inspect relevant code before making claims.
- Keep changes minimal and scoped to the requested task.
- Do not refactor unrelated code.
- Preserve public behavior unless the task explicitly asks to change it.
- Prefer existing patterns and helper APIs.
- Do not add dependencies unless necessary and explicitly justified.
- Keep code readable for research/thesis work.
- If unsure about a repo-specific fact, write `to be confirmed` rather than guessing.
- For large changes, propose or implement a small first step that can be tested quickly.
- When debugging, identify whether the issue is environment, data format, cache, model backend, or code logic.

Before editing, explain:

1. what was learned from inspection,
2. what will change,
3. what will not change,
4. how the change will be verified.

## Files and Folders Agents Must Not Touch

Unless the user explicitly requests it, do not modify:

- `datasets/` contents, except reading `datasets/link.txt`.
- `demo/Corpus.json`.
- downloaded model directories under `models/`.
- `results_*.json` and `*_eval.json`.
- benchmark working/cache dirs such as `complextr_dir/`, `timeqa_dir/`, `tempreason_dir/`, `work_dir/`, and `dyg_rag_cache_*/`.
- dependency files: `requirements.txt`, `requirements-kaggle.txt`.
- notebooks, experiment outputs, generated caches, downloaded artifacts.

`CLAUDE.md` has been migrated into this file and should remain deleted unless the user asks to restore it.

## Verification

Choose the lightest verification that matches the change:

- Documentation-only changes: inspect Markdown and run `git diff -- AGENTS.md`.
- Python changes: run `python -m py_compile` on touched files.
- Pipeline changes: run `python reproduce/run.py --dataset <name> --max_questions N` only when dependencies, datasets, models, and API credentials are available.
- Evaluation changes: test with an existing result JSON and compare generated `*_eval.json`.

If verification cannot run because datasets, models, GPU, network, or API keys are missing, state that clearly.

## References

- DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning, arXiv 2507.13396, github.com/RingBDStack/DyG-RAG.
- nano-graphrag by Gustavo Ye.
- IA-RAG: Interval-Algebra-Driven Temporal Reasoning for Dynamic Knowledge Retrieval, arXiv 2606.06044.
- HippoRAG / HippoRAG 2: previous candidate base method for the project.
- It's High Time: A Survey of Temporal Information Retrieval and Question Answering, arXiv 2505.20243.
- TimeQA, TempReason, ComplexTR, TempLAMA.
- Microsoft GraphRAG, MuSiQue, 2WikiMultiHopQA, HotpotQA.

Use these references to guide design and comparison, but do not overfit implementation decisions to one paper.
