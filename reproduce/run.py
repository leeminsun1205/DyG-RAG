from __future__ import annotations

import warnings
# Silence two harmless third-party warnings printed before/while loading models:
warnings.filterwarnings("ignore", message="A NumPy version")            # scipy vs numpy 1.23.5
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")  # torch loading BGE-M3

import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, time, asyncio, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import aiohttp.client_exceptions
import torch

from graphrag import GraphRAG, QueryParam
from graphrag.base import BaseKVStorage
from graphrag._utils import compute_args_hash, logger

import tiktoken
tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logging.getLogger("DyG-RAG").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Single, parameterized reproduction driver for all three temporal-QA
# benchmarks. The three datasets used identical config + pipeline and differed
# only in dataset path + working dir, so they are unified here:
#
#   python reproduce/run.py --dataset complextr      # or timeqa / tempreason
#
# Faithful to the paper except the LLM backend (served via the OpenAI API or an
# OpenAI-compatible endpoint instead of a local vLLM server). Embedding (BGE-M3),
# reranker (TinyBERT), and NER (dslim/bert-base-NER) are unchanged.
#
# Configure the LLM via environment variables:
#   export OPENAI_API_KEY="sk-..."        # real key (Kaggle secret); "EMPTY" for vLLM
#   export LLM_MODEL="gpt-4.1-nano"       # any OpenAI chat model (or local model name)
#   export OPENAI_BASE_URL="..."          # optional; unset = api.openai.com (set for vLLM)
#   export LOCAL_BGE_PATH="BAAI/bge-m3"   # BGE-M3 local path or HF id (auto-downloads)
# ---------------------------------------------------------------------------

# dataset key -> folder label under datasets/<Label>/{Corpus,Question}.json
DATASETS = {
    "complextr":  "ComplexTR",
    "timeqa":     "TimeQA",
    "tempreason": "TempReason",
}

################################################################################
# 0. Configuration (env-driven; shared by all datasets)
################################################################################
# --- LLM model name (accept LLM_MODEL, fall back to legacy QWEN_BEST) ---------
BEST_MODEL_NAME = os.getenv("LLM_MODEL") or os.getenv("QWEN_BEST") or "gpt-4.1-nano"

# --- Base URL: None => official OpenAI endpoint; set => custom/VLLM endpoint ---
# Accept OPENAI_BASE_URL (preferred) or legacy VLLM_BASE_URL. Empty string => None.
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL") or None

# --- API key: real OpenAI key for the API path, or "EMPTY" for a local server -
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or "EMPTY"

# --- Embedding: BGE-M3 as in the paper. Accepts a local path or a HF model id --
LOCAL_BGE_PATH = os.getenv("LOCAL_BGE_PATH") or "BAAI/bge-m3"

# Query retry settings (transient endpoint errors)
MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds

################################################################################
# 1. Embedding function (BGE-M3, local)
################################################################################
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model: SentenceTransformer

    async def __call__(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        encode = lambda: self.model.encode(
            texts,
            batch_size=32,  # Smaller batch size to help with memory issues
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        if loop.is_running():
            return await loop.run_in_executor(None, encode)
        else:
            return encode()

    # Make the model not serializable for GraphRAG initialization
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state

    # Restore model reference during deserialization (will be None, but structure preserved)
    def __setstate__(self, state):
        self.__dict__.update(state)

def get_bge_embedding_func() -> EmbeddingFunc:
    gpu_count = torch.cuda.device_count()
    using_cuda = gpu_count > 0
    device = "cuda" if using_cuda else "cpu"

    model_kwargs = {}
    if gpu_count > 1:
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    st_model = SentenceTransformer(
        LOCAL_BGE_PATH,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )

    return EmbeddingFunc(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        max_token_size=8192,
        model=st_model,
    )

################################################################################
# 2. LLM call function (with cache)
################################################################################
# --- Rate / timeout control for the LLM endpoint ----------------------------
# The proxy is hard-capped at LLM_RPM requests/minute. With best_model_max_async
# concurrent callers (16) plus gleaning, the burst rate blows past that cap and
# the proxy stalls held-over requests -> the insert freezes ("+10% then stuck").
# We pace request *starts* to <= LLM_RPM/min and give each request a timeout +
# retries so a stalled/429'd call recovers instead of wedging a slot forever.
LLM_RPM = float(os.getenv("LLM_RPM") or 60)        # proxy hard cap (requests/min)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT") or 120)
_MIN_INTERVAL = 60.0 / LLM_RPM if LLM_RPM > 0 else 0.0

_async_client: AsyncOpenAI | None = None
_rate_lock = asyncio.Lock()
_last_call_ts = 0.0

async def _rate_limit() -> None:
    """Space consecutive request starts >= _MIN_INTERVAL apart (global pacing)."""
    global _last_call_ts
    if _MIN_INTERVAL <= 0:
        return
    async with _rate_lock:
        now = asyncio.get_event_loop().time()
        wait = _last_call_ts + _MIN_INTERVAL - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_ts = asyncio.get_event_loop().time()

def _build_async_client() -> AsyncOpenAI:
    # base_url=None -> official OpenAI endpoint; set -> VLLM/custom endpoint.
    # Reuse one client; bound each request so a stalled connection can't wedge a slot.
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=LLM_TIMEOUT,
            max_retries=5,  # SDK backs off on 429/timeout (honors Retry-After)
        )
    return _async_client

async def _chat_completion(model: str, messages: list[dict[str, str]], **kwargs) -> str:
    client = _build_async_client()
    kwargs.setdefault("temperature", 0)  # deterministic / reproducible (paper used vLLM default)
    await _rate_limit()
    response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content

async def _llm_with_cache(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    hashing_kv: BaseKVStorage | None = None,
    **kwargs,
) -> str:
    """General LLM wrapper, supporting GraphRAG cache interface."""
    history_messages = history_messages or []
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history_messages)
    msgs.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, msgs)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    answer = await _chat_completion(model=model, messages=msgs, **kwargs)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": answer, "model": model}})
        await hashing_kv.index_done_callback()
    return answer

async def best_model_func(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str, str]] | None = None, **kwargs) -> str:
    return await _llm_with_cache(
        prompt,
        model=BEST_MODEL_NAME,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

################################################################################
# 3. Corpus loading + insertion
################################################################################
def load_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Warning: Unable to parse line: {line[:50]}... Error: {e}")
            return data

def insert_corpus(graph_rag, corpus_file: Path) -> int:
    assert corpus_file.exists(), f"{corpus_file} not found"
    start = time.time()
    logger.info(f"Starting to load documents from {corpus_file}...")
    processed_docs = 0
    try:
        corpus_data = load_json_file(corpus_file)
        total_docs = len(corpus_data)
        logger.info(f"Successfully loaded {total_docs} documents")

        all_docs = []
        for idx, obj in enumerate(tqdm(corpus_data, desc="Processing documents", total=total_docs)):
            enriched_content = f"Title: {obj['title']}\nDocument ID: {idx}\n\n{obj['context']}"
            all_docs.append(enriched_content)

        logger.info(f"Starting to insert all {len(all_docs)} documents into GraphRAG...")
        start_process_time = time.time()
        try:
            graph_rag.insert(all_docs)
            process_time = time.time() - start_process_time
            logger.info(f"All documents processed! Time: {process_time:.1f}s, avg/doc: {process_time/len(all_docs):.2f}s")
            processed_docs = len(all_docs)
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            import traceback
            traceback.print_exc()
            processed_docs = 0

        logger.info(f"Successfully processed: {processed_docs}/{total_docs} documents")
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        import traceback
        traceback.print_exc()

    # Ensure final persistence (GraphRAG has no persist method; call the async done hook)
    logger.info("Executing final persistence...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(graph_rag._insert_done())
    logger.info(f"Insertion done. Total time {time.time() - start:.1f}s")
    return processed_docs

################################################################################
# 4. Query test set and save results
################################################################################
TIMEOUT_EXCEPTIONS = (
    asyncio.TimeoutError,
    aiohttp.client_exceptions.ClientConnectorError,
    aiohttp.client_exceptions.ServerTimeoutError,
    aiohttp.client_exceptions.ClientOSError,
    ConnectionRefusedError,
    ConnectionError,
    TimeoutError,
)

async def process_single_query(question_obj, query_idx, graph_rag, mode, top_k):
    """Process a single query and return the result"""
    try:
        if "question" not in question_obj:
            logger.error(f"Query {query_idx} missing question field: {str(question_obj)[:100]}...")
            return {
                "question_id": question_obj.get("id", f"q{query_idx}"),
                "error": "Missing question field",
                "status": "invalid_format"
            }

        question_id = question_obj.get("id", f"q{query_idx}")
        question_text = question_obj["question"]
        logger.info(f"Processing question {question_id}: {question_text[:50]}...")

        query_param = QueryParam(
            top_k=top_k,
            mode=mode,
            response_type="Short Answer or Direct Response"
        )
        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                result = await graph_rag.aquery(question_text, param=query_param)
                query_time = time.time() - start_time
                break
            except TIMEOUT_EXCEPTIONS as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Query {question_id} timed out, retrying ({attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Query {question_id} failed after {MAX_RETRIES} attempts: {str(e)}")
                    return {
                        "question_id": question_id,
                        "question": question_text,
                        "answer": None,
                        "error": str(e),
                        "query_time": time.time() - start_time,
                        "status": "failed"
                    }

        return {
            "question_id": question_id,
            "question": question_text,
            "answer": result,
            "query_time": query_time,
            "status": "success",
            "golden_answer": question_obj.get("answer", "")
        }
    except Exception as e:
        logger.error(f"Error processing question {query_idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "question_id": question_obj.get("id", f"q{query_idx}"),
            "question": question_obj.get("question", ""),
            "answer": None,
            "error": str(e),
            "status": "exception",
            "traceback": traceback.format_exc()
        }

async def process_queries(graph_rag, label, questions_file: Path, results_file: Path,
                          mode, top_k, concurrency, max_questions):
    logger.info(f"Loading {label} questions from {questions_file}...")
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}")
        return []

    if max_questions and max_questions > 0:
        questions = questions[:max_questions]
        logger.info(f"Limiting to first {len(questions)} questions (--max_questions)")

    total_questions = len(questions)
    logger.info(f"Successfully loaded {total_questions} questions")
    valid_questions = sum(1 for q in questions if "question" in q and "answer" in q)
    if valid_questions < total_questions:
        logger.warning(f"{total_questions - valid_questions} questions missing required fields!")

    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_process(question, idx):
        async with semaphore:
            return await process_single_query(question, idx, graph_rag, mode, top_k)

    tasks = [bounded_process(question, i) for i, question in enumerate(questions)]

    def _dump(extra_meta=None):
        meta = {
            "dataset": label,
            "total_questions": total_questions,
            "completed_questions": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_mode": mode,
            "query_top_k": top_k,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({"metadata": meta, "results": results}, f, ensure_ascii=False, indent=2)

    start_time = time.time()
    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {label} queries"):
        results.append(await completed_task)
        if len(results) % 5 == 0:
            _dump()

    _dump({
        "total_time": time.time() - start_time,
        "avg_time_per_question": (time.time() - start_time) / len(results) if results else 0,
    })

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    exception_count = sum(1 for r in results if r["status"] == "exception")
    logger.info(f"{label} query processing completed!")
    logger.info(f"Total: {total_questions}, Success: {success_count}, Failed: {failed_count}, Exception: {exception_count}")
    logger.info(f"Results saved to {results_file}")
    return results

################################################################################
# 5. Driver
################################################################################
def parse_args():
    p = argparse.ArgumentParser(description="Run DyG-RAG reproduction on a temporal-QA benchmark")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                   help="which benchmark: complextr | timeqa | tempreason")
    p.add_argument("--mode", default="dynamic", help="query mode (only 'dynamic' is implemented)")
    p.add_argument("--top_k", type=int, default=20, help="top-k events/chunks per query")
    p.add_argument("--concurrency", type=int, default=5, help="max concurrent queries")
    p.add_argument("--max_questions", type=int, default=0,
                   help="0 = all questions; >0 limits to the first N (quick test)")
    p.add_argument("--results_file", default=None, help="override output JSON path")
    p.add_argument("--version-cot", dest="version_cot", action="store_true",
                   help="(ours) enable conflict-aware version timeline in the answer context; default off = baseline")
    p.add_argument("--version-cot-seeds", dest="version_cot_seeds", action="store_true",
                   help="(ours) let Version-CoT also read reranked seed events; implies --version-cot")
    p.add_argument("--interval-events", dest="interval_events", action="store_true",
                   help="(ours) store rule-based start/end interval metadata on events; default off = baseline")
    p.add_argument("--allen-edges", dest="allen_edges", action="store_true",
                   help="(ours) store Allen interval relation metadata on event graph edges; implies --interval-events")
    p.add_argument("--allen-traversal", dest="allen_traversal", action="store_true",
                   help="(ours) bias graph traversal with Allen/time metadata; implies --allen-edges and --interval-events")
    p.add_argument("--normalize-query-time", dest="normalize_query_time", action="store_true",
                   help="(ours) normalize explicit date-offset query times such as '25 years before January 1944'; default off = baseline")
    p.add_argument("--interval-rerank", dest="interval_rerank", action="store_true",
                   help="(ours) blend query-window interval relevance into seed reranking; implies --interval-events")
    return p.parse_args()

def main():
    args = parse_args()
    label = DATASETS[args.dataset]
    if args.version_cot_seeds and not args.version_cot:
        logger.warning("--version-cot-seeds requires Version-CoT; enabling --version-cot for this run")
        args.version_cot = True
    if args.allen_traversal and not args.allen_edges:
        logger.warning("--allen-traversal requires Allen edge metadata; enabling --allen-edges for this run")
        args.allen_edges = True
    if args.allen_edges and not args.interval_events:
        logger.warning("--allen-edges requires interval metadata; enabling --interval-events for this run")
        args.interval_events = True
    if args.interval_rerank and not args.interval_events:
        logger.warning("--interval-rerank requires interval metadata; enabling --interval-events for this run")
        args.interval_events = True

    work_dir = Path(f"{args.dataset}_dir")
    work_dir.mkdir(exist_ok=True)
    corpus_file = Path(f"datasets/{label}/Corpus.json")
    questions_file = Path(f"datasets/{label}/Question.json")
    suffixes = []
    if args.version_cot_seeds:
        suffixes.append("vcot-seeds")
    elif args.version_cot:
        suffixes.append("vcot")
    if args.interval_events:
        suffixes.append("interval-events")
    if args.allen_edges:
        suffixes.append("allen-edges")
    if args.allen_traversal:
        suffixes.append("allen-traversal")
    if args.normalize_query_time:
        suffixes.append("normalize-query-time")
    if args.interval_rerank:
        suffixes.append("interval-rerank")
    feature_suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    results_file = (Path(args.results_file) if args.results_file
                    else Path(f"results_{args.dataset}_mode-{args.mode}_topk-{args.top_k}{feature_suffix}.json"))

    print("🔧 Checking configuration...")
    print(f"   Dataset        : {args.dataset} -> datasets/{label}/")
    print(f"   LLM model      : {BEST_MODEL_NAME}")
    print(f"   LLM base_url   : {LLM_BASE_URL or 'https://api.openai.com/v1 (default)'}")
    print(f"   Embedding      : BGE-M3 @ {LOCAL_BGE_PATH}")
    print(f"   Working dir    : {work_dir}")
    print(f"   Results file   : {results_file}")
    print(f"   Version-CoT    : {'ON (ours)' if args.version_cot else 'OFF (baseline)'}")
    print(f"   Version-CoT seeds: {'ON (ours)' if args.version_cot_seeds else 'OFF (baseline)'}")
    print(f"   Interval events: {'ON (ours)' if args.interval_events else 'OFF (baseline)'}")
    print(f"   Allen edges    : {'ON (ours)' if args.allen_edges else 'OFF (baseline)'}")
    print(f"   Allen traversal: {'ON (ours)' if args.allen_traversal else 'OFF (baseline)'}")
    print(f"   Query time norm: {'ON (ours)' if args.normalize_query_time else 'OFF (baseline)'}")
    print(f"   Interval rerank: {'ON (ours)' if args.interval_rerank else 'OFF (baseline)'}")

    # --- DyG-RAG initialization ---
    embedding_func = get_bge_embedding_func()
    model_ref = embedding_func.model
    embedding_func.model = None  # avoid serializing the model during init

    graph_rag = GraphRAG(
        working_dir=str(work_dir),
        embedding_func=embedding_func,
        best_model_func=best_model_func,
        cheap_model_func=best_model_func,
        enable_llm_cache=True,
        best_model_max_token_size=16384,
        cheap_model_max_token_size=16384,
        model_path="./models",
        ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        ner_model_name="dslim_bert_base_ner",
        enable_version_cot=args.version_cot,
        enable_version_cot_seed_events=args.version_cot_seeds,
        enable_interval_events=args.interval_events,
        enable_allen_edges=args.allen_edges,
        enable_allen_traversal=args.allen_traversal,
        enable_query_time_normalization=args.normalize_query_time,
        enable_interval_rerank=args.interval_rerank,
    )
    embedding_func.model = model_ref

    # --- Insert corpus ---
    insert_corpus(graph_rag, corpus_file)

    # --- Query + save ---
    logger.info(f"Starting {label} {args.mode} query processing...")
    loop = asyncio.get_event_loop()
    query_results = loop.run_until_complete(process_queries(
        graph_rag, label, questions_file, results_file,
        args.mode, args.top_k, args.concurrency, args.max_questions,
    ))

    success_count = sum(1 for r in query_results if r["status"] == "success")
    if success_count > 0:
        avg_query_time = sum(r.get("query_time", 0) for r in query_results if r["status"] == "success") / success_count
        logger.info(f"Average response time for successful {label} queries: {avg_query_time:.2f} seconds")

    # --- Evaluate ---
    try:
        from graphrag.evaluate import run_evaluation
        logger.info("Starting query result evaluation...")
        m = run_evaluation(results_file)
        if m:
            # Paper headline metrics = Accuracy + Recall; the rest printed for rigor.
            line = "=" * 44
            print("\n" + line)
            print(f"  Evaluation — {label}")
            print(line)
            print(f"  Accuracy       : {m.get('accuracy', 0):.2f}")
            print(f"  Recall         : {m.get('recall', 0):.2f}")
            print(f"  Precision      : {m.get('precision', 0):.2f}")
            print(f"  F1             : {m.get('f1', 0):.2f}")
            print(f"  EM             : {m.get('em', 0):.2f}")
            print(f"  Avg query time : {m.get('avg_query_time', 0):.2f} s")
            print(line + "\n")
    except ImportError:
        logger.warning("Evaluation module not imported; run graphrag/evaluate.py separately.")
    except Exception as e:
        logger.error(f"Error in evaluation process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
