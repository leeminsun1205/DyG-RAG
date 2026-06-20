"""
Research step 2a: turn already-extracted DyG-RAG events into structured
versioned facts, then run conflict detection on them.

STANDALONE & READ-ONLY w.r.t. the baseline:
  - reads <dataset>_dir/graph_event_dynamic_graph.graphml (built by the normal run)
  - does NOT import or modify graphrag/ or reproduce/run.py
  - does NOT touch the LLM-extraction cache (no re-extraction, no extra $ on indexing)
So the baseline command `python reproduce/run.py --dataset complextr` is unchanged.

It DOES make one cheap LLM call per event (subject/attribute/value/interval).
Use --max-events to sample a few dozen for a quick, cheap smoke test first.

Usage (run from repo root):
  # offline logic check, no API, no data needed:
  python experiments/versioning/extract_facts.py --selftest

  # real run on the built graph (set the same env as the baseline):
  export OPENAI_API_KEY=... OPENAI_BASE_URL=https://llm.wokushop.com/v1
  export LLM_MODEL=gemini-2.5-flash-lite
  python experiments/versioning/extract_facts.py --dataset complextr --max-events 40
  python experiments/versioning/extract_facts.py --dataset complextr            # all events
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from pathlib import Path

# import the conflict logic from step 1
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.versioning.conflict import (
    VersionedFact, Interval, detect_conflicts, render_timeline, SINGLE_VALUED, MULTI_VALUED,
)

ATTRS = sorted(SINGLE_VALUED | MULTI_VALUED)

PROMPT_TMPL = """You extract structured temporal facts from ONE event sentence.

Output a JSON array. Each element:
  {{"subject": "<person/entity the fact is about>",
    "attribute": "<one of: {attrs}, or other>",
    "value": "<the value of that attribute>",
    "valid_from": <start year as integer, or null>,
    "valid_to": <end year as integer, or null>}}

Rules:
- Only emit a fact if the sentence states a datable, stateful relation
  (who someone was married to / which party / employer / position / school / team / coach / chair / owner).
- valid_to = null means ongoing / no end given. valid_from = null means start unknown.
- If a year is a single point (e.g. "married in 1965"), set valid_from = that year, valid_to = null.
- If the sentence has no such fact, output [].
- Output ONLY the JSON array, nothing else.

Known event timestamp hint: {timestamp}
Event sentence: {sentence}
Context: {context}"""


def _extract_json_array(text: str):
    """Pull the first JSON array out of an LLM response (tolerates code fences/prose)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def _year(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\d{4}", str(v))
    return int(m.group()) if m else None


def rows_to_facts(rows, source_id: str) -> list[VersionedFact]:
    facts = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        subj, attr, val = r.get("subject"), r.get("attribute"), r.get("value")
        if not (subj and attr and val):
            continue
        if attr == "other":
            continue
        facts.append(VersionedFact(
            subject=str(subj).strip(),
            attribute=str(attr).strip().lower(),
            value=str(val).strip(),
            interval=Interval(_year(r.get("valid_from")), _year(r.get("valid_to"))),
            source_id=source_id,
        ))
    return facts


def load_events(graphml_path: Path, max_events: int):
    import networkx as nx
    g = nx.read_graphml(graphml_path)
    events = []
    for node_id, data in g.nodes(data=True):
        sentence = (data.get("sentence") or "").strip()
        if not sentence:
            continue
        events.append({
            "event_id": node_id,
            "sentence": sentence,
            "timestamp": data.get("timestamp") or "static",
            "context": (data.get("context") or "")[:500],
            "source_id": data.get("source_id") or "",
        })
    if max_events and max_events > 0:
        events = events[:max_events]
    return events


def build_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT") or 120),
        max_retries=5,
    )


def llm_extract(client, model, ev):
    prompt = PROMPT_TMPL.format(attrs=", ".join(ATTRS), timestamp=ev["timestamp"],
                                sentence=ev["sentence"], context=ev["context"])
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_json_array(resp.choices[0].message.content or "")


def run_selftest():
    """Verify parse + conflict wiring with a canned LLM response — no API, no data."""
    fake = '```json\n[{"subject":"X","attribute":"spouse","value":"A","valid_from":2002,"valid_to":2005},' \
           '{"subject":"X","attribute":"spouse","value":"B","valid_from":2003,"valid_to":2007}]\n```'
    rows = _extract_json_array(fake)
    facts = rows_to_facts(rows, "chunk-test")
    assert len(facts) == 2, facts
    conflicts = detect_conflicts(facts)
    assert len(conflicts) == 1, conflicts            # overlapping spouse 2003-2005 -> conflict
    assert facts[0].superseded                       # older (2002 start) superseded
    print("selftest OK: parsed 2 facts, detected 1 conflict, older marked superseded")
    print(render_timeline(facts, "X", "spouse"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="complextr")
    ap.add_argument("--work-dir", default=None, help="defaults to <dataset>_dir")
    ap.add_argument("--max-events", type=int, default=0, help="0 = all; use a small N for a cheap smoke test")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL") or "gpt-4.1-nano")
    ap.add_argument("--rpm", type=float, default=float(os.getenv("LLM_RPM") or 60))
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    work_dir = Path(args.work_dir or f"{args.dataset}_dir")
    graphml = work_dir / "graph_event_dynamic_graph.graphml"
    if not graphml.exists():
        sys.exit(f"ERROR: {graphml} not found. Run the baseline first to build it.")

    events = load_events(graphml, args.max_events)
    print(f"Loaded {len(events)} events from {graphml}")

    client = build_client()
    min_interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    all_facts: list[VersionedFact] = []
    last = 0.0
    for i, ev in enumerate(events, 1):
        if min_interval:
            wait = last + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            last = time.time()
        try:
            rows = llm_extract(client, args.model, ev)
            all_facts.extend(rows_to_facts(rows, ev["source_id"]))
        except Exception as e:
            print(f"  [warn] event {i} failed: {e}")
        if i % 20 == 0 or i == len(events):
            print(f"  {i}/{len(events)} events -> {len(all_facts)} facts so far")

    conflicts = detect_conflicts(all_facts)

    print(f"\n=== {len(all_facts)} versioned facts, {len(conflicts)} conflicts ===")
    for c in conflicts:
        print(f"- {c.older.subject} / {c.older.attribute}: "
              f"'{c.older.value}' {c.older.interval}  vs  '{c.newer.value}' {c.newer.interval}")

    out = Path(args.out or f"facts_{args.dataset}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "facts": [vars(x) | {"interval": str(x.interval)} for x in all_facts],
            "conflicts": [{"older": vars(c.older) | {"interval": str(c.older.interval)},
                           "newer": vars(c.newer) | {"interval": str(c.newer.interval)},
                           "reason": c.reason} for c in conflicts],
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
