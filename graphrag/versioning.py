"""
Conflict-aware temporal versioning for DyG-RAG (production module).

Pure logic (interval / Allen overlap / conflict / supersession / timeline) plus a
query-time helper `build_version_section` that turns the events retrieved for a
query into an explicit, ordered "version timeline" block — so the answer LLM can
pick the version valid at the queried time instead of dumping every version.

Wired into GraphRAG.dynamic_query ONLY when `enable_version_cot=True`; the default
(False) leaves the baseline pipeline byte-for-byte unchanged.

Design notes (validated on real ComplexTR extraction):
- single-valued attrs (only `spouse` without an object scope) -> overlapping
  different values = conflict; multi-valued (position, employer, ...) -> people
  hold several at once, so overlap is legitimate, NOT a conflict.
- Allen "meets" vs "overlaps": touching boundaries (e.g. [2002–2005] then
  [2005–2008]) is a clean succession, not a conflict (strict interior overlap).
"""
from __future__ import annotations

import json, re
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

SINGLE_VALUED = {"spouse"}
MULTI_VALUED = {"employer", "education", "political party", "team",
                "position", "head coach", "chair", "owner"}
ATTRS = sorted(SINGLE_VALUED | MULTI_VALUED)

NEG_INF, POS_INF = float("-inf"), float("inf")


def _norm(v: str) -> str:
    return " ".join(str(v).lower().split())


@dataclass
class Interval:
    start: Optional[int] = None
    end: Optional[int] = None

    @property
    def lo(self) -> float:
        return NEG_INF if self.start is None else self.start

    @property
    def hi(self) -> float:
        return POS_INF if self.end is None else self.end

    def overlaps(self, other: "Interval") -> bool:
        # strict interior overlap; touching boundaries = succession, not conflict
        return self.lo < other.hi and other.lo < self.hi

    def __str__(self) -> str:
        a = "?" if self.start is None else str(self.start)
        b = "now" if self.end is None else str(self.end)
        return f"[{a}-{b}]"


@dataclass
class VersionedFact:
    subject: str
    attribute: str
    value: str
    interval: Interval
    source_id: str = ""
    superseded: bool = False

    def key(self):
        return (_norm(self.subject), _norm(self.attribute))


def detect_conflicts(facts: list[VersionedFact]) -> list[tuple[VersionedFact, VersionedFact]]:
    """Dedup, then flag single-valued (entity, attribute) slots with overlapping
    differing values. Marks the older (earlier-starting) fact superseded."""
    seen, deduped = set(), []
    for f in facts:
        sig = (f.key(), _norm(f.value), f.interval.lo, f.interval.hi)
        if sig not in seen:
            seen.add(sig)
            deduped.append(f)
    facts[:] = deduped

    conflicts = []
    by_key = defaultdict(list)
    for f in facts:
        by_key[f.key()].append(f)
    for (subj, attr), group in by_key.items():
        if attr not in SINGLE_VALUED:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if _norm(a.value) == _norm(b.value) or not a.interval.overlaps(b.interval):
                    continue
                older, newer = (a, b) if a.interval.lo <= b.interval.lo else (b, a)
                older.superseded = True
                conflicts.append((older, newer))
    return conflicts


def _row_valid_at(iv: Interval, lo: float, hi: float) -> bool:
    # inclusive overlap (touching boundaries count) — used for "valid at asked time"
    # highlighting, unlike the strict-interior overlap used for conflict detection.
    return iv.lo <= hi and lo <= iv.hi


def _fmt_asked(asked: tuple) -> str:
    lo, hi = asked
    return str(lo) if lo == hi else f"{lo}-{hi}"


def render_timeline(facts: list[VersionedFact], subject: str, attribute: str,
                    asked: Optional[tuple] = None) -> str:
    rows = [f for f in facts if _norm(f.subject) == _norm(subject) and _norm(f.attribute) == _norm(attribute)]
    rows.sort(key=lambda f: f.interval.lo)
    lines = [f"[{subject} | {attribute}]"]
    for f in rows:
        flags = []
        if asked and _row_valid_at(f.interval, asked[0], asked[1]):
            flags.append(f"<- valid at asked time {_fmt_asked(asked)}")
        if f.superseded:
            flags.append("superseded")
        suffix = ("   " + ", ".join(flags)) if flags else ""
        lines.append(f"  {f.interval} -> {f.value}{suffix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query-time extraction + section building
# ---------------------------------------------------------------------------
_BATCH_PROMPT = """Extract structured temporal facts from the numbered event sentences below.

Output ONLY a JSON array. Each element:
  {{"subject": "<entity the fact is about>",
    "attribute": "<one of: {attrs}, or other>",
    "value": "<value of that attribute>",
    "valid_from": <start year int or null>, "valid_to": <end year int or null>}}

Rules: emit a fact only for a datable, stateful relation (spouse/party/employer/
position/education/team/head coach/chair/owner). A single point year -> valid_from=year,
valid_to=null. If a sentence has no such fact, skip it. Output [] if none.

Events:
{sentences}"""


def _extract_json_array(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1 or e < s:
        return []
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return []


def _year(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\d{4}", str(v))
    return int(m.group()) if m else None


def _rows_to_facts(rows, source_id: str) -> list[VersionedFact]:
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        subj, attr, val = r.get("subject"), r.get("attribute"), r.get("value")
        if not (subj and attr and val) or attr == "other":
            continue
        out.append(VersionedFact(str(subj).strip(), str(attr).strip().lower(), str(val).strip(),
                                 Interval(_year(r.get("valid_from")), _year(r.get("valid_to"))), source_id))
    return out


def _asked_years(time_constraints: Optional[dict]) -> Optional[tuple]:
    """Extract the asked year (or [start,end] window) from the query's normalized
    time_constraints dict, as an (lo, hi) int tuple. None if no year is present."""
    if not time_constraints:
        return None
    s, e = _year(time_constraints.get("start_time")), _year(time_constraints.get("end_time"))
    if s is None and e is None:
        return None
    lo = s if s is not None else e
    hi = e if e is not None else s
    return (min(lo, hi), max(lo, hi))


async def build_version_section(events: list[dict], llm_func,
                                time_constraints: Optional[dict] = None) -> str:
    """One LLM call over retrieved event sentences -> version timelines for any
    (entity, attribute) with >= 2 versions. Returns "" if nothing useful.

    The block is STRICTLY ADDITIVE: it is a temporal index that points the answer
    LLM at the version valid at the asked time (marked '<-'), but it never tells
    the model to drop other values or to abbreviate a specific name to a general
    one. This keeps the baseline's breadth (which inclusion-accuracy rewards) while
    fixing wrong-time selections. `time_constraints` (the query's parsed
    {start_time,end_time}) drives the '<- valid at asked time' highlight."""
    sentences = [e.get("sentence", "").strip() for e in events if e.get("sentence", "").strip()]
    if len(sentences) < 2:
        return ""
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = _BATCH_PROMPT.format(attrs=", ".join(ATTRS), sentences=numbered)
    resp = await llm_func(prompt)
    facts = _rows_to_facts(_extract_json_array(resp), "query")
    if not facts:
        return ""
    detect_conflicts(facts)  # marks superseded in place

    asked = _asked_years(time_constraints)
    groups = defaultdict(list)
    for f in facts:
        groups[f.key()].append(f)
    blocks = [render_timeline(facts, g[0].subject, g[0].attribute, asked)
              for g in groups.values() if len(g) >= 2]
    if not blocks:
        return ""
    header = (
        "Fact version timelines (temporal index for entities in the question). "
        "Each line is one time-bounded value; the line marked '<-' is the version "
        "valid at the asked time — use it as the PRIMARY answer. Keep the most "
        "specific wording from the events/chunks for that value (do NOT shorten a "
        "specific name to a more general one), and do NOT omit other relevant "
        "details you would otherwise report. Lines marked 'superseded' are older "
        "values overridden by a conflicting newer one.\n"
    )
    return header + "\n".join(blocks)
