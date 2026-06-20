"""
Offline prototype for conflict-aware temporal versioning (research step 1).

This is a STANDALONE module — it does NOT import or modify graphrag/. It works on
structured "versioned facts" of the form (subject, attribute, value, validity
interval). The DyG-RAG pipeline and its reproduction are untouched.

Pipeline demonstrated here:
    events  ->  (subject, attribute, value, [valid_from, valid_to]) tuples
            ->  group by (subject, attribute)
            ->  Allen-interval overlap  ->  conflict detection (single-valued attrs only)
            ->  mark older as `superseded` (never delete)
            ->  render a per-fact version timeline for Time-CoT

Run:  python experiments/versioning/conflict.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Attribute cardinality. Single-valued = at most one value can hold at a given
# time, so two overlapping different values is a CONFLICT (one must be wrong /
# outdated). Multi-valued = several can hold at once (e.g. two employers), so
# overlap is legitimate and NOT a conflict.
# Keys follow ComplexTR question types.
# ---------------------------------------------------------------------------
SINGLE_VALUED = {"spouse", "head coach", "chair", "owner", "position"}
MULTI_VALUED = {"employer", "education", "political party", "team"}


def _norm(value: str) -> str:
    """Normalize a value string for equality comparison (lowercase, strip)."""
    return " ".join(value.lower().split())


# ---------------------------------------------------------------------------
# Validity interval. start/end are years (int) or None for an open bound.
# None start = "since forever / unknown"; None end = "ongoing / still valid".
# ---------------------------------------------------------------------------
NEG_INF = float("-inf")
POS_INF = float("inf")


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
        # STRICT interior overlap (end year treated as exclusive). Intervals that
        # merely touch at a boundary (e.g. [2002–2005] then [2005–2008]) are Allen
        # "meets" = a clean succession of versions, NOT a co-existing conflict.
        return self.lo < other.hi and other.lo < self.hi

    def __str__(self) -> str:
        a = "?" if self.start is None else str(self.start)
        b = "now" if self.end is None else str(self.end)
        return f"[{a}–{b}]"


@dataclass
class VersionedFact:
    subject: str
    attribute: str          # e.g. "political party"
    value: str              # e.g. "True Path Party"
    interval: Interval
    source_id: str = ""     # provenance (chunk-/event- id); for future reliability scoring
    superseded: bool = False
    superseded_by: Optional[int] = None  # index into the fact list

    def key(self) -> tuple[str, str]:
        return (_norm(self.subject), _norm(self.attribute))


@dataclass
class Conflict:
    older: VersionedFact
    newer: VersionedFact
    reason: str = ""


def detect_conflicts(facts: list[VersionedFact]) -> list[Conflict]:
    """Find (entity, attribute) pairs whose single-valued state has overlapping,
    differing values. Marks the older fact `superseded` (kept, not deleted)."""
    conflicts: list[Conflict] = []
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, f in enumerate(facts):
        by_key[f.key()].append(i)

    for (subj, attr), idxs in by_key.items():
        if attr not in SINGLE_VALUED:
            continue  # multi-valued attribute: overlap is legitimate, skip
        # pairwise check within the same (entity, attribute)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                fa, fb = facts[idxs[a]], facts[idxs[b]]
                if _norm(fa.value) == _norm(fb.value):
                    continue
                if not fa.interval.overlaps(fb.interval):
                    continue  # different values at non-overlapping times = versions, not conflict
                # newer = later validity start; tie-break: open-ended end
                older, newer = (fa, fb) if fa.interval.lo <= fb.interval.lo else (fb, fa)
                older.superseded = True
                older.superseded_by = facts.index(newer)
                conflicts.append(Conflict(older, newer,
                    reason=f"overlapping {older.interval} vs {newer.interval} for single-valued '{attr}'"))
    return conflicts


def render_timeline(facts: list[VersionedFact], subject: str, attribute: str) -> str:
    """Version timeline block for a fact — what would be injected into Time-CoT
    so the LLM can pick the version valid at the queried time instead of dumping all."""
    rows = [f for f in facts
            if _norm(f.subject) == _norm(subject) and _norm(f.attribute) == _norm(attribute)]
    rows.sort(key=lambda f: f.interval.lo)
    lines = [f"Versions of [{subject} | {attribute}]:"]
    for f in rows:
        flag = "  (SUPERSEDED)" if f.superseded else ""
        lines.append(f"  {f.interval}  ->  {f.value}{flag}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo on real (entity, attribute) cases pulled from your ComplexTR run.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    facts = [
        # El Potro Álvarez — spouse (single-valued), sequential -> versions, no conflict
        VersionedFact("El Potro Álvarez", "spouse", "Astrid Carolina Herrera", Interval(2002, 2005), "chunk-a"),
        VersionedFact("El Potro Álvarez", "spouse", "Mariángel Ruiz",          Interval(2005, 2008), "chunk-a"),
        VersionedFact("El Potro Álvarez", "spouse", "Dayana Colmenares",       Interval(2013, None), "chunk-a"),

        # Hans Kramers — employer (multi-valued), overlapping -> legitimate, NOT a conflict
        VersionedFact("Hans Kramers", "employer", "Utrecht University",          Interval(1926, 1934), "chunk-b"),
        VersionedFact("Hans Kramers", "employer", "Delft University of Technology", Interval(1931, 1952), "chunk-b"),

        # Jüri Adams — political party. Suppose two sources disagree for overlapping
        # years (a real data-conflict). Here we DO want a conflict flagged.
        VersionedFact("Jüri Adams", "head coach", "Party A", Interval(1991, 1995), "chunk-c"),  # contrived single-valued conflict
        VersionedFact("Jüri Adams", "head coach", "Party B", Interval(1993, 1998), "chunk-d"),
    ]

    conflicts = detect_conflicts(facts)

    print("=== Detected conflicts ===")
    if not conflicts:
        print("(none)")
    for c in conflicts:
        print(f"- {c.older.subject} / {c.older.attribute}: "
              f"'{c.older.value}' {c.older.interval} vs '{c.newer.value}' {c.newer.interval}")
        print(f"    reason: {c.reason}")
        print(f"    -> '{c.older.value}' marked SUPERSEDED by '{c.newer.value}'")

    print("\n=== Version timelines (Time-CoT blocks) ===")
    print(render_timeline(facts, "El Potro Álvarez", "spouse"))
    print()
    print(render_timeline(facts, "Hans Kramers", "employer"))
    print()
    print(render_timeline(facts, "Jüri Adams", "head coach"))
