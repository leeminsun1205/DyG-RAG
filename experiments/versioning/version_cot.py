"""
Research step 3a (offline core): pick the fact version valid at a target time,
and render a focused "version timeline" block for Time-CoT.

This is the reusable logic behind the eventual graphrag wiring. It is standalone
and offline-testable. The actual injection into the query pipeline will live
behind a flag (enable_version_cot) so the baseline stays unchanged.

This directly targets the observed failure mode: on "before/after/X-years" questions
DyG-RAG lists ALL versions and lets inclusion-accuracy save it; here we (a) pick the
version valid at the queried time and (b) hand the LLM a compact, ordered block that
marks superseded versions, so it can answer the right one instead of dumping all.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.versioning.conflict import VersionedFact, Interval, _norm, SINGLE_VALUED


def _matches(f: VersionedFact, subject: str, attribute: str) -> bool:
    return _norm(f.subject) == _norm(subject) and _norm(f.attribute) == _norm(attribute)


def versions_of(facts, subject, attribute):
    rows = [f for f in facts if _matches(f, subject, attribute)]
    rows.sort(key=lambda f: f.interval.lo)
    return rows


def pick_at(facts, subject, attribute, year: int):
    """Return the version(s) valid AT `year` (interval contains it). Prefers a
    non-superseded version when several overlap (the conflict-resolution case)."""
    rows = versions_of(facts, subject, attribute)
    hits = [f for f in rows if f.interval.lo <= year < f.interval.hi]
    if not hits:
        return []
    if attribute in SINGLE_VALUED:
        live = [f for f in hits if not f.superseded]
        return (live or hits)[:1]   # single-valued -> one answer, prefer the non-superseded
    return hits                      # multi-valued -> all concurrently-valid values


def pick_relative(facts, subject, attribute, pivot_year: int, mode: str):
    """mode='before' -> latest version ending at/before pivot; 'after' -> earliest starting at/after pivot."""
    rows = versions_of(facts, subject, attribute)
    if mode == "before":
        cand = [f for f in rows if f.interval.hi <= pivot_year]
        return [cand[-1]] if cand else []
    if mode == "after":
        cand = [f for f in rows if f.interval.lo >= pivot_year]
        return [cand[0]] if cand else []
    return pick_at(facts, subject, attribute, pivot_year)


def version_cot_block(facts, subject, attribute, target_year=None) -> str:
    """Compact timeline for the prompt. Marks superseded; highlights the target match."""
    rows = versions_of(facts, subject, attribute)
    if not rows:
        return ""
    picked = set(id(f) for f in pick_at(facts, subject, attribute, target_year)) if target_year is not None else set()
    lines = [f"Known versions of [{subject} | {attribute}]"
             + (f" (asked about {target_year})" if target_year is not None else "") + ":"]
    for f in rows:
        tags = []
        if f.superseded:
            tags.append("superseded")
        if id(f) in picked:
            tags.append("<-- valid at the asked time")
        tag = ("  [" + ", ".join(tags) + "]") if tags else ""
        lines.append(f"  {f.interval}  ->  {f.value}{tag}")
    return "\n".join(lines)


if __name__ == "__main__":
    facts = [
        VersionedFact("El Potro Álvarez", "spouse", "Astrid Carolina Herrera", Interval(2002, 2005)),
        VersionedFact("El Potro Álvarez", "spouse", "Mariángel Ruiz",          Interval(2005, 2008)),
        VersionedFact("El Potro Álvarez", "spouse", "Dayana Colmenares",       Interval(2013, None)),
    ]
    S, A = "El Potro Álvarez", "spouse"

    def names(rs): return [r.value for r in rs]

    # version-pick checks (this is the metric the new benchmark will use)
    assert names(pick_at(facts, S, A, 2006)) == ["Mariángel Ruiz"], pick_at(facts, S, A, 2006)
    assert names(pick_at(facts, S, A, 2003)) == ["Astrid Carolina Herrera"]
    assert names(pick_at(facts, S, A, 2020)) == ["Dayana Colmenares"]
    assert names(pick_at(facts, S, A, 2010)) == [], "gap between marriages -> no spouse"
    assert names(pick_relative(facts, S, A, 2013, "before")) == ["Mariángel Ruiz"]
    assert names(pick_relative(facts, S, A, 2008, "after")) == ["Dayana Colmenares"]
    print("version-pick selftest OK")
    print()
    print(version_cot_block(facts, S, A, target_year=2006))
