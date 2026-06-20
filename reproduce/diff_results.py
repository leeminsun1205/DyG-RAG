"""
A/B diff between two results_*.json runs (e.g. baseline vs --version-cot).

Scores each question with the SAME inclusion-accuracy rule as graphrag/evaluate.py
(multi-answer: ALL golds must be covered), then classifies every question as:
  win   : A wrong -> B right   (the component helped)
  loss  : A right -> B wrong   (the component hurt / suppressed the gold)
  tie+  : both right
  tie-  : both wrong

Prints net = wins - losses and dumps case studies for win / loss.

Usage:
  python reproduce/diff_results.py BASELINE.json VCOT.json [--cases N]
"""
import argparse, json, sys, os, re, string, csv, io

# --- scoring logic copied verbatim from graphrag/evaluate.py (Evaluator) ---
# inlined to avoid importing the package (torch/pandas not installed locally)

def normalize_answer(s):
    s = re.sub(r'\b(a|an|the|An|The)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def eval_accuracy(prediction, ground_truth):
    prediction = prediction.replace("Yes", "yes").replace("No", "no")
    ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
    return 1 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0


def parse_multiple_answers(answer_str):
    answer_str = answer_str.strip()
    if "|" in answer_str:
        return [a.strip() for a in answer_str.split("|")]
    if '"' in answer_str:
        try:
            return [a.strip() for a in next(csv.reader(io.StringIO(answer_str))) if a.strip()]
        except Exception:
            pass
    if "," in answer_str:
        return [a.strip() for a in answer_str.split(",")]
    return [answer_str]


def acc_row(pred: str, gold: str) -> int:
    """Inclusion accuracy with multi-answer rule (all golds covered)."""
    pred = pred or ""
    golds = parse_multiple_answers(gold or "")
    return 1 if all(eval_accuracy(pred, g) == 1 for g in golds) else 0


def load(path):
    rows = json.load(open(path))["results"]
    out = {}
    for r in rows:
        if r.get("status") != "success":
            continue
        out[r["question_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("vcot")
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--dump", default=None, help="write ALL wins+losses (full answers) to this JSON file")
    args = ap.parse_args()

    A, B = load(args.baseline), load(args.vcot)
    common = sorted(set(A) & set(B))
    print(f"baseline rows: {len(A)} | vcot rows: {len(B)} | common (both success): {len(common)}\n")

    wins, losses, tie_pos, tie_neg = [], [], [], []
    for qid in common:
        a, b = A[qid], B[qid]
        sa = acc_row(a["answer"], a["golden_answer"])
        sb = acc_row(b["answer"], b["golden_answer"])
        if sa == 0 and sb == 1:
            wins.append((qid, a, b))
        elif sa == 1 and sb == 0:
            losses.append((qid, a, b))
        elif sa == 1 and sb == 1:
            tie_pos.append(qid)
        else:
            tie_neg.append(qid)

    n = len(common)
    print("=" * 60)
    print(f"  wins  (baseline✗ -> vcot✓) : {len(wins)}")
    print(f"  losses(baseline✓ -> vcot✗) : {len(losses)}")
    print(f"  net (wins - losses)        : {len(wins) - len(losses)}  ({100*(len(wins)-len(losses))/n:+.2f} pp)")
    print(f"  both correct               : {len(tie_pos)}")
    print(f"  both wrong                 : {len(tie_neg)}")
    print(f"  baseline acc               : {100*(len(tie_pos)+len(losses))/n:.2f}")
    print(f"  vcot acc                   : {100*(len(wins)+len(tie_pos))/n:.2f}")
    print("=" * 60)

    def dump(title, items):
        print(f"\n{'#'*60}\n# {title} (showing {min(args.cases,len(items))} of {len(items)})\n{'#'*60}")
        for qid, a, b in items[:args.cases]:
            print(f"\n--- qid {qid} ---")
            print(f"Q   : {a['question']}")
            print(f"GOLD: {a['golden_answer']}")
            print(f"BASE: {a['answer'][:400]}")
            print(f"VCOT: {b['answer'][:400]}")

    dump("WINS", wins)
    dump("LOSSES", losses)

    if args.dump:
        def rec(qid, a, b):
            return {"question_id": qid, "question": a["question"], "gold": a["golden_answer"],
                    "baseline_answer": a["answer"], "vcot_answer": b["answer"]}
        out = {"wins": [rec(*x) for x in wins], "losses": [rec(*x) for x in losses]}
        json.dump(out, open(args.dump, "w"), ensure_ascii=False, indent=2)
        print(f"\nwrote full case studies -> {args.dump}")


if __name__ == "__main__":
    main()
