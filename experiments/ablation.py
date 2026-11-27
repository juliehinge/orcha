"""Analysis for the abstract ablation (throwaway; not part of the app).

The runs themselves go through the real eval path:
  uv run orcha evals run --ablation full --repeat 1 --langfuse
which also writes experiments/results/<run>-rep<n>__<timestamp>.json.
See experiments/run_matrix.sh for all four runs.

  uv run python experiments/ablation.py ceiling   # no LLM: text quality per variant
  uv run python experiments/ablation.py analyze   # compare the finished runs
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, pstdev

import app.extractors.pdfplumber as plumber
from app.evals.ablation import BUCKETS, CHECKS, RUNS, TEXT_VARIANTS
from app.evals.dataset import load_local_dataset
from app.evals.run import _resolve_dataset_root

HERE = Path(__file__).resolve().parent
TEXTS = HERE / "texts"
RESULTS = HERE / "results"
PAGES = [1, 2]  # same as the eval

# (label, run B, run A, what B - A measures)
COMPARISONS = [
    ("extraction", "extraction", "baseline", "new pdfplumber settings"),
    ("prompt", "full", "extraction", "new description prompt"),
    ("total", "full", "baseline", "both (the check is replayed below)"),
]


def rid_of(item):
    return item["input"]["record_id"].strip("_")


def gt_abstract(item):
    exp = item["expected_output"].get("description")
    if isinstance(exp, list):
        exp = exp[0] if exp else None
    if isinstance(exp, dict):
        exp = next(iter(exp.values()), None)
    return exp or None


# --- ceiling: no LLM ---------------------------------------------------------


def extract_texts(variant, root, items):
    """Extract (and cache) page 1-2 text for every item with one text variant."""
    out_dir = TEXTS / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    plumber._TEXT_SETTINGS = TEXT_VARIANTS[variant]
    texts = {}
    for item in items:
        path = out_dir / f"{rid_of(item)}.txt"
        if not path.exists():
            try:
                pdf = (root / item["input"]["pdf_path"]).read_bytes()
                text = plumber.PdfplumberExtractor().extract(pdf, PAGES)["full_text"]
            except Exception as exc:  # broken PDFs fail in every variant
                print(f"  extract failed {rid_of(item)}: {exc!r}", file=sys.stderr)
                text = ""
            path.write_text(text, encoding="utf-8")
        texts[rid_of(item)] = path.read_text(encoding="utf-8")
    return texts


def best_window_score(gt, text):
    """Evaluator score a perfect copy of the GT could get from this text."""
    gt, text = re.sub(r"\s+", " ", gt), re.sub(r"\s+", " ", text)
    blocks = [
        b
        for b in SequenceMatcher(None, text, gt, autojunk=False).get_matching_blocks()
        if b.size >= 8
    ]
    if not blocks:
        return 0.0
    start = max(blocks[0].a - blocks[0].b, 0)
    end = blocks[-1].a + len(gt) - blocks[-1].b
    return SequenceMatcher(None, gt, text[start:end], autojunk=False).ratio()


def cmd_ceiling(_args):
    root = _resolve_dataset_root(None, None)
    items = load_local_dataset(root, "ceiling")
    with_gt = [i for i in items if gt_abstract(i)]
    scores = {}
    for variant in TEXT_VARIANTS:
        texts = extract_texts(variant, root, items)
        scores[variant] = {
            rid_of(i): best_window_score(gt_abstract(i), texts[rid_of(i)])
            for i in with_gt
        }
    (HERE / "ceiling.json").write_text(json.dumps(scores, indent=1))
    print(f"{len(with_gt)} items with a GT abstract; best score a perfect copy could get:")
    print(f"  {'variant':8} {'mean':>6} {'>=0.8':>6} {'<0.5 (unreachable)':>20}")
    for variant, s in scores.items():
        v = list(s.values())
        print(
            f"  {variant:8} {mean(v):6.3f} {sum(x >= 0.8 for x in v):6} "
            f"{sum(x < 0.5 for x in v):20}"
        )
    print("\nitems that change by >= 0.1 between old and new:")
    for rid in sorted(scores["old"], key=lambda r: scores["new"][r] - scores["old"][r]):
        if abs(scores["new"][rid] - scores["old"][rid]) >= 0.1:
            print(
                f"  {rid:12} " + " ".join(f"{v}={scores[v][rid]:.2f}" for v in scores)
            )


# --- analyze -----------------------------------------------------------------


def load_results():
    """{run: {repeat: {rid: ablation info}}}; the newest file wins per run/repeat."""
    data = defaultdict(dict)
    for path in sorted(RESULTS.glob("*-rep*__*.json")):  # timestamps sort in order
        payload = json.loads(path.read_text())
        meta = payload["ablation"]
        recs = {it["ablation"]["record_id"]: it["ablation"] for it in payload["items"]}
        missing = payload["total_items"] - len(recs)
        run, rep = meta["ablation_run"], int(meta["repeat"])
        if rep in data[run]:
            print(f"note: {path.name} replaces an earlier {run}-rep{rep} file")
        data[run][rep] = recs
        if missing:
            print(f"note: {path.name}: {missing} items failed (excluded)")
    return data


def item_scores(reps, check="C2"):
    """Per-item description score averaged over repeats."""
    by = defaultdict(list)
    for recs in reps.values():
        for rid, rec in recs.items():
            by[rid].append(rec["checks"][check]["description"]["score"])
    return {rid: mean(v) for rid, v in by.items()}


def bootstrap_ci(deltas, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def cmd_analyze(args):
    data = load_results()
    if not data:
        sys.exit(f"no results in {RESULTS} yet")
    # Items whose GT abstract isn't in any variant's text score 0 everywhere.
    unreachable = set()
    ceiling = HERE / "ceiling.json"
    if ceiling.exists():
        c = json.loads(ceiling.read_text())
        unreachable = {rid for rid in c["old"] if max(c[v][rid] for v in c) < 0.5}
    elif args.reachable:
        print("note: run `ceiling` first to exclude unreachable items")

    def keep(rid):
        return not (args.reachable and rid in unreachable)

    scope = (
        f"reachable items only ({len(unreachable)} unreachable excluded)"
        if args.reachable
        else "all items"
    )
    print(f"Scope: {scope}. Description score, presence check C2 unless noted.\n")

    print("== Runs ==")
    print(
        f"{'run':10} {'cfg':7} {'reps':>4} {'n':>4} {'mean':>6}  "
        + "  ".join(f"{b:>13}" for b in BUCKETS)
    )
    for run in RUNS:
        if run not in data:
            continue
        reps = data[run]
        scores = {r: s for r, s in item_scores(reps).items() if keep(r)}
        first = reps[min(reps)]
        counts = Counter(
            rec["checks"]["C2"]["bucket"] for rid, rec in first.items() if keep(rid)
        )
        print(
            f"{run:10} {'/'.join(RUNS[run]):7} {len(reps):4} {len(scores):4} "
            f"{mean(scores.values()):6.3f}  "
            + "  ".join(f"{counts[b]:13}" for b in BUCKETS)
        )

    print("\n== Noise between repeats (same config) ==")
    for run, reps in data.items():
        if len(reps) < 2:
            continue
        common = {r for r in set.intersection(*(set(x) for x in reps.values())) if keep(r)}

        def score(k, rid):
            return reps[k][rid]["checks"]["C2"]["description"]["score"]

        sds = [pstdev([score(k, rid) for k in reps]) for rid in common]
        means = [mean(score(k, rid) for rid in common) for k in reps]
        flips = [
            rid
            for rid in common
            if len({reps[k][rid]["checks"]["C2"]["bucket"] for k in reps}) > 1
        ]
        print(
            f"{run}: run means {', '.join(f'{m:.3f}' for m in means)}; "
            f"per-item SD mean {mean(sds):.3f}; {len(flips)}/{len(common)} items "
            f"change bucket between repeats ({', '.join(sorted(flips)[:6])})"
        )

    print("\n== Effects (paired over items, B - A, mean over repeats) ==")
    print(
        f"{'change':10} {'B - A':22} {'n':>4} {'delta':>7} {'95% CI':>17} "
        f"{'better':>6} {'worse':>6}  measures"
    )
    for label, b, a, what in COMPARISONS:
        if a not in data or b not in data:
            print(f"{label:10} {b + ' - ' + a:22} (missing runs)")
            continue
        sa, sb = item_scores(data[a]), item_scores(data[b])
        common = sorted(r for r in sa.keys() & sb.keys() if keep(r))
        deltas = [sb[r] - sa[r] for r in common]
        lo, hi = bootstrap_ci(deltas)
        print(
            f"{label:10} {b + ' - ' + a:22} {len(common):4} {mean(deltas):+7.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] {sum(d > 0.05 for d in deltas):6} "
            f"{sum(d < -0.05 for d in deltas):6}  {what}"
        )
    print("\n== Bucket moves (first repeat, A -> B) ==")
    for label, b, a, _ in COMPARISONS:
        if a not in data or b not in data:
            continue
        ra, rb = data[a][min(data[a])], data[b][min(data[b])]
        moves = defaultdict(list)
        for rid in ra.keys() & rb.keys():
            ba, bb = ra[rid]["checks"]["C2"]["bucket"], rb[rid]["checks"]["C2"]["bucket"]
            if keep(rid) and ba != bb:
                moves[(ba, bb)].append(rid)
        text = "; ".join(
            f"{x}->{y} x{len(r)} ({', '.join(sorted(r)[:3])})"
            for (x, y), r in sorted(moves.items(), key=lambda kv: -len(kv[1]))
        )
        print(f"{label:10} {text or 'no moves'}")

    print("\n== Presence check replay (same model outputs, mean over repeats) ==")
    print(f"{'run':10} {'C0 orig':>8} {'C1 fix1':>8} {'C2 now':>8}  items where the check decides")
    for run, reps in data.items():
        means = [
            mean(v for r, v in item_scores(reps, check).items() if keep(r))
            for check in CHECKS
        ]
        decided = set()
        for recs in reps.values():
            for rid, rec in recs.items():
                kept = {bool(rec["checks"][c]["description"]["predicted"]) for c in CHECKS}
                if keep(rid) and len(kept) > 1:
                    decided.add(rid)
        print(
            f"{run:10} {means[0]:8.3f} {means[1]:8.3f} {means[2]:8.3f}  "
            f"{', '.join(sorted(decided)) or '-'}"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ceiling").set_defaults(fn=cmd_ceiling)
    a = sub.add_parser("analyze")
    a.add_argument(
        "--all-items",
        dest="reachable",
        action="store_false",
        help="include items whose abstract isn't in any text variant",
    )
    a.set_defaults(fn=cmd_analyze)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
