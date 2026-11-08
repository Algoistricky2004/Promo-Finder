#!/usr/bin/env python3
"""Promo finder CLI.

  python run.py --pack ./pack --out ./out --judge heuristic                    # free, offline, no model
  python run.py --pack ./pack --out ./out --judge anthropic --model <model-id>  # real LLM judge
  python run.py --pack ./pack --out ./out --judge openai --model <id> [--base-url <url>] [--mode select|free]
      select (default for live models): model picks among code-built 20-60 s windows; free: model proposes stitched segments
  python run.py --pack ./pack --out ./out --judge cache --cache judgments/judgments.json   # replay saved judgments
"""
import argparse, json, os, sys, csv
from promo_finder.parse import load_pack, fmt
from promo_finder.heuristic import candidates
from promo_finder.judge import judge_episode, fix_episode, select_episode
from promo_finder.validate import validate, fmt_segments, anchor
from promo_finder.report import rank, write_outputs
from promo_finder.rerank import rerank_promos

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True); ap.add_argument("--out", default="out")
ap.add_argument("--judge", choices=["heuristic", "anthropic", "openai", "cache"], default="heuristic")
ap.add_argument("--model", default=os.environ.get("PROMO_MODEL")); ap.add_argument("--base-url")
ap.add_argument("--repair", action=argparse.BooleanOptionalAction, default=None)
ap.add_argument("--mode", choices=["select", "free"], default="select", help="select: model picks among code-built 20-60s windows (robust, recommended for small models). free: model proposes stitched segments itself")
ap.add_argument("--select-k", type=int, default=6)
ap.add_argument("--rerank", type=int, default=40, help="editor pass: one extra call compares the top N promos against each other and may drop weak ones (0 = off; never runs for --judge cache)")
ap.add_argument("--rerank-drop-max", type=float, default=0.5)
ap.add_argument("--max-overlap", type=float, default=0.3, help="drop a lower-scored promo that overlaps a higher-scored one by more than this share of the shorter clip")
ap.add_argument("--cache"); ap.add_argument("--audience", choices=["cold", "warm"], default="cold")
ap.add_argument("--shortlist-k", type=int, default=4); ap.add_argument("--episodes", help="e.g. 1-10 or 3,5,7")
a = ap.parse_args()

eps = load_pack(a.pack)
if a.episodes:
    sel = set()
    for part in a.episodes.split(","):
        lo, _, hi = part.partition("-"); sel |= set(range(int(lo), int(hi or lo) + 1))
    eps = [e for e in eps if e.num in sel]
by_num = {e.num: e for e in load_pack(a.pack)}
os.makedirs(a.out, exist_ok=True)

# Stage 1: heuristic candidates (always run; used as shortlist hint and as the offline baseline)
rows, short = [], {}
for e in eps:
    cs = candidates(e, k=a.shortlist_k); short[e.num] = cs
    for c in cs:
        rows.append([e.num, fmt(c.start), fmt(c.end), round(c.end - c.start, 1), round(c.hook, 2), round(c.cliff, 2),
                     round(c.standalone, 2), round(c.intensity, 2), round(c.total, 3), "; ".join(c.notes)])
with open(os.path.join(a.out, "heuristic_candidates.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ep", "start", "end", "len_s", "hook", "cliff", "standalone", "intensity", "total", "signals"]); w.writerows(rows)

if a.judge == "heuristic":
    print(f"heuristic-only: wrote {len(rows)} candidates to {a.out}/heuristic_candidates.csv (no LLM judge, no ranking beyond score)")
    sys.exit(0)

cache = json.load(open(a.cache)) if a.judge == "cache" else None
if a.judge != "cache" and not a.model: sys.exit("--model (or env PROMO_MODEL) required for LLM judges")
repair = (a.judge != 'cache') if a.repair is None else a.repair
valid, rejected = [], []
for e in eps:
    prev = by_num[e.num - 1].continuity if (e.num - 1) in by_num else ""
    try:
        if a.judge != "cache" and a.mode == "select": raw = select_episode(e, a.judge, a.model, candidates(e, k=a.select_k), prev, a.base_url)
        else: raw = judge_episode(e, a.judge, a.model, prev, short[e.num], a.base_url, cache)
    except Exception as ex:
        print(f"ep{e.num:>2}: JUDGE CALL FAILED ({ex}) - skipping this episode"); continue
    for p in raw:
        p = dict(p); p["ep"] = e.num
        try: ok, errs, warns = validate(p, by_num, repair=repair)
        except Exception as ex: ok, errs, warns = None, [f"validator error: {ex!r}"], []
        if not ok and repair:
            p2 = anchor(p, by_num)
            if p2:
                ok, _, w2 = validate(p2, by_num, repair=True)
                if ok: warns = w2 + ["re-anchored from the model's hook/cliffhanger quotes"]
        if ok: ok["validator_warnings"] = warns; valid.append(ok)
        else:
            rejected.append({"ep": e.num, "promo": p, "errors": errs})
            print(f"      rejected ep{e.num}: {errs[0][:140]}")
    if repair and a.judge != "cache" and a.mode == "free":
        bad = [r for r in rejected if r["ep"] == e.num and not r.get("retried")]
        if bad:
            try: fixed = fix_episode(e, a.judge, a.model, bad, prev, short[e.num], a.base_url)
            except Exception as ex: fixed = []; print(f"      retry failed: {ex}")
            for r in bad: r["retried"] = True
            for p in fixed:
                p = dict(p); p["ep"] = e.num
                try: ok, errs, warns = validate(p, by_num, repair=True)
                except Exception as ex: ok, errs, warns = None, [repr(ex)], []
                if not ok:
                    p2 = anchor(p, by_num)
                    if p2:
                        ok, _, w2 = validate(p2, by_num, repair=True)
                        if ok: warns = w2 + ["re-anchored from the model's hook/cliffhanger quotes"]
                if ok:
                    ok["validator_warnings"] = warns + ["fixed by model retry"]; valid.append(ok); print(f"      retry kept a promo for ep{e.num}")
                else: print(f"      retry still rejected ep{e.num}: {errs[0][:120]}")
    print(f"ep{e.num:>2}: judge returned {len(raw)}, kept so far {len(valid)}, rejected so far {len(rejected)}")
from promo_finder.report import composite
valid.sort(key=lambda p: -composite(p, a.audience)); kept = []
for p in valid:
    dup = None
    for q in kept:
        ov = sum(max(0, min(s["end"], t["end"]) - max(s["start"], t["start"])) for s in p["segments"] for t in q["segments"] if s["ep"] == t["ep"])
        if ov / min(p["runtime_s"], q["runtime_s"]) > a.max_overlap: dup = (q, ov)
    if dup:
        rejected.append({"ep": p["segments"][0]["ep"], "promo": p, "errors": [f"duplicate: overlaps a higher-scored promo by {dup[1]:.1f}s (> {a.max_overlap:.0%} of the shorter clip)"]})
        print(f"      dropped duplicate in ep{p['segments'][0]['ep']}: {dup[1]:.1f}s overlap"); continue
    kept.append(p)
valid = kept
ranked = rank(valid, a.audience)
if a.rerank > 0 and a.judge != "cache" and ranked:
    try:
        ranked, dropped, info = rerank_promos(ranked, by_num, a.judge, a.model, a.base_url, n=a.rerank, drop_max=a.rerank_drop_max)
        rejected += dropped
        for i, p in enumerate(ranked, 1): p["rank"] = i
        print(f"editor pass: {info}")
    except Exception as ex:
        print(f"editor pass FAILED ({ex}) - keeping the score order")
write_outputs(ranked, rejected, a.out, a.audience)
print(f"\n{len(ranked)} promos ranked, {len(rejected)} rejected by validator -> {a.out}/PROMOS.md, promos.csv, promos.json")
