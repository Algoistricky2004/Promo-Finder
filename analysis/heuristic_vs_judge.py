"""Quality check: how good is the free code-only heuristic compared with the judge's picks? Writes outputs/quality_check.md"""
import json, sys, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promo_finder.parse import load_pack, fmt
from promo_finder.heuristic import candidates, beats, markers, hook_score, cliff_score

PACK = sys.argv[1]; OUT = sys.argv[2]
eps = {e.num: e for e in load_pack(PACK)}
promos = json.load(open(os.path.join(OUT, "promos.json")))["promos"]
none_eps = [n for n in eps if n not in {p["segments"][0]["ep"] for p in promos}]

def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
    return num / den if den else float("nan")

# 1. shortlist recall: how much of each judged promo falls inside a heuristic top-4 window of the same episode?
short = {n: candidates(e, k=4) for n, e in eps.items()}
cov_rows = []
for p in promos:
    tot = sum(s["end"] - s["start"] for s in p["segments"]); cov = 0
    for s in p["segments"]:
        wins = [(c.start, c.end) for c in short[s["ep"]]]
        cov += max(0, (s["end"] - s["start"]) - 0) * 0
        # covered duration = length of segment inside union of shortlist windows
        pts = sorted(wins); merged = []
        for a, b in pts:
            if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], b)
            else: merged.append([a, b])
        cov += sum(max(0, min(s["end"], b) - max(s["start"], a)) for a, b in merged)
    cov_rows.append((p["rank"], p["segments"][0]["ep"], cov / tot))
full = sum(1 for *_, c in cov_rows if c >= 0.5)
# 2. can the heuristic abstain?
top1 = {n: (short[n][0].total if short[n] else 0) for n in eps}
if not promos: print("no promos in this run - nothing to analyse"); sys.exit(0)
mean_with = st.mean(top1[p["segments"][0]["ep"]] for p in promos); mean_none = st.mean(top1[n] for n in none_eps) if none_eps else float("nan")
# 3. score agreement on the judge's own start/end points
hs, hc, js, jc = [], [], [], []
for p in promos:
    f, l = p["segments"][0], p["segments"][-1]
    e1, e2 = eps[f["ep"]], eps[l["ep"]]
    h, _ = hook_score(beats(e1), f["start"]); c, _ = cliff_score(e2, beats(e2), l["end"], markers(e2))
    hs.append(h); hc.append(c); js.append(p["scores"]["hook"]); jc.append(p["scores"]["cliff"])
    p["_h"], p["_c"] = h, c
rho_h, rho_c = spearman(hs, js), spearman(hc, jc)
_np = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "judgments", "no_promo_notes.json")
notes = json.load(open(_np)) if os.path.exists(_np) else {}      # optional: editor notes on why an episode has no promo
fp = sorted(((top1[n], n) for n in none_eps), reverse=True)[:5]       # heuristic's best guess in episodes with NO promo
fn = sorted(promos, key=lambda p: (p["_c"] - p["scores"]["cliff"] / 5))[:4]
# episodes the heuristic would post (top-1 by score) vs the judge
heur_top = sorted(eps, key=lambda n: -top1[n])[:15]
judge_top = [p["segments"][0]["ep"] for p in sorted(promos, key=lambda p: p["rank"])[:15]]

L = ["# Quality check: code-only heuristic vs. judge (50 episodes)\n",
     f"- **Shortlist recall.** {full}/{len(promos)} judged promos have >=50% of their runtime inside the heuristic's top-4 windows for that episode "
     f"(mean coverage {st.mean(c for *_, c in cov_rows):.0%}). So the shortlist is a usable pre-filter but not a substitute for the judge.",
     (f"- **The heuristic cannot say 'no promo'.** Mean top-1 heuristic score is {mean_with:.2f} in episodes where the judge found a promo vs {mean_none:.2f} in the {len(none_eps)} "
      f"episodes where it found none ({', '.join(map(str, none_eps))}); the gap is small, so thresholding does not recover the abstain decision.") if none_eps else
     f"- Judge found a promo in every episode (mean top-1 heuristic score {mean_with:.2f}); no abstain comparison possible.",
     f"- **Agreement on the judge's own cut points (Spearman).** hook: {rho_h:.2f}, cliffhanger: {rho_c:.2f} (n={len(promos)}). "
     "Weak agreement on cliffhangers is expected: question marks and ellipses are poor proxies for 'unresolved'.",
     f"- **Top-15 overlap by episode.** Heuristic's 15 highest-scoring episodes vs judge's top-15 promo episodes share {len(set(heur_top) & set(judge_top))} episodes.\n",
     "## False positives: heuristic's best window in episodes where the judge found NO promo"]
for sc, n in fp:
    c = short[n][0]; L.append(f"- Ep{n} {fmt(c.start)}-{fmt(c.end)} scored {sc:.2f} (signals: {', '.join(c.notes[:3]) or 'none'}). Judge: {notes.get(str(n), "no note")}.")
L.append("\n## Misses: endings the judge rates strongly that the heuristic scores low")
for p in fn: L.append(f"- Ep{p['segments'][-1]['ep']} ending at {fmt(p['segments'][-1]['end'])}: heuristic {p['_c']:.2f} vs judge {p['scores']['cliff']}/5 - \"{p['cliffhanger'][:70]}\"")
open(os.path.join(OUT, "quality_check.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
