"""Compare two promos.json runs (e.g. cached judge vs a live model, or model A vs model B).
usage: python analysis/compare_runs.py outA/promos.json outB/promos.json
For each promo in A: best temporal IoU with any promo of B in the same episode. Reports matched share at IoU>=0.5."""
import json, sys
A = json.load(open(sys.argv[1]))["promos"]; B = json.load(open(sys.argv[2]))["promos"]
def spans(p): return [(s["ep"], s["start"], s["end"]) for s in p["segments"]]
def iou(p, q):
    inter = uni = 0.0
    for ep in {e for e, *_ in spans(p)} | {e for e, *_ in spans(q)}:
        a = [(s, e) for x, s, e in spans(p) if x == ep]; b = [(s, e) for x, s, e in spans(q) if x == ep]
        for s1, e1 in a:
            for s2, e2 in b: inter += max(0, min(e1, e2) - max(s1, s2))
        uni += sum(e - s for s, e in a) + sum(e - s for s, e in b)
    uni -= inter
    return inter / uni if uni else 0
best = [max((iou(p, q) for q in B), default=0) for p in A]
m = sum(1 for x in best if x >= 0.5)
print(f"A: {len(A)} promos, B: {len(B)} promos; {m}/{len(A)} of A matched by B at IoU>=0.5 (mean best IoU {sum(best)/len(best):.2f})")
epsA = {p['segments'][0]['ep'] for p in A}; epsB = {p['segments'][0]['ep'] for p in B}
print(f"episodes with a promo: A={len(epsA)}, B={len(epsB)}, both={len(epsA & epsB)}")
