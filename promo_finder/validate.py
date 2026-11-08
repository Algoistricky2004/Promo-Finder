"""Stage 3 (pure code): never trust the judge. Snap to real event boundaries, verify quotes, enforce rules.

A promo from the judge is accepted only if
  * every segment snaps to real script event boundaries (no invented timestamps),
  * no cut lands inside a spoken line,
  * total runtime is 20-60 s,
  * the claimed hook text occurs in the first 3 s, the cliffhanger text in the last 6 s (verbatim, normalised).
Anything else is rejected with a reason -- that is the main guard against LLM hallucinated timestamps/quotes.
"""
import re
from .parse import fmt
from .heuristic import beats, MIN_LEN, MAX_LEN

def _norm(t):
    t = t.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9' ]+", " ", t).split()

def _contains(hay, needle):
    h, n = _norm(hay), _norm(needle)
    if not n: return False
    if len(n) > 14: n = n[:14]          # long quotes: match first 14 words
    return any(h[i:i+len(n)] == n for i in range(len(h) - len(n) + 1))

def to_sec(x):
    if isinstance(x, (int, float)): return float(x)
    x = str(x).strip()
    if ":" in x:
        sec = 0.0
        for q in [float(t) for t in x.split(":")]: sec = sec * 60 + q
        return sec
    return float(x)

def snap_segment(ep, s, e, tol_s=1.0, tol_e=1.0, repair=False):
    bs = beats(ep)
    if repair: tol_s = tol_e = 3.0
    cs = [b for b in bs if s - tol_s <= b.start <= s + 0.6]
    ce = [b for b in bs if e - 0.6 <= b.end <= e + tol_e]
    if not cs: return None, f"start {s} matches no event start (±{tol_s}s)"
    if not ce: return None, f"end {e} matches no event end (±{tol_e}s)"
    s2 = min(cs, key=lambda b: abs(b.start - s)).start
    e2 = min(ce, key=lambda b: abs(b.end - e)).end
    if e2 <= s2: return None, "end before start"
    # no cut inside a spoken line
    for b in bs:
        if b.kind == "DIALOGUE":
            if b.start < s2 - 0.05 and b.end > s2 + 0.05:
                if repair: s2 = b.start
                else: return None, f"start cuts through line at {fmt(b.start)}"
            if b.start < e2 - 0.05 and b.end > e2 + 0.05:
                if repair: e2 = b.end
                else: return None, f"end cuts through line at {fmt(b.start)}"
    return (s2, e2), None

def validate(promo, eps_by_num, repair=False):
    errs, warns, segs = [], [], []
    for sg in promo["segments"]:
        epn = sg.get("ep", promo["ep"])
        ep = eps_by_num.get(epn)
        if ep is None: errs.append(f"unknown episode {epn}"); continue
        try: s_in, e_in = to_sec(sg["start"]), to_sec(sg["end"])
        except Exception: errs.append(f"ep{epn}: unreadable timestamps {sg!r}"); continue
        snapped, err = snap_segment(ep, s_in, e_in, repair=repair)
        if err: errs.append(f"ep{epn}: {err}"); continue
        if abs(snapped[0] - s_in) > 0.05 or abs(snapped[1] - e_in) > 0.05:
            warns.append(f"ep{epn} {'repaired' if repair else 'snapped'} {s_in}-{e_in} -> {snapped[0]:.2f}-{snapped[1]:.2f}")
        segs.append({"ep": epn, "start": snapped[0], "end": snapped[1]})
    if errs: return None, errs, warns
    total = sum(s["end"] - s["start"] for s in segs)
    if total < MIN_LEN or total > MAX_LEN: errs.append(f"runtime {total:.1f}s outside {MIN_LEN:.0f}-{MAX_LEN:.0f}s")
    # hook / cliffhanger text must really be there
    first, last = segs[0], segs[-1]
    fb = [b for b in beats(eps_by_num[first["ep"]]) if first["start"] - 0.01 <= b.start <= first["start"] + 3.0]
    lb = [b for b in beats(eps_by_num[last["ep"]]) if last["end"] - 6.0 <= b.end <= last["end"] + 0.01]
    promo = {**promo, "hook": str(promo.get("hook", "")), "cliffhanger": str(promo.get("cliffhanger", ""))}
    if not any(_contains(b.text, promo["hook"]) for b in fb): errs.append(f"hook text not found in first 3s: {promo['hook']!r}")
    if not any(_contains(b.text, promo["cliffhanger"]) for b in lb): errs.append(f"cliffhanger text not found in last 6s: {promo['cliffhanger']!r}")
    if errs: return None, errs, warns
    out = dict(promo); out["segments"] = segs; out["runtime_s"] = round(total, 1)
    return out, [], warns

def fmt_segments(segs):
    f1 = lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}"
    return " + ".join(f"Ep{s['ep']} {f1(s['start'])}–{f1(s['end'])}" for s in segs)


def anchor(promo, eps_by_num):
    """Rebuild a promo as ONE continuous clip from the model's own hook and cliffhanger quotes.
    The model says WHICH lines are the hook and the cliffhanger (they must exist in the script); code supplies the timestamps.
    Returns None if no hook/cliffhanger pair gives a 20-60 s clip."""
    ep = eps_by_num.get(promo.get("ep"))
    if ep is None: return None
    bs = beats(ep)
    H = [b for b in bs if _contains(b.text, str(promo.get("hook", "")))]
    K = [b for b in bs if _contains(b.text, str(promo.get("cliffhanger", "")))]
    if not H or not K: return None
    try:
        ref_s = min(to_sec(s["start"]) for s in promo["segments"]); ref_e = max(to_sec(s["end"]) for s in promo["segments"])
    except Exception: ref_s = ref_e = 0.0
    best = None
    for h in H:
        for k in K:
            d = k.end - h.start
            if d < MIN_LEN or d > MAX_LEN: continue
            cost = abs(h.start - ref_s) + abs(k.end - ref_e)
            if best is None or cost < best[0]: best = (cost, h, k)
    if not best: return None
    return {**promo, "segments": [{"start": best[1].start, "end": best[2].end}]}
