"""Compare a live-model run with a reference run (e.g. an editor's or a stronger model's picks).
usage: python analysis/live_vs_reference.py LIVE_PROMOS_JSON REF_PROMOS_JSON [NO_PROMO_NOTES_JSON] [LATE_FROM_EP=36] > report.md
Reports agreement, whether the model abstains, score inflation and context handling. No pack needed."""
import json, sys, collections, itertools, statistics as st
live = json.load(open(sys.argv[1]))["promos"]; ref = json.load(open(sys.argv[2]))["promos"]
notes = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3] != "-" else {}
late_from = int(sys.argv[4]) if len(sys.argv) > 4 else 36
ep = lambda p: p["segments"][0]["ep"]
def iou(p, q):
    inter = sum(max(0, min(a["end"], b["end"]) - max(a["start"], b["start"])) for a in p["segments"] for b in q["segments"] if a["ep"] == b["ep"])
    uni = sum(s["end"] - s["start"] for s in p["segments"]) + sum(s["end"] - s["start"] for s in q["segments"]) - inter
    return inter / uni if uni else 0
L = []
L.append(f"# Live run vs reference run\n\n- live promos: {len(live)} in {len({ep(p) for p in live})} episodes; reference promos: {len(ref)} in {len({ep(p) for p in ref})} episodes.")
best = [max((iou(r, l) for l in live if ep(l) == ep(r)), default=0) for r in ref]
L.append(f"- **Agreement.** {sum(b >= .5 for b in best)}/{len(ref)} reference promos are matched by a live promo at IoU >= 0.5 ({sum(b >= .3 for b in best)}/{len(ref)} at >= 0.3); mean best IoU {st.mean(best):.2f}. "
         f"Top-10 episodes shared: {len({ep(p) for p in ref[:10]} & {ep(p) for p in live[:10]})}/10.")
ref_eps = {ep(p) for p in ref}; none_eps = sorted(e for e in {ep(p) for p in live} | set(map(int, notes)) if e not in ref_eps) if not notes else sorted(map(int, notes))
lp = [p for p in live if ep(p) in none_eps]
L.append(f"- **Abstention.** The reference found no promo in {len(none_eps)} episodes; the live model returned {len(lp)} promos there, {sum(p['rank'] <= 20 for p in lp)} of them in its top 20 (ranks {sorted(p['rank'] for p in lp)[:8]}).")
for k in ("hook", "cliff", "standalone", "intensity"):
    c = collections.Counter(p["scores"][k] for p in live); L.append(f"- score `{k}`: {dict(sorted(c.items()))}")
L.append(f"- **Inflation.** cliffhanger = 5 for {sum(p['scores']['cliff'] == 5 for p in live)}/{len(live)} promos; composite >= 0.95 for {sum(p['composite'] >= .95 for p in live)}.")
lt = [p for p in live if ep(p) >= late_from]
L.append(f"- **Context.** Episodes >= {late_from}: {len(lt)} live promos, needs_context = {dict(collections.Counter(p['needs_context'] for p in lt))}.")
L.append("\n## Where each top reference promo sits in the live ranking (same episode: live rank, IoU)")
for r in ref[:10]:
    L.append(f"- ref #{r['rank']} ep{ep(r)}: {[(l['rank'], round(iou(r, l), 2)) for l in live if ep(l) == ep(r)]}")
print("\n".join(L))
