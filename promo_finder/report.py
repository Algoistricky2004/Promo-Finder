"""Stage 4 (pure code): deterministic ranking + output files. Ranking is code over judge scores so it is auditable."""
import csv, json, os
from .validate import fmt_segments
from .parse import fmt

W = {"hook": 0.30, "cliff": 0.30, "standalone": 0.25, "intensity": 0.15}
CONTEXT_PENALTY = {"none": 0.0, "some": 0.04, "heavy": 0.10}

def composite(p, audience="cold"):
    s = p["scores"]
    base = sum(W[k] * s[k] for k in W) / 5.0
    pen = CONTEXT_PENALTY.get(p.get("needs_context", "none"), 0.0) if audience == "cold" else 0.0
    return round(base - pen, 4)

def rank(promos, audience="cold"):
    for p in promos: p["composite"] = composite(p, audience)
    ranked = sorted(promos, key=lambda p: (-p["composite"], p["segments"][0]["ep"]))
    for i, p in enumerate(ranked, 1): p["rank"] = i
    return ranked

def write_outputs(ranked, rejected, outdir, audience):
    os.makedirs(outdir, exist_ok=True)
    json.dump({"audience": audience, "promos": ranked, "rejected_by_validator": rejected},
              open(os.path.join(outdir, "promos.json"), "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "promos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "composite", "runtime_s", "segments", "hook", "cliffhanger", "why", "hook_1-5", "cliff_1-5",
                    "standalone_1-5", "intensity_1-5", "needs_context", "flags", "editor_note"])
        for p in ranked:
            s = p["scores"]
            w.writerow([p["rank"], p["composite"], p["runtime_s"], fmt_segments(p["segments"]), p["hook"], p["cliffhanger"],
                        p["why"], s["hook"], s["cliff"], s["standalone"], s["intensity"], p.get("needs_context", "none"),
                        "; ".join(p.get("flags", [])), p.get("editor_note", "")])
    L = [f"# Ranked promos ({len(ranked)}), audience = {audience}\n"]
    for p in ranked:
        L.append(f"## #{p['rank']}  {fmt_segments(p['segments'])}  ({p['runtime_s']}s, score {p['composite']})")
        L.append(f"- **Hook:** {p['hook']}\n- **Cliffhanger:** {p['cliffhanger']}\n- **Why it works:** {p['why']}")
        s = p["scores"]
        L.append(f"- Scores (1-5): hook {s['hook']}, cliff {s['cliff']}, standalone {s['standalone']}, intensity {s['intensity']}; context needed: {p.get('needs_context','none')}")
        if p.get("flags"): L.append(f"- Flags: {'; '.join(p['flags'])}")
        if p.get("editor_note"): L.append(f"- Editor pass: {p['editor_note']}")
        L.append("")
    open(os.path.join(outdir, "PROMOS.md"), "w", encoding="utf-8").write("\n".join(L))
