"""Stage 5: editor pass. Absolute 1-5 scores saturate (a small model gave cliffhanger=5 to 74% of promos), so the top of the list is
decided by comparing clips against EACH OTHER in one call. The model may also drop clips it would not post.
Safe by construction: the model only returns ids; any invalid / partial / failed answer falls back to the original order."""
import json, random
from .judge import call_openai, call_anthropic, parse_json
from .heuristic import beats

RERANK_SYSTEM = """You are the head editor choosing which promo clips to post FIRST for a vertical microdrama (ReelShort / DramaBox style). Each clip is 20-60 seconds. The audience is NEW: they have seen nothing before the clip.

Order the clips from best to worst promo. A top clip: opens on a hook (slap, reveal, threat, shocking line); cuts at an UNRESOLVED moment (an unanswered question, a threat about to land, an interrupted arrival or reveal); makes sense to a stranger.
DROP a clip (do not post it at all) if: the ending resolves the scene (a finished speech, an agreement, comfort); the hook is weak; it only makes sense with earlier episodes; or it shows nearly the same moment as a better clip. It is fine to drop many weak clips.
Compare the clips with each other; do not rate them in isolation. The clip numbers are random and carry no meaning.

Return ONLY JSON: {"order":[clip numbers, best first],"drop":[clip numbers],"notes":{"<clip number>":"one short reason"}}. Every clip number must appear exactly once in "order" or "drop"."""

def _lines(by_num, seg):
    ep = by_num[seg["ep"]]
    return [(f"{b.speaker}: " if b.kind == "DIALOGUE" else "ACTION: ") + b.text.strip()[:140]
            for b in beats(ep) if b.start >= seg["start"] - 0.01 and b.end <= seg["end"] + 0.01]

def _describe(by_num, p, cid):
    first = _lines(by_num, p["segments"][0]); last = _lines(by_num, p["segments"][-1])
    if len(p["segments"]) == 1 and len(first) <= 6: shown = first
    else: shown = first[:3] + ["..."] + last[-3:]
    ep = by_num[p["segments"][0]["ep"]]
    ctx = f" | model says context needed: {p.get('needs_context', '?')}"
    return (f"=== CLIP {cid} (episode {ep.num}, {p['runtime_s']:.0f}s{ctx}) ===\nEpisode logline: {ep.logline}\n" + "\n".join(shown))

def rerank_promos(promos, by_num, provider, model, base_url=None, n=40, drop_max=0.5, seed=0):
    """promos: list already ordered best-first. Returns (new_order, dropped, info)."""
    top, rest = promos[:n], promos[n:]
    if len(top) < 3: return promos, [], "skipped (fewer than 3 promos)"
    pres = random.Random(seed).sample(range(len(top)), len(top))            # shuffled so list position cannot bias the model
    user = "\n\n".join(_describe(by_num, top[pres[k]], k + 1) for k in range(len(top)))
    txt = (call_openai(RERANK_SYSTEM, user, model, base_url=base_url or "https://api.openai.com/v1") if provider == "openai"
           else call_anthropic(RERANK_SYSTEM, user, model))
    try: ans = parse_json(txt)
    except Exception: return promos, [], "fallback: reply was not valid JSON"
    valid = set(range(1, len(top) + 1)); seen = set(); order = []
    for x in ans.get("order", []):
        try: x = int(x)
        except Exception: continue
        if x in valid and x not in seen: seen.add(x); order.append(x)
    drops = []
    for x in ans.get("drop", []):
        try: x = int(x)
        except Exception: continue
        if x in valid and x not in drops: drops.append(x)
    if not order and not drops: return promos, [], "fallback: no usable ids in the reply"
    drops = [x for x in drops]; cap = int(drop_max * len(top))
    forced_keep = drops[cap:]; drops = drops[:cap]                           # never drop more than drop_max of the list
    notes = ans.get("notes", {}) if isinstance(ans.get("notes", {}), dict) else {}
    note = lambda k: str(notes.get(str(k), notes.get(k, "")))[:200]
    order = [x for x in order if x not in drops]
    placed = set(order) | set(drops)
    # clips the model forgot (or wanted dropped beyond the cap) go after the ordered ones, in their original relative order
    tail = sorted((k for k in valid if k not in placed), key=lambda k: pres[k - 1])
    new_top = []
    for k in order + tail:
        p = top[pres[k - 1]]; p["editor_note"] = note(k); new_top.append(p)
    dropped = []
    for k in drops:
        p = top[pres[k - 1]]; p["editor_note"] = note(k)
        dropped.append({"ep": p["segments"][0]["ep"], "promo": p, "errors": ["dropped by editor pass: " + (note(k) or "no reason given")]})
    return new_top + rest, dropped, f"ordered {len(order)} of {len(top)}, dropped {len(drops)}" + (f", kept {len(forced_keep)} extra drops (cap {drop_max:.0%})" if forced_keep else "")
