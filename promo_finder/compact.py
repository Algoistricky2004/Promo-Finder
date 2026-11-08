"""Compact, token-cheap transcript view (dialogue + actions) used in LLM prompts."""
from .parse import fmt

def t(x): return f"{int(x//60)}:{x%60:04.1f}"

def compact_episode(ep, start=None, end=None, action_chars=150, seconds=False):
    f = (lambda x: f"{x:.1f}") if seconds else t
    lines = []
    for sc in ep.scenes:
        lines.append(f"## S{sc.idx:02d} {sc.heading} [{f(sc.start)}-{f(sc.end)}]")
        for ev in sc.events:
            if start is not None and (ev.end < start or ev.start > end): continue
            if ev.kind in ("CAMERA", "SOUND"): continue
            if ev.kind == "DIALOGUE":
                tone = f" ({ev.tone})" if ev.tone else ""
                lines.append(f"[{f(ev.start)}-{f(ev.end)}] {ev.speaker}{tone}: {ev.text}")
            else:
                lines.append(f"[{f(ev.start)}-{f(ev.end)}] ACTION: {ev.text[:action_chars]}")
    return "\n".join(lines)
