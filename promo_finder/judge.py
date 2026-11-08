"""Stage 2: the judge. One call per episode. Model is a plug-in; output is validated by validate.py.

Providers (zero third-party deps, plain HTTPS):
  anthropic  -> https://api.anthropic.com/v1/messages          (env ANTHROPIC_API_KEY)
  openai     -> any OpenAI-compatible /chat/completions         (env OPENAI_API_KEY, --base-url; works for Gemini's
                OpenAI-compatible endpoint, OpenRouter, vLLM, Ollama ...)
  cache      -> replays a saved judgments JSON (offline; used to re-run the pipeline without an API)
Swap the model with --model / env PROMO_MODEL. Nothing else changes.
"""
import json, os, re, time, urllib.request, urllib.error
from .compact import compact_episode, t

SYSTEM = """You are a promo editor for vertical microdrama (ReelShort / DramaBox style). You read an episode screenplay with timestamps and pick the best PROMO clips for Instagram/TikTok/YouTube to pull in NEW viewers.

A promo is 20-60 seconds TOTAL, built from one continuous stretch or up to 4 short segments of the same episode, in order. A good promo:
1. HOOK: in the first 3 seconds, something that stops a scroller (a slap, reveal, shocking line, threat, accusation).
2. CLIFFHANGER: cuts at an unresolved moment so the viewer needs the next part. Cut BEFORE the answer/payoff.
3. STANDS ALONE: a stranger who never saw the show can follow who is fighting whom and what is at stake.

Rules:
- Use ONLY timestamps that appear in the script. Start at the start of a line/action, end at the end of a line/action. Never cut inside a spoken line.
- `hook` and `cliffhanger` must be copied VERBATIM from the script (a spoken line, or the start of an action line). They are machine-checked; invented text is rejected.
- Return 0 promos if the episode has none worth posting (quiet, exposition-only, resolved). Do not pad. Max 3 per episode.
- Be honest in scores: 1-5 for hook, cliff, standalone, intensity. needs_context: none | some | heavy (heavy = a cold viewer would be lost).
- Flag content a platform or editor should double check (e.g. torture, sexual content) in `flags`.

Return ONLY JSON: {"promos":[{"segments":[{"start":<sec>,"end":<sec>}],"hook":"...","cliffhanger":"...","why":"one line",
"scores":{"hook":n,"cliff":n,"standalone":n,"intensity":n},"needs_context":"none|some|heavy","flags":[]}]}
ALL timestamps in the script are in SECONDS (71.3 means 1 minute 11.3 seconds). Answer in the same unit: copy start/end values exactly as printed in the script.
LENGTH RULE: runtime = the SUM of (end - start) over all segments and must be between 20 and 60 seconds. A single 10-second exchange is too short: extend it with the lines before/after it in the same scene, or add a second segment. Check your arithmetic before answering."""

def build_user_prompt(ep, prev_continuity="", shortlist=None):
    secs = lambda x: f"{x:.1f}"
    p = [f"EPISODE {ep.num}  (runtime {t(ep.duration)})", f"Logline: {ep.logline}", f"Continuity: {ep.continuity}"]
    if prev_continuity: p.append(f"Story so far (previous episode): {prev_continuity}")
    if shortlist:
        p.append("Code-generated shortlist of windows with high surface signals (a hint, not a limit; you may pick others):")
        for c in shortlist: p.append(f"  - {secs(c.start)}s to {secs(c.end)}s  (hook {c.hook:.2f}, cliff {c.cliff:.2f})")
    p.append("\nSCRIPT (ALL times are in SECONDS, e.g. 71.3 = 1 min 11.3 s; answer in the same unit):\n" + compact_episode(ep, action_chars=220, seconds=True))
    return "\n".join(p)

def _post(url, headers, body, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
            if e.code not in (429, 500, 502, 503, 504, 529): raise last
        except (TimeoutError, OSError, urllib.error.URLError) as e:
            last = RuntimeError(f"network error: {e!r}")
        time.sleep(2 ** attempt * float(os.environ.get("PROMO_RETRY_BASE", "3")))   # backoff seconds; tests set 0
    raise last

def call_anthropic(system, user, model, **kw):
    r = _post(os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages",
              {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
              {"model": model, "max_tokens": 4000, "system": system, "messages": [{"role": "user", "content": user}]})
    return "".join(b.get("text", "") for b in r["content"])

def call_openai(system, user, model, base_url="https://api.openai.com/v1", **kw):
    r = _post(base_url.rstrip("/") + "/chat/completions",
              {"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none"), "content-type": "application/json"},
              {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **({"temperature": 0.2} if re.match(r"gpt-[34]", model) else {}),
               **({"response_format": {"type": "json_object"}} if "api.openai.com" in base_url else {})})
    return r["choices"][0]["message"]["content"]

def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))

def judge_episode(ep, provider, model, prev_continuity="", shortlist=None, base_url=None, cache=None):
    if provider == "cache":
        return cache.get(str(ep.num), [])
    user = build_user_prompt(ep, prev_continuity, shortlist)
    retry = "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
    for attempt in range(2):
        u = user if attempt == 0 else user + retry
        txt = call_openai(SYSTEM, u, model, base_url=base_url or "https://api.openai.com/v1") if provider == "openai" else call_anthropic(SYSTEM, u, model)
        try: return parse_json(txt).get("promos", [])
        except Exception: continue
    return []


def fix_episode(ep, provider, model, bad, prev_continuity="", shortlist=None, base_url=None):
    """One self-correction round: show the model what the checker rejected and why."""
    user = build_user_prompt(ep, prev_continuity, shortlist)
    probs = "\n".join(f"- promo {i+1}: " + json.dumps({k: r["promo"].get(k) for k in ("segments", "hook", "cliffhanger")}) +
                      "\n  problems: " + "; ".join(r["errors"]) for i, r in enumerate(bad))
    user += ("\n\nYOUR PREVIOUS ANSWER FOR THIS EPISODE WAS REJECTED BY A CHECKER:\n" + probs +
             "\nReturn corrected promos in the same JSON format. Use only SECONDS exactly as printed in the script. Runtime (sum of end-start) must be 20-60 s. "
             "The hook text must be spoken or described in the first 3 s of the clip and the cliffhanger text in the last 6 s, copied verbatim. "
             "Return an empty list if no valid promo exists.")
    txt = call_openai(SYSTEM, user, model, base_url=base_url or "https://api.openai.com/v1") if provider == "openai" else call_anthropic(SYSTEM, user, model)
    try: return parse_json(txt).get("promos", [])
    except Exception: return []


SELECT_SYSTEM = """You are a promo editor for vertical microdrama (ReelShort / DramaBox style). Below are CANDIDATE CLIPS from one episode. Each is already a valid 20-60 second continuous clip with exact start and end. Pick the best ones (0 to 3) to post on Instagram/TikTok/YouTube to pull in NEW viewers.

A good promo: HOOK in the first 3 seconds (a slap, reveal, shocking line, threat); ends on an unresolved CLIFFHANGER; a stranger can follow who is fighting whom and what is at stake.
Judge each candidate only by what its text shows, including how it STARTS and how it ENDS. Return 0 picks if none is good.
SCORING RUBRIC (1-5, use the full range; most candidates deserve 2-3; give a 5 only for a standout):
- hook: 5 = the very first line or action is a slap, reveal, threat or shocking line. 3 = decent conflict but a slow or generic opening. 1 = exposition or small talk.
- cliff: look ONLY at the last 2-3 lines. 5 = something is cut off unresolved (an unanswered question, a threat about to land, an arrival or reveal interrupted, a decision pending). 1-2 = the scene resolves there (a speech finished, an agreement, comfort, a mic-drop line) - that is NOT a cliffhanger, however strong the line. In "why", name the actual last line and say what is left unresolved; never call a statement a question.
- standalone: 5 = a stranger understands who is fighting whom and why. 1 = needs earlier episodes.
- intensity: emotional or physical stakes per second.
Rank by how well the clip would work as a promo for a NEW viewer; two picks from the same episode must be clearly different moments.

Return ONLY JSON: {"picks":[{"id":<candidate id>,"scores":{"hook":n,"cliff":n,"standalone":n,"intensity":n},"needs_context":"none|some|heavy","why":"one line","flags":[]}]}"""

def _window_beats(ep, c):
    from .heuristic import beats
    return [b for b in beats(ep) if b.start >= c.start - 0.01 and b.end <= c.end + 0.01]

def build_select_prompt(ep, cands, prev_continuity=""):
    p = [f"EPISODE {ep.num}", f"Logline: {ep.logline}", f"Continuity: {ep.continuity}"]
    if prev_continuity: p.append(f"Story so far (previous episode): {prev_continuity}")
    for i, c in enumerate(cands, 1):
        lines = [f"[{b.start:.1f}-{b.end:.1f}] " + (f"{b.speaker}: " if b.kind == "DIALOGUE" else "ACTION: ") + b.text[:200] for b in _window_beats(ep, c)]
        p.append(f"\n=== CANDIDATE {i}  ({c.end - c.start:.0f}s, from {c.start:.1f}s to {c.end:.1f}s) ===\n" + "\n".join(lines))
    return "\n".join(p)

def select_episode(ep, provider, model, cands, prev_continuity="", base_url=None):
    """Model picks among code-built windows. Timestamps and runtime are valid by construction; the model only judges."""
    user = build_select_prompt(ep, cands, prev_continuity)
    txt = (call_openai(SELECT_SYSTEM, user, model, base_url=base_url or "https://api.openai.com/v1") if provider == "openai"
           else call_anthropic(SELECT_SYSTEM, user, model))
    try: picks = parse_json(txt).get("picks", [])
    except Exception: return []
    out, seen = [], set()
    for pk in picks:
        try: i = int(pk["id"])
        except Exception: continue
        if not 1 <= i <= len(cands) or i in seen: continue
        seen.add(i); c = cands[i - 1]; win = _window_beats(ep, c)
        if not win: continue
        first3 = [b for b in win if b.start <= c.start + 3.0]
        sc = pk.get("scores", {})
        try: scores = {k: max(1, min(5, int(sc[k]))) for k in ("hook", "cliff", "standalone", "intensity")}
        except Exception: continue
        ctx = pk.get("needs_context"); fl = pk.get("flags", [])
        out.append({"segments": [{"start": c.start, "end": c.end}], "hook": (first3[0] if first3 else win[0]).text.strip().strip('"\u201c\u201d'), "cliffhanger": win[-1].text.strip().strip('"\u201c\u201d'),
                    "why": str(pk.get("why", "")), "scores": scores, "needs_context": ctx if ctx in ("none", "some", "heavy") else "some",
                    "flags": fl if isinstance(fl, list) else []})
    return out
