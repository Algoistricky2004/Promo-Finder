"""Stage 1 (pure code, free, deterministic): generate and score candidate promo windows.

This is deliberately a *recall + pre-filter* stage. It cannot understand the story; it scores
surface signals (hook words, question/ellipsis endings, script's own 'cliffhanger'/'freeze' markers,
dialogue density, dead air, name load). The LLM judge (judge.py) does the real judging on the
shortlist, and validate.py checks everything it returns against the script.
"""
import re, math
from dataclasses import dataclass, field

MIN_LEN, MAX_LEN = 20.0, 60.0

# Generic melodrama/thriller hook vocabulary (English). No setting words (titles, period nouns) - it must not be tuned to one show.
# Per-genre lexicons should be loaded from config at scale; see README "Known limits".
HOOK_WORDS = r"guards?|arrest|poison|dead|die[sd]?|kill|hang|execut|marry|marrying|husband|wife|betray|secret|lie[sd]?|liar|blood|who are you|how dare|get out|leave|never|proof|confess|traitor|fire|burn|explo|prison|chain|ruin|worthless|imposter|alive|stole|planted|swear|vow|enough|forbidden|pregnan|divorce|fired|police|lawyer|truth|revenge"
VIOLENCE = r"slap|strike|strikes|struck|shove|hurl|fling|flings|slam|drag|tackle|collaps|snatch|rips?|crush|smash|burst|fireball|explod|lunge|choke|throat|iron|whip|crop|knife|blade|sword|club|shatter"
SHOUT_TONE = r"shout|scream|roar|furious|venom|menac|cold|vicious|savage|hysteric|enraged|boom|command|threat|sinister|ominous|lethal|dangerous|gloat|triumph"
CLOSURE = r"^(thank you|thank god|goodnight|very well|it is enough|i accept|we will see|we'll see|then we will face|yes,? my lady|i am sorry)\b"
CLIFF_MARK = r"cliffhanger|cut to black|cuts? (sharply|to black)|freez|holds? on|fade to black|fades? to black|smash cut|final beat"
# whole-word-start matching: "ring" must not fire on "bring"/"during"/"string", "iron" not on "environment"
HOOK_WORDS, VIOLENCE, SHOUT_TONE, CLIFF_MARK = (r"\b(?:" + x + ")" for x in (HOOK_WORDS, VIOLENCE, SHOUT_TONE, CLIFF_MARK))
_WB = True
FUNC_SPEAKER = {"guard","royal guard","judge","priest","maid","lord","constable","housekeeper","informant"}

def _words(t): return re.findall(r"[A-Za-z']+", t)

def norm_name(s):
    """Show-agnostic speaker normalisation: strip parentheticals and honorifics. (No per-show alias tables.)"""
    s = re.sub(r"\(.*?\)", "", s).strip().upper()
    return re.sub(r"^(LADY|LORD|THE|MISS|DUKE|DUCHESS|SIR|MR|MRS|DR)\s+", "", s)

@dataclass
class Cand:
    ep: int; start: float; end: float
    hook: float; cliff: float; standalone: float; intensity: float; total: float
    notes: list = field(default_factory=list)

def beats(ep):
    ev = [e for s in ep.scenes for e in s.events if e.kind in ("DIALOGUE", "ACTION")]
    return sorted(ev, key=lambda e: (e.start, e.end))

def markers(ep):
    """script-side cliffhanger markers (SOUND / CAMERA / ACTION text, e.g. "scene freezes", "cut to black")"""
    out = []
    for s in ep.scenes:
        for e in s.events:
            if e.kind in ("SOUND", "CAMERA", "ACTION") and re.search(CLIFF_MARK, e.text, re.I):
                out.append((e.start, e.end))
    return out

def hook_score(bs, s):
    first = [b for b in bs if s - 0.01 <= b.start <= s + 3.0]
    if not first: return 0.0, ["no event in first 3s"]
    sc, notes = 0.0, []
    d = [b for b in first if b.kind == "DIALOGUE"]
    if d:
        sc += 0.30; l = d[0]
        n = len(_words(l.text))
        if n <= 10: sc += 0.10
        if re.search(SHOUT_TONE, l.tone or "", re.I): sc += 0.15; notes.append("intense tone")
        if "!" in l.text or "?" in l.text: sc += 0.05
        if n <= 2 and not re.search(HOOK_WORDS, l.text, re.I): sc -= 0.20; notes.append("weak 1-2 word opener")
        if "V.O." in (l.tone or "") or "V.O." in l.speaker: sc -= 0.10
    txt = " ".join(b.text for b in first)
    if re.search(HOOK_WORDS, txt, re.I): sc += 0.25; notes.append("hook lexicon")
    if any(b.kind == "ACTION" for b in first) and re.search(VIOLENCE, txt, re.I): sc += 0.20; notes.append("physical beat")
    return max(0.0, min(1.0, sc)), notes

def cliff_score(ep, bs, e, mk):
    last = [b for b in bs if e - 6.0 <= b.end <= e + 0.01]
    if not last: return 0.0, ["no event in last 6s"]
    sc, notes = 0.0, []
    lb = last[-1]
    t = lb.text.strip()
    if lb.kind == "DIALOGUE":
        if t.rstrip('"').endswith("?"): sc += 0.25; notes.append("ends on question")
        if re.search(r"(\.\.\.|…|—|-)\"?$", t): sc += 0.25; notes.append("ends on trailing/cut-off line")
        if re.search(CLOSURE, t.strip('"').lower()): sc -= 0.25; notes.append("closure line")
        if "V.O." in (lb.tone or "") or "V.O." in lb.speaker: sc -= 0.15; notes.append("ends on V.O. reflection")
        if re.search(r"tender|soft|gentle|peaceful|warm|reflective", lb.tone or "", re.I): sc -= 0.10
    else:
        if re.search(VIOLENCE, t, re.I): sc += 0.20; notes.append("ends on physical beat")
        if re.search(r"burst|doors? (open|swing)|steps? (in|forward)|emerg|appear|arrive|enters?", t, re.I): sc += 0.20; notes.append("arrival/entrance")
    if re.search(HOOK_WORDS, " ".join(b.text for b in last[-2:]), re.I): sc += 0.15
    if any(a <= e + 1.0 and b >= e - 4.0 for a, b in mk): sc += 0.30; notes.append("script cliffhanger marker")
    if e >= ep.duration - 3.0: sc += 0.10; notes.append("episode end")
    return max(0.0, min(1.0, sc)), notes

def standalone_score(ep, bs, s, e, early_bias):
    win = [b for b in bs if b.start >= s - 0.01 and b.end <= e + 0.01]
    if not win: return 0.0, []
    names = {norm_name(b.speaker) for b in win if b.kind == "DIALOGUE"}
    sc, notes = 0.5, []
    sc += 0.2 if len(names) <= 3 else (0.0 if len(names) == 4 else -0.15)
    dl = [b for b in win if b.kind == "DIALOGUE"]
    if dl:
        avg_w = sum(len(_words(b.text)) for b in dl) / len(dl)
        if avg_w > 14: sc -= 0.15; notes.append("exposition-heavy")
    caps = [w for b in win for w in re.findall(r"(?<![.!?\"] )(?<!^)\b[A-Z][a-z]{3,}\b", b.text)]
    if len(caps) / max(1, sum(len(_words(b.text)) for b in win)) > 0.05: sc -= 0.15; notes.append("many proper nouns")
    scn = [sc_ for sc_ in ep.scenes if sc_.start <= s + 0.5 and sc_.end >= e - 0.5]
    if scn: sc += 0.1
    sc += early_bias * (1 - (ep.num - 1) / 49)
    return max(0.0, min(1.0, sc)), notes

def intensity_score(bs, s, e):
    win = [b for b in bs if b.start >= s - 0.01 and b.end <= e + 0.01 and b.kind == "DIALOGUE"]
    if not win: return 0.0, ["no dialogue"]
    dens = len(win) / (e - s)
    sc = min(1.0, dens / 0.5) * 0.5
    sc += 0.3 * sum(1 for b in win if re.search(SHOUT_TONE, b.tone or "", re.I)) / len(win)
    sc += 0.2 * sum(1 for b in win if "!" in b.text or "?" in b.text) / len(win)
    gaps = [win[i+1].start - win[i].end for i in range(len(win)-1)]
    notes = []
    if gaps and max(gaps) > 8: sc -= 0.2; notes.append("dead air >8s")
    return max(0.0, min(1.0, sc)), notes

def candidates(ep, early_bias=0.10, k=3, iou=0.35):
    bs = beats(ep); mk = markers(ep)
    dl = [b for b in bs if b.kind == "DIALOGUE"]
    inside = lambda t: any(b.start < t - 0.05 and b.end > t + 0.05 for b in dl)   # t falls inside a spoken line
    starts = sorted({b.start for b in bs if not inside(b.start)})
    ends = sorted({b.end for b in bs if not inside(b.end)})
    pool = []
    for s in starts:
        h, hn = hook_score(bs, s)
        if h < 0.3: continue
        for e in ends:
            L = e - s
            if L < MIN_LEN or L > MAX_LEN: continue
            c, cn = cliff_score(ep, bs, e, mk)
            st, sn = standalone_score(ep, bs, s, e, early_bias)
            it, inn = intensity_score(bs, s, e)
            length_bonus = 0.05 if 25 <= L <= 45 else 0.0
            tot = 0.30*h + 0.30*c + 0.20*st + 0.20*it + length_bonus
            pool.append(Cand(ep.num, s, e, h, c, st, it, tot, hn + cn + sn + inn))
    pool.sort(key=lambda c: -c.total)
    keep = []
    for c in pool:
        ok = True
        for k_ in keep:
            inter = max(0, min(c.end, k_.end) - max(c.start, k_.start))
            uni = max(c.end, k_.end) - min(c.start, k_.start)
            if inter / uni > iou: ok = False; break
        if ok: keep.append(c)
        if len(keep) >= k: break
    return keep
