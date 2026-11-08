"""Parse Kuku screenplay .txt files into structured episodes -> scenes -> events."""
import re, glob, os, json
from dataclasses import dataclass, field, asdict

TS = r"(\d+):(\d+)\.(\d+)"
def _sec(m, i=1):  # mm:ss.cc -> seconds
    return int(m.group(i))*60 + int(m.group(i+1)) + float("0." + m.group(i+2))

@dataclass
class Event:
    kind: str            # CAMERA | ACTION | SOUND | DIALOGUE
    speaker: str | None
    tone: str | None
    start: float
    end: float
    text: str

@dataclass
class Scene:
    idx: int
    heading: str
    start: float
    end: float
    events: list = field(default_factory=list)

@dataclass
class Episode:
    num: int
    logline: str
    continuity: str
    duration: float
    scenes: list = field(default_factory=list)
    path: str = ""

def fmt(t):
    t = max(0, t); return f"{int(t//60):02d}:{t%60:05.2f}"

def parse_episode(path):
    raw = open(path, encoding="utf-8").read()
    num = int(re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", raw).group(1))
    logline = (re.search(r"EPISODE LOGLINE:\s*(.+)", raw) or [None, ""])[1].strip()
    cont = (re.search(r"CONTINUITY NOTES:\s*(.+)", raw) or [None, ""])[1].strip()
    body = raw.split("--- SCRIPT START ---", 1)[-1].split("--- SCRIPT END", 1)[0]
    ep = Episode(num, logline, cont, 0.0, path=os.path.basename(path))
    for sc in re.split(r"\n(?=SCENE \d+\s*\n)", body.strip()):
        m = re.match(r"SCENE (\d+)\s*\nScene heading:\s*(.+)\nScene start:\s*" + TS + r"\nScene end:\s*" + TS, sc, re.I)
        if not m: continue
        scene = Scene(int(m.group(1)), m.group(2).strip(), _sec(m, 3), _sec(m, 6))
        # event blocks: **LABEL (tone):**\nStart: (..) | End: (..)\n<text>
        for em in re.finditer(r"\*\*(.+?)(?:\s*\((.*?)\))?:\*\*\s*\nStart:\s*\(" + TS + r"\)\s*\|\s*End:\s*\(" + TS + r"\)\s*\n(.*?)(?=\n\*\*|\Z)", sc, re.S):
            label, tone = em.group(1).strip(), em.group(2)
            s, e = _sec(em, 3), _sec(em, 6)
            text = em.group(9).strip()
            if label == "CAMERA": kind, spk = "CAMERA", None
            elif label == "ACTION": kind, spk = "ACTION", None
            elif label.startswith("SOUND"): kind, spk = "SOUND", None
            else: kind, spk = "DIALOGUE", label
            scene.events.append(Event(kind, spk, tone, s, e, text))
        scene.events.sort(key=lambda x: (x.start, x.end))
        ep.scenes.append(scene)
    ep.duration = max((s.end for s in ep.scenes), default=0.0)
    return ep

def load_pack(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    eps = [parse_episode(f) for f in files]
    return sorted(eps, key=lambda e: e.num)

if __name__ == "__main__":
    import sys
    eps = load_pack(sys.argv[1])
    print(len(eps), "episodes")
    for e in eps:
        nd = sum(1 for s in e.scenes for ev in s.events if ev.kind=="DIALOGUE")
        print(f"Ep{e.num:>3} dur={fmt(e.duration)} scenes={len(e.scenes)} dialogue={nd} events={sum(len(s.events) for s in e.scenes)}")
