import glob, json, os, re, sys, threading, unittest, copy
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promo_finder.parse import load_pack
from promo_finder.validate import validate, anchor
from promo_finder.judge import judge_episode
PACK = os.environ.get("PACK", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pack"))
EPS = {e.num: e for e in load_pack(PACK)} if os.path.isdir(PACK) else {}
_JP = os.path.join(os.path.dirname(__file__), "..", "judgments", "judgments.json")
GOOD = json.load(open(_JP)) if os.path.exists(_JP) else None      # saved judgments are not in the public repo (they quote Kuku's scripts)
HAVE = GOOD is not None and os.path.isdir(PACK)
def base(ep=3):
    p = copy.deepcopy(GOOD[str(ep)][0]); p["ep"] = ep; return p

@unittest.skipUnless(HAVE, 'needs ./pack (Kuku scripts) and judgments/judgments.json')
class Parser(unittest.TestCase):
    def test_every_event_parsed(self):
        for f in glob.glob(PACK + "/**/*.txt", recursive=True):
            raw = open(f, encoding="utf-8").read()
            n = len(re.findall(r"^Start: \(", raw, re.M))
            num = int(re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", raw).group(1))
            self.assertEqual(n, sum(len(s.events) for s in EPS[num].scenes), f)
    def test_50_episodes(self): self.assertEqual(len(EPS), 50)

@unittest.skipUnless(HAVE, 'needs ./pack (Kuku scripts) and judgments/judgments.json')
class Validator(unittest.TestCase):
    def ok(self, p): return validate(p, EPS)[0] is not None
    def test_good_promo_passes(self): self.assertTrue(self.ok(base()))
    def test_all_saved_judgments_pass(self):
        for k, v in GOOD.items():
            for p in v: q = dict(p); q["ep"] = int(k); self.assertTrue(self.ok(q), k)
    def test_invented_timestamp_rejected(self):
        p = base(); p["segments"][0]["start"] += 7.37; self.assertFalse(self.ok(p))
    def test_cut_inside_line_rejected(self):
        # real bug caught in this project: a segment starting at 1:16.2 in ep19 lands inside the line spoken 1:13.5-1:16.4
        p = base(19); p["segments"][1]["start"] = 76.2
        out, errs, _ = validate(p, EPS)
        self.assertIsNone(out); self.assertTrue(any("cuts through line" in e for e in errs), errs)
    def test_near_miss_snaps_not_rejects(self):
        p = base(1); p["segments"][0]["start"] = 12.0     # within 1s of a real boundary (11.3) -> snapped, with a warning
        out, errs, warns = validate(p, EPS)
        self.assertIsNotNone(out); self.assertAlmostEqual(out["segments"][0]["start"], 11.3, 2); self.assertTrue(warns)
    def test_too_short_rejected(self):
        p = base(); p["segments"] = p["segments"][:1]; self.assertFalse(self.ok(p))
    def test_too_long_rejected(self):
        p = base(1); p["segments"] = [{"start": 11.3, "end": 85.0}]; self.assertFalse(self.ok(p))
    def test_fake_hook_rejected(self):
        p = base(); p["hook"] = "I will destroy your entire family."; self.assertFalse(self.ok(p))
    def test_fake_cliffhanger_rejected(self):
        p = base(); p["cliffhanger"] = "The king is dead."; self.assertFalse(self.ok(p))
    def test_unknown_episode_rejected(self):
        p = base(); p["segments"][0]["ep"] = 99; self.assertFalse(self.ok(p))
    def test_hook_quote_outside_first_3s_rejected(self):
        p = base(); p["hook"] = p["cliffhanger"]; self.assertFalse(self.ok(p))   # a line from the END cannot be the hook

@unittest.skipUnless(HAVE, 'needs ./pack (Kuku scripts) and judgments/judgments.json')
class Anchor(unittest.TestCase):
    def test_montage_is_rebuilt_as_one_clip_from_quotes(self):
        # shape of a real small-model answer: a few ~2 s segments (<20 s total) around a valid hook and cliffhanger
        g = base(7)   # a single continuous promo (hook and cliffhanger ~24 s apart)
        tiny = [{"start": g["segments"][0]["start"], "end": g["segments"][0]["start"] + 2.0}, {"start": g["segments"][-1]["end"] - 2.0, "end": g["segments"][-1]["end"]}]
        p = dict(g, segments=tiny)
        self.assertIsNone(validate(p, EPS)[0])
        out, errs, _ = validate(anchor(p, EPS), EPS, repair=True)
        self.assertIsNotNone(out, errs); self.assertTrue(20 <= out["runtime_s"] <= 60)
    def test_fake_quote_cannot_be_anchored(self):
        p = dict(base(3), hook="I will destroy your entire family."); self.assertIsNone(anchor(p, EPS))
    def test_hook_and_cliff_too_far_apart_not_anchored(self):
        p = base(1)   # a stitched promo: its hook and cliffhanger are far more than 60 s apart in the episode
        self.assertIsNone(anchor(p, EPS))

@unittest.skipUnless(HAVE, 'needs ./pack (Kuku scripts) and judgments/judgments.json')
class ProviderPlumbing(unittest.TestCase):
    """Exercises the real HTTP code paths against a local mock server (NOT the real APIs)."""
    @classmethod
    def setUpClass(cls):
        ans = {"promos": GOOD["3"]}
        class H(BaseHTTPRequestHandler):
            def do_POST(s):
                n = int(s.headers["Content-Length"]); body = json.loads(s.rfile.read(n))
                text = "```json\n" + json.dumps(ans) + "\n```"
                out = {"choices": [{"message": {"content": text}}]} if s.path.endswith("/chat/completions") else {"content": [{"type": "text", "text": text}]}
                b = json.dumps(out).encode(); s.send_response(200); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
            def log_message(*a): pass
        cls.srv = HTTPServer(("127.0.0.1", 0), H); cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()
    def test_openai_compatible(self):
        os.environ["OPENAI_API_KEY"] = "x"
        r = judge_episode(EPS[3], "openai", "any-model", base_url=f"http://127.0.0.1:{self.port}/v1")
        p = dict(r[0]); p["ep"] = 3; self.assertIsNotNone(validate(p, EPS)[0])
    def test_anthropic(self):
        os.environ["ANTHROPIC_API_KEY"] = "x"; os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{self.port}"
        r = judge_episode(EPS[3], "anthropic", "any-model")
        p = dict(r[0]); p["ep"] = 3; self.assertIsNotNone(validate(p, EPS)[0])

if __name__ == "__main__": unittest.main(verbosity=2)
