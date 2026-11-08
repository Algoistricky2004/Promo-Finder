"""End-to-end tests of the live-LLM code paths (run.py --judge openai) against a local mock server that imitates real failure modes.
NOT the real API. Needs ./pack and judgments/judgments.json (neither is in the public repo), otherwise skipped."""
import json, os, re, subprocess, sys, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.environ.get("PACK", os.path.join(ROOT, "pack"))
JP = os.path.join(ROOT, "judgments", "judgments.json")
HAVE = os.path.isdir(PACK) and os.path.exists(JP)
GOOD = json.load(open(JP)) if HAVE else {}

def mm_ss(x): return f"{int(x // 60)}:{x % 60:04.1f}"

class Mock:
    """behavior(user_prompt, body) -> (status, payload dict)"""
    def __init__(self, behavior):
        self.seen = []
        outer = self
        class H(BaseHTTPRequestHandler):
            def do_POST(s):
                body = json.loads(s.rfile.read(int(s.headers["Content-Length"]))); user = body["messages"][1]["content"]
                outer.seen.append((body, user)); status, payload = behavior(user, body)
                d = json.dumps(payload).encode(); s.send_response(status); s.send_header("Content-Length", str(len(d))); s.end_headers(); s.wfile.write(d)
            def log_message(*a): pass
        self.srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
    def close(self): self.srv.shutdown()

def chat(content): return 200, {"choices": [{"message": {"content": json.dumps(content)}}]}
def ep_of(user): return re.search(r"EPISODE (\d+)", user).group(1)

def run(mock, out, model="gpt-4o-mini", extra=()):
    env = {"OPENAI_API_KEY": "x", "PATH": os.environ.get("PATH", ""), "PROMO_RETRY_BASE": "0"}
    r = subprocess.run([sys.executable, "run.py", "--pack", PACK, "--out", out, "--judge", "openai", "--model", model, "--base-url", mock.url, *extra],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    p = os.path.join(ROOT, out, "promos.json")
    return r, (json.load(open(p)) if os.path.exists(p) else None)

@unittest.skipUnless(HAVE, "needs ./pack and judgments/judgments.json")
class LivePaths(unittest.TestCase):
    OUT = "/tmp/_live_test_out"

    def test_prompt_shows_plain_seconds(self):
        m = Mock(lambda u, b: chat({"promos": []}))
        run(m, self.OUT, extra=("--mode", "free", "--episodes", "3")); m.close()
        lines = [l for l in m.seen[0][1].split("\n") if l.startswith("[")]
        self.assertTrue(lines); self.assertFalse(any(":" in l.split("]")[0] for l in lines), "m:ss timestamps leaked into the prompt")

    def test_free_mode_sloppy_answers_repaired_and_fake_quote_rejected(self):
        def sloppy(user, body):
            n = ep_of(user); out = []
            for p in GOOD.get(n, []):
                q = json.loads(json.dumps(p))
                for s in q["segments"]: s["start"] = round(s["start"] + 1.7); s["end"] = mm_ss(s["end"] - 1.2)    # rounded, shifted, mm:ss string
                out.append(q)
            if n == "3": out.append(dict(out[0], hook="I will destroy your entire family."))                      # hallucinated quote
            return chat({"promos": out})
        m = Mock(sloppy); _, d = run(m, self.OUT, extra=("--mode", "free", "--episodes", "1-12", "--rerank", "0")); m.close()
        m2 = Mock(sloppy); _, d0 = run(m2, self.OUT, extra=("--mode", "free", "--episodes", "1-12", "--no-repair", "--rerank", "0")); m2.close()
        self.assertGreaterEqual(len(d["promos"]), 1); self.assertGreater(len(d["promos"]), len(d0["promos"]))
        self.assertFalse(any("destroy your entire family" in p["hook"] for p in d["promos"]))

    def test_free_mode_retry_recovers_short_mm_ss_answers(self):
        def beh(user, body):
            n = ep_of(user)
            if "REJECTED BY A CHECKER" in user: return chat({"promos": GOOD.get(n, [])})
            return chat({"promos": [dict(p, segments=[{"start": p["segments"][0]["start"] / 60 + 0.06, "end": p["segments"][0]["start"] + 9}]) for p in GOOD.get(n, [])]})
        m = Mock(beh); _, d = run(m, self.OUT, extra=("--mode", "free", "--episodes", "1-5", "--rerank", "0")); m.close()
        self.assertEqual(len(d["promos"]), sum(len(GOOD.get(str(i), [])) for i in range(1, 6)))

    def test_temperature_only_for_gpt3_gpt4(self):
        def beh(user, body):
            if body["model"].startswith("gpt-6") and "temperature" in body: return 400, {"error": {"message": "Unsupported parameter: temperature"}}
            return chat({"picks": []})
        for model, expect in (("gpt-4o-mini", True), ("gpt-6-luna", False)):
            m = Mock(beh); r, _ = run(m, self.OUT, model=model, extra=("--episodes", "3")); m.close()
            self.assertEqual(all("temperature" in b for b, _ in m.seen), expect, model)
            self.assertNotIn("FAILED", r.stdout, model)

    def test_select_mode_drops_bad_picks_and_all_clips_valid(self):
        def beh(user, body):
            return chat({"picks": [{"id": 1, "scores": {"hook": 4, "cliff": 3, "standalone": 4, "intensity": 4}, "needs_context": "none", "why": "ok", "flags": []},
                                   {"id": 99, "scores": {"hook": 5, "cliff": 5, "standalone": 5, "intensity": 5}},      # id out of range
                                   {"id": 2, "scores": {"hook": "x"}},                                               # malformed scores
                                   {"id": 3, "scores": {"hook": 9, "cliff": 0, "standalone": 3, "intensity": 3}, "needs_context": "weird"}]})
        m = Mock(beh); r, d = run(m, self.OUT, model="gpt-6-luna", extra=("--episodes", "1-10", "--rerank", "0")); m.close()
        self.assertEqual(len(m.seen), 10)
        self.assertTrue(d["promos"])
        self.assertTrue(all(r["errors"][0].startswith("duplicate") for r in d["rejected_by_validator"]))   # only duplicates may be dropped
        for i, p in enumerate(d["promos"]):                                                                   # and no two kept promos overlap > 30%
            for q in d["promos"][i + 1:]:
                ov = sum(max(0, min(s["end"], t["end"]) - max(s["start"], t["start"])) for s in p["segments"] for t in q["segments"] if s["ep"] == t["ep"])
                self.assertLessEqual(ov / min(p["runtime_s"], q["runtime_s"]), 0.3 + 1e-9)
        for p in d["promos"]:
            self.assertTrue(20 <= p["runtime_s"] <= 60); self.assertTrue(all(1 <= v <= 5 for v in p["scores"].values()))
            self.assertIn(p["needs_context"], ("none", "some", "heavy"))

    def test_one_failed_call_does_not_kill_the_run(self):
        calls = {"n": 0}
        def beh(user, body):
            calls["n"] += 1
            return (500, {"error": "boom"}) if ep_of(user) == "2" else chat({"picks": []})
        m = Mock(beh); r, d = run(m, self.OUT, extra=("--episodes", "1-3")); m.close()
        self.assertIn("JUDGE CALL FAILED", r.stdout); self.assertIn("ep 3:", r.stdout)   # the run continued to episode 3
        self.assertIsNotNone(d)

    # ---------------- editor (rerank) pass ----------------
    @staticmethod
    def _picks(user):
        return {"picks": [{"id": 1, "scores": {"hook": 5, "cliff": 5, "standalone": 5, "intensity": 5}, "needs_context": "none", "why": "ok", "flags": []},
                          {"id": 3, "scores": {"hook": 4, "cliff": 4, "standalone": 4, "intensity": 4}, "needs_context": "some", "why": "ok", "flags": []}]}

    def _editor_mock(self, rerank_reply):
        def beh(user, body):
            if "head editor" in body["messages"][0]["content"]: return rerank_reply(user)
            return chat(self._picks(user))
        return Mock(beh)

    def test_editor_pass_reorders_and_drops(self):
        def reply(user):
            n = user.count("=== CLIP "); ids = list(range(1, n + 1))
            return chat({"order": ids[::-1][:-2], "drop": [1, 2], "notes": {"3": "best moment", "1": "resolved ending", "2": "needs earlier episodes"}})
        m = self._editor_mock(reply); r, d = run(m, self.OUT, model="gpt-4o-mini", extra=("--episodes", "1-12")); m.close()
        self.assertIn("editor pass: ordered", r.stdout)
        dropped = [x for x in d["rejected_by_validator"] if x["errors"][0].startswith("dropped by editor pass")]
        self.assertEqual(len(dropped), 2)
        kept = {(p["segments"][0]["ep"], p["segments"][0]["start"]) for p in d["promos"]}
        self.assertTrue(all((x["promo"]["segments"][0]["ep"], x["promo"]["segments"][0]["start"]) not in kept for x in dropped))
        self.assertEqual([p["rank"] for p in d["promos"]], list(range(1, len(d["promos"]) + 1)))
        self.assertTrue(any(p.get("editor_note") for p in d["promos"]))

    def test_editor_pass_garbage_reply_falls_back_to_score_order(self):
        m = self._editor_mock(lambda u: (200, {"choices": [{"message": {"content": "sorry, I cannot do that"}}]})); r, d = run(m, self.OUT, extra=("--episodes", "1-12")); m.close()
        self.assertIn("fallback", r.stdout)
        comp = [p["composite"] for p in d["promos"]]; self.assertEqual(comp, sorted(comp, reverse=True))

    def test_editor_pass_cannot_drop_more_than_half(self):
        m = self._editor_mock(lambda u: chat({"order": [], "drop": list(range(1, u.count("=== CLIP ") + 1)), "notes": {}}))
        r, d = run(m, self.OUT, extra=("--episodes", "1-12")); m.close()
        dropped = [x for x in d["rejected_by_validator"] if x["errors"][0].startswith("dropped by editor pass")]
        n_total = len(d["promos"]) + len(dropped)
        self.assertLessEqual(len(dropped), n_total // 2 + 1); self.assertGreater(len(d["promos"]), 0)

    def test_editor_pass_failed_call_keeps_the_run(self):
        m = self._editor_mock(lambda u: (500, {"error": "boom"})); r, d = run(m, self.OUT, extra=("--episodes", "1-12")); m.close()
        self.assertIn("editor pass FAILED", r.stdout); self.assertTrue(d["promos"])

    def test_editor_pass_can_be_switched_off(self):
        m = self._editor_mock(lambda u: chat({"order": [], "drop": []})); r, d = run(m, self.OUT, extra=("--episodes", "1-12", "--rerank", "0")); m.close()
        self.assertFalse(any("head editor" in b["messages"][0]["content"] for b, _ in m.seen))

if __name__ == "__main__": unittest.main(verbosity=2)
