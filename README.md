# Promo Finder (Kuku AI Generalist assignment, Problem 1)

Reads microdrama screenplays (timestamped `.txt`) and finds 20-60 s promo clips: a hook in the first 3 s, a cliffhanger at the cut, understandable on its own.
Per promo it outputs: episode, start/end of every segment, hook, cliffhanger, one-line why, 1-5 scores, rank.

**Run in Colab:** https://colab.research.google.com/github/Algoistricky2004/Promo-Finder/blob/main/Kuku_Promo_Finder_Colab.ipynb (clones this repo; you upload only the scripts zip).

**Confidentiality.** Kuku's scripts, the judgments that quote them and the raw outputs are *not* in this repo. The repo is code only. Put the scripts in `./pack/` (any folder layout of `*.txt`).
The saved judge output and all raw outputs from the 50-episode pack are in the separate raw-outputs folder shared in the submission form.

## Run on Colab (live OpenAI judge, CPU runtime is enough)
Open `Kuku_Promo_Finder_Colab.ipynb`. Cell 1 clones this repo; cell 1b asks for Kuku's scripts zip. Then: API key -> settings (model, mode) -> test on 3 episodes -> full run -> results -> download.

## Run locally (Python 3.10+, no third-party packages)
```bash
python run.py --pack ./pack --out out --judge heuristic        # free, offline, ~2 s: shortlist only (heuristic_candidates.csv)
export OPENAI_API_KEY=...                                      # any OpenAI-compatible endpoint
python run.py --pack ./pack --out out --judge openai --model <model-id> [--base-url <url>] [--mode select|free]
export ANTHROPIC_API_KEY=...
python run.py --pack ./pack --out out --judge anthropic --model <model-id>
python run.py --pack ./pack --out out --judge cache --cache judgments.json   # replay saved judgments (from the raw-outputs folder)
python run.py ... --audience warm     # no penalty for context-heavy promos (retargeting viewers of earlier episodes)
python run.py ... --episodes 1-10     # subset
python run.py ... --rerank 40         # editor pass on the top 40 (default for live runs; 0 = off; never runs for --judge cache)
python run.py ... --max-overlap 0.3   # drop a lower-scored promo overlapping a higher-scored one by > 30% of the shorter clip (default)
python analysis/heuristic_vs_judge.py ./pack out         # quality check vs. the heuristic
python analysis/compare_runs.py outA/promos.json outB/promos.json   # agreement between two runs / two models
python analysis/live_vs_reference.py live/promos.json ref/promos.json   # agreement, abstention, score inflation, context handling
PACK=./pack python -m unittest discover -s tests -v      # 29 tests (need the pack + saved judgments; skipped otherwise)
```
The model is a flag (`--model`, `--base-url`, env `PROMO_MODEL`); nothing else changes. `temperature` is sent only to gpt-3/gpt-4 models (GPT-6 / reasoning models reject it).

## Pipeline
1. `parse.py` - episodes -> scenes -> events (CAMERA / ACTION / SOUND / DIALOGUE, in seconds). Checked: parsed event count == raw `Start:` count in all 50 files.
2. `heuristic.py` - every 20-60 s window aligned to events (never cutting a spoken line), scored on surface signals; top-k non-overlapping windows per episode.
3. Judge (LLM, one call per episode, `judge.py`), two modes:
   - `--mode select` (default for live models): the model picks among the code-built windows and scores them. Timestamps and runtime are valid by construction. Cannot stitch segments.
   - `--mode free`: the model proposes segments itself (can stitch). Needs a strong model; output is repaired (`--repair`), re-anchored from its own quotes (`anchor()`), and gets one self-correction retry.
4. `validate.py` - never trusts the model: snaps to real event boundaries, rejects mid-line cuts, runtime outside 20-60 s, hook text not in the first 3 s, cliffhanger text not in the last 6 s.
5. `rerank.py` - editor pass (live runs): one extra call shows the top N promos (shuffled) and asks the model to order them against each other and drop clips it would not post (resolved endings, weak hooks, needs earlier episodes, duplicates). It only returns ids; invalid, partial or failed answers fall back to the score order; at most half the list can be dropped.
6. `report.py` - rank = 0.30 hook + 0.30 cliffhanger + 0.25 stand-alone + 0.15 intensity (judge scores), minus 0.04 / 0.10 if some / heavy context is needed (cold audience).
`promos.json -> rejected_by_validator` lists first-pass rejections, including ones later recovered by repair or retry (those promos also appear in `promos`).

## What was measured with a live model (OpenAI chat model; reference = a ranking made by Claude reading all 50 scripts)
- **`--mode free`, first try (episodes 1-3):** every timestamp and quote was real, but all 6 promos were 3-4 fragments of ~2 s (7-10 s total) and failed the 20-60 s rule; one retry did not change that. `anchor()` rebuilt 2 of 6 as one clip from the model's own quotes; for the other 4 the hook and cliffhanger were more than 60 s apart. Hence `--mode select` is the default.
- **`--mode select`, full 50 episodes, no editor pass:** 87 promos in 50/50 episodes, all strictly valid; 15/34 reference promos matched at IoU >= 0.5; 1 of the top-10 episodes shared; 27 promos in the 16 episodes where the reference found none (5 in its top 20); cliffhanger = 5 for 64/87.
- **`--mode select` + editor pass (final):** 90 promos in 50/50 episodes, all strictly valid (21-44.7 s), 20 near-duplicates dropped. 16/34 reference promos matched (23/34 at >= 0.3); 3 of the top-10 episodes shared; the finished comeback speech fell from #3 to #9. But the editor pass dropped 0 of 40 clips, 27 promos are still in the 16 no-promo episodes (7 in the top 20), cliffhanger = 5 for 63/90, and 0 of 29 late-season promos were flagged as needing heavy context. So the ranking is better but not trustworthy: the output is a valid shortlist for an editor.
- **Next, not built:** force abstention with a threshold, compare clips with a stronger model, log editor accept/reject to calibrate the weights.

## Known limits
- Text only: no delivery, performance or visuals beyond ACTION lines. Stitched promos (free mode) need an editor preview for continuity.
- No performance data (retention, CTR) behind the scores; the saved ground truth is one reader's judgment.
- Select mode can only choose windows the heuristic proposes (top-6 windows cover >=50% of all 34 saved promos, 32/34 with top-4) and cannot stitch.
- The hook lexicon in `heuristic.py` is generic English melodrama vocabulary; at scale it should be a per-genre / per-language config. Generated scripts can drift on names; the tool never matches on names, but check on-screen text.
