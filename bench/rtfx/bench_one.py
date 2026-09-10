#!/usr/bin/env python
"""Time ONE diarization system on a set of clips and print a JSON result.

Run by rtfx.py, once per system, in whichever virtualenv that system needs (SUPlime and
community-1 want pyannote.audio 4.x; DiariZen vendors its own 3.1.1 fork, so the two cannot
share an environment). Nothing here is suplime-specific beyond the repo ids.

    python bench_one.py --system suplime --clips a.wav b.wav --repeats 3

Every system is timed exactly as its own documentation tells users to run it — default
batch sizes, default parameters, model already loaded and warmed up. What is timed is the
one call that turns a file into speaker turns, nothing else.
"""
import argparse
import json
import os
import statistics
import sys
import time
import wave


def clip_seconds(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def load(system: str, device: str):
    """Return (callable(path) -> annotation, load_seconds, description)."""
    import torch

    t0 = time.time()
    if system in ("suplime", "suplime-large"):
        from pyannote.audio import Pipeline

        repo = "rewayai/suplime" if system == "suplime" else "rewayai/suplime-large"
        pipeline = Pipeline.from_pretrained(repo).to(torch.device(device))
        run = lambda p: pipeline(p).speaker_diarization
        desc = repo

    elif system == "community-1":
        from pyannote.audio import Pipeline

        repo = "pyannote/speaker-diarization-community-1"
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit(f"{repo} is gated: set HF_TOKEN to a token that has accepted "
                             f"its conditions, or skip this system")
        pipeline = Pipeline.from_pretrained(repo, token=token).to(torch.device(device))
        # 4.x returns a container; 3.x-style pipelines return the Annotation itself
        run = lambda p: getattr(pipeline(p), "speaker_diarization", None) or pipeline(p)
        desc = repo

    elif system == "diarizen":
        from diarizen.pipelines.inference import DiariZenPipeline

        repo = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
        pipeline = DiariZenPipeline.from_pretrained(repo)
        run = lambda p: pipeline(p, sess_name="bench")
        desc = repo

    else:
        raise SystemExit(f"unknown system {system!r}")

    return run, time.time() - t0, desc


# pyannote fires a hook after each stage: segmentation, speaker_counting, embeddings
# (once per batch, then once at the end) and discrete_diarization. Timing the gaps between
# them says which stage a pipeline actually spends its time in — the question "why is one
# system 2x faster" is not answerable from a single total.
STAGES = [("segmentation", "segmentation"), ("speaker_counting", "speaker counting"),
          ("embeddings", "embedding extraction"), ("discrete_diarization", "clustering")]


def profile(system: str, run, clip: str, sync) -> dict:
    """Per-stage seconds for one run of the longest clip. pyannote pipelines only."""
    if system == "diarizen":
        return {"unsupported": "DiariZenPipeline.__call__ takes no hook argument"}

    import inspect

    events = []

    def hook(name, artifact=None, file=None, total=None, completed=None):
        sync()
        events.append((name, time.time()))

    # run() closes over the pipeline; call it again with the hook if the signature allows
    try:
        sync()
        t0 = time.time()
        pipeline = run.__closure__[0].cell_contents
        if "hook" not in inspect.signature(pipeline.apply).parameters:
            return {"unsupported": "pipeline.apply takes no hook argument"}
        pipeline(clip, hook=hook)
        sync()
        total_s = time.time() - t0
    except Exception as e:  # profiling is a nicety, never fail the benchmark for it
        return {"unsupported": f"{type(e).__name__}: {e}"}

    marks, seen = {}, t0
    for key, label in STAGES:
        hits = [t for n, t in events if n == key]
        if not hits:
            continue
        marks[label] = round(hits[-1] - seen, 2)
        seen = hits[-1]
    marks["output"] = round(total_s - (seen - t0), 2)
    marks["total"] = round(total_s, 2)
    marks["percent"] = {k: round(100 * v / total_s) for k, v in marks.items() if k != "total"}
    return {"clip": os.path.basename(clip), **marks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["suplime", "suplime-large", "community-1", "diarizen"])
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile", action="store_true",
                    help="also report where the time goes inside the pipeline (pyannote "
                         "systems only; adds a synchronise per stage, so run it separately "
                         "from the timing runs)")
    a = ap.parse_args()

    import torch

    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    run, load_s, desc = load(a.system, a.device)

    # warm up on the shortest clip: the first call pays for CUDA context, kernel autotuning
    # and any lazy weight materialisation, none of which belongs in a throughput number
    warm = min(a.clips, key=clip_seconds)
    run(warm)
    sync()

    out = {"system": a.system, "model": desc, "device": a.device, "load_sec": round(load_s, 1),
           "torch": torch.__version__, "clips": []}
    if a.device.startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)

    if a.profile:
        out["profile"] = profile(a.system, run, max(a.clips, key=clip_seconds), sync)

    for clip in a.clips:
        dur = clip_seconds(clip)
        times, speakers = [], None
        for _ in range(a.repeats):
            sync()
            t0 = time.time()
            ann = run(clip)
            sync()
            times.append(time.time() - t0)
            speakers = len(ann.labels())
        med = statistics.median(times)
        out["clips"].append({
            "clip": os.path.basename(clip), "audio_sec": round(dur, 1),
            "times_sec": [round(t, 2) for t in times], "median_sec": round(med, 2),
            "rtfx": round(dur / med, 1), "rtf": round(med / dur, 5), "speakers": speakers,
        })
        print(f"[{a.system}] {os.path.basename(clip)}: {dur/60:.1f} min audio in "
              f"{med:.1f} s -> {dur/med:.0f}x real time ({speakers} speakers)",
              file=sys.stderr, flush=True)

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
