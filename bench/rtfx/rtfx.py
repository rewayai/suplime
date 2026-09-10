#!/usr/bin/env python
"""RTFx benchmark: how many times faster than real time each diarization system runs.

    python rtfx.py                              # all four systems, all clips
    python rtfx.py --systems suplime suplime-large
    python rtfx.py --repeats 5 --out results.json

RTFx = seconds of audio / seconds of wall clock. Higher is faster; 50x means an hour of
audio in 72 seconds. The reciprocal, RTF, is also reported.

Each system runs in its own virtualenv because they cannot share one: SUPlime and
community-1 need pyannote.audio 4.x, DiariZen vendors a 3.1.1 fork with numpy < 2. This
script only dispatches; bench_one.py does the timing.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# in the image; falls back to the current interpreter when run outside it
VENVS = {"suplime": "/opt/venv/suplime/bin/python", "diarizen": "/opt/venv/diarizen/bin/python"}
SYSTEM_VENV = {"suplime": "suplime", "suplime-large": "suplime",
               "community-1": "suplime", "diarizen": "diarizen"}
ALL = ["suplime", "suplime-large", "diarizen", "community-1"]


def interpreter(system: str) -> str:
    path = VENVS[SYSTEM_VENV[system]]
    return path if Path(path).exists() else sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--clips-dir", type=Path, default=Path(os.environ.get("RTFX_CLIPS", HERE / "clips")))
    ap.add_argument("--clips", nargs="*", help="explicit wav paths (default: every wav in --clips-dir)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("rtfx_results.json"))
    a = ap.parse_args()

    clips = [str(p) for p in (map(Path, a.clips) if a.clips else sorted(a.clips_dir.glob("*.wav")))]
    if not clips:
        raise SystemExit(f"no clips found in {a.clips_dir} (see prepare_clips.py)")
    print(f"clips: {', '.join(Path(c).name for c in clips)}\n", flush=True)

    results = []
    for system in a.systems:
        if system == "community-1" and not os.environ.get("HF_TOKEN"):
            print(f"-- {system}: SKIPPED, gated model and no HF_TOKEN in the environment\n", flush=True)
            continue
        py = interpreter(system)
        print(f"-- {system}  ({py})", flush=True)
        proc = subprocess.run(
            [py, str(HERE / "bench_one.py"), "--system", system, "--repeats", str(a.repeats),
             "--device", a.device, "--clips", *clips],
            capture_output=True, text=True)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"-- {system}: FAILED (exit {proc.returncode})\n", flush=True)
            results.append({"system": system, "error": proc.stderr.strip().splitlines()[-1:] or ["failed"]})
            continue
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(flush=True)

    ok = [r for r in results if "clips" in r]
    if ok:
        gpu = ok[0].get("gpu", a.device)
        names = [c["clip"] for c in ok[0]["clips"]]
        w = max(len(r["system"]) for r in ok) + 2
        print(f"\nRTFx on {gpu} — higher is faster (x real time)\n")
        print(f"{'system':{w}}" + "".join(f"{n:>14}" for n in names) + f"{'load s':>9}")
        for r in ok:
            row = {c["clip"]: c["rtfx"] for c in r["clips"]}
            cells = "".join(f"{row[n]:>14.1f}" if n in row and row[n] < 10
                            else f"{row.get(n, float('nan')):>14.0f}" for n in names)
            print(f"{r['system']:{w}}" + cells + f"{r['load_sec']:>9.0f}")
        print(f"\n(each figure is the median of {a.repeats} runs, after a warm-up run; "
              f"model load excluded)")

    a.out.write_text(json.dumps({"clips": [Path(c).name for c in clips], "repeats": a.repeats,
                                 "results": results}, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
