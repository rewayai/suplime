#!/usr/bin/env python
"""Release gate: prove a *published* SUPlime repo works from a clean install.

    python tools/release_check.py --repo rewayai/suplime          # download from the Hub
    python tools/release_check.py --repo rewayai/suplime-large
    python tools/release_check.py --dir hf                        # or check a local staging dir
    python tools/release_check.py --repo rewayai/suplime --audio meeting.wav

The unit tests deliberately skip everything that needs weights (hf/ and
reproducible_research/ are git-ignored, so a normal clone has no checkpoints), which means
`pytest` passing says nothing about the artifact people actually download. This script
closes that gap: it fetches the repo, loads the pipeline exactly as the model card tells
users to, runs it on one short file, and checks the published hyper-parameters survived the
round trip. Run it after every upload, before announcing a release.

Exit code 0 = the artifact is loadable and runs; non-zero = do not announce.

`--audio` is what turns this from a smoke test into a real check: the built-in synthetic
signal is not speech, so the VAD correctly finds nothing in it. Point it at a short
recording (~1 min); a full meeting on CPU takes tens of minutes, and `--device cuda` is
three orders of magnitude more comfortable.
"""
import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def synth_wav(path: Path, seconds: float = 12.0, sr: int = 16000) -> Path:
    """A synthetic signal to drive the pipeline end to end without a corpus. It is NOT speech,
    so the VAD is expected to find nothing in it — that is why the speech assertion below only
    applies to a real --audio file, while the synthetic path checks the model's raw output."""
    rng = np.random.RandomState(0)
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for i, start in enumerate(np.arange(0, seconds, 1.5)):
        seg = (t >= start) & (t < start + 1.2)
        carrier = np.sin(2 * np.pi * (140 if i % 2 else 210) * t[seg])
        x[seg] = carrier * (1 + 0.5 * rng.randn(seg.sum())) * 0.3
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype(np.int16).tobytes())
    return path


EXPECTED = {"clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.72},
            "segmentation": {"min_duration_off": 0.0}}


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="Hugging Face repo id, e.g. rewayai/suplime")
    src.add_argument("--dir", type=Path, help="local staging directory instead of a download")
    ap.add_argument("--audio", type=Path, help="wav to run (default: a synthetic 12 s file)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.repo:
        from huggingface_hub import snapshot_download

        # allow_patterns keeps reproducible_research/ (thousands of RTTMs) out of the download
        root = Path(snapshot_download(a.repo, allow_patterns=["*.yaml", "*.bin", "*.md"]))
        print(f"[check] downloaded {a.repo} -> {root}")
    else:
        root = a.dir.resolve()
        print(f"[check] using local {root}")

    # from here on nothing may touch the network: a release that needs it is broken
    os.environ["HF_HUB_OFFLINE"] = "1"
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(root / "config.yaml").to(torch.device(a.device))
    print(f"[check] loaded pipeline on {a.device}: {type(pipeline).__name__}")

    params = pipeline.parameters(instantiated=True)
    if params != EXPECTED:
        print(f"[check] FAIL: published hyper-parameters changed\n  got      {params}\n  expected {EXPECTED}")
        return 1
    print(f"[check] hyper-parameters match the published values: {params}")

    seg = pipeline._segmentation.model
    n = sum(p.numel() for p in seg.parameters())
    frames, classes = seg.num_frames(160000), seg.specifications.num_powerset_classes
    print(f"[check] segmentation: {type(seg).__name__}, {n:,} params, {frames} frames / 10 s, "
          f"{classes} powerset classes, normalize_waveform={seg.hparams.get('normalize_waveform', False)}")

    # deterministic, corpus-free: the head must emit a valid powerset distribution.
    # The activation is LogSoftmax, so the probabilities are exp(scores).
    with torch.inference_mode():
        scores = seg(torch.zeros(1, 1, 160000, device=torch.device(a.device)))
    ok = scores.shape == (1, frames, classes) and torch.allclose(
        scores.exp().sum(-1).cpu(), torch.ones(1, frames), atol=1e-3)
    if not ok:
        print(f"[check] FAIL: segmentation output {tuple(scores.shape)} is not a powerset "
              f"log-distribution over {classes} classes")
        return 1
    print(f"[check] segmentation output OK: {tuple(scores.shape)}, exp(rows) sum to 1")

    audio = a.audio or synth_wav(Path(os.environ.get("TMPDIR", "/tmp")) / "suplime_release_check.wav")
    out = pipeline(str(audio))
    ann = out.speaker_diarization
    turns = list(ann.itertracks(yield_label=True))
    print(f"[check] {audio.name}: {len(turns)} turns, {len(ann.labels())} speaker(s), "
          f"{ann.get_timeline().duration():.1f} s of speech")
    if a.audio and not turns:
        print("[check] FAIL: no speech found in a real recording")
        return 1
    if not a.audio:
        print("[check] note: the synthetic file is not speech, so 0 turns is expected here; "
              "pass --audio <real recording> for an end-to-end speech assertion")
    print("[check] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
