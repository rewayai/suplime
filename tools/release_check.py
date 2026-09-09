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
import subprocess
import sys
import wave
from pathlib import Path

# Before ANY import that pulls in huggingface_hub: constants.HF_HUB_OFFLINE is evaluated
# once, at import time, from the environment. Setting it after the import (as this script
# first did) left the --repo path online while claiming otherwise. The download itself runs
# in a child process with the flag off, so this process never needs the network.
os.environ["HF_HUB_OFFLINE"] = "1"

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


# 3.4 s of read speech from torchaudio's tutorial assets (VOiCES, CC BY 4.0). Small, stable
# and public, so CI can make a real "did any speech come back" assertion without a corpus.
SPEECH_SAMPLE_URL = ("https://download.pytorch.org/torchaudio/tutorial-assets/"
                     "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

EXPECTED = {"clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.72},
            "segmentation": {"min_duration_off": 0.0}}

# Both variants share the pipeline config, the frame count and the class count, so those
# checks alone cannot tell them apart: `upload_hf.py --repo ...-large --dir hf` would pass.
# Parameter count and the normalisation flag are what actually identify the model.
VARIANTS = {"base":  {"params": 114_323_207, "normalize": False},
            "large": {"params": 349_374_819, "normalize": True}}


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="Hugging Face repo id, e.g. rewayai/suplime")
    src.add_argument("--dir", type=Path, help="local staging directory instead of a download")
    ap.add_argument("--audio", type=Path, help="wav to run (default: a synthetic 12 s file)")
    ap.add_argument("--speech-sample", action="store_true",
                    help="download a small public speech clip and use it as --audio, so the "
                         "end-to-end assertion runs without a local corpus (what CI does)")
    ap.add_argument("--expect", choices=sorted(VARIANTS),
                    help="which model this repo must contain (default: inferred from the "
                         "repo id or directory name)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.repo:
        # download in a child process with the offline flag cleared, so THIS process stays
        # offline from its first import and the check below is a real guarantee
        code = ("import sys; from huggingface_hub import snapshot_download; "
                "print(snapshot_download(sys.argv[1], "
                "allow_patterns=['*.yaml', '*.bin', '*.md']))")
        env = {**os.environ, "HF_HUB_OFFLINE": "0"}
        root = Path(subprocess.run([sys.executable, "-c", code, a.repo], env=env, check=True,
                                   capture_output=True, text=True).stdout.strip())
        print(f"[check] downloaded {a.repo} -> {root}")
    else:
        root = a.dir.resolve()
        print(f"[check] using local {root}")

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
    normalize = bool(seg.hparams.get("normalize_waveform", False))
    print(f"[check] segmentation: {type(seg).__name__}, {n:,} params, {frames} frames / 10 s, "
          f"{classes} powerset classes, normalize_waveform={normalize}")

    name = a.expect or ("large" if "large" in (a.repo or str(a.dir)).lower() else "base")
    want = VARIANTS[name]
    if (n, normalize) != (want["params"], want["normalize"]):
        print(f"[check] FAIL: expected the {name} model ({want['params']:,} params, "
              f"normalize_waveform={want['normalize']}) but this repo holds {n:,} params, "
              f"normalize_waveform={normalize} — wrong weights published?")
        return 1
    print(f"[check] identity OK: this is the {name} model")

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

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    if a.speech_sample and not a.audio:
        import urllib.request

        a.audio = tmpdir / "suplime_speech_sample.wav"
        if not a.audio.exists():
            urllib.request.urlretrieve(SPEECH_SAMPLE_URL, a.audio)
        print(f"[check] speech sample: {a.audio}")
    audio = a.audio or synth_wav(tmpdir / "suplime_release_check.wav")
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
