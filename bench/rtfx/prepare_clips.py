#!/usr/bin/env python
"""Cut the benchmark clips from a long meeting recording.

    python prepare_clips.py --source ES2004b.Mix-Headset.wav --out clips/

Three lengths, because RTFx is not one number: a short file is dominated by fixed costs
(padding the last window, building the clustering) while a long one shows steady-state
throughput. 1 / 5 / 30 minutes covers both ends of what people actually run.

The default source is AMI (ES2004b, Mix-Headset), which is CC BY 4.0 and so can be shipped
inside the image; any 16 kHz mono wav works. Clips are cut from 60 s in, past the "is
everyone here" preamble, so all three contain real multi-speaker conversation.
"""
import argparse
import wave
from pathlib import Path

LENGTHS = {"clip_01min.wav": 60, "clip_05min.wav": 300, "clip_30min.wav": 1800}
SKIP = 60  # seconds of preamble to drop


def cut(src: Path, dst: Path, seconds: int, offset: int) -> float:
    with wave.open(str(src)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if sr != 16000 or ch != 1:
            raise SystemExit(f"{src}: expected 16 kHz mono, got {sr} Hz / {ch} ch")
        available = w.getnframes() / sr - offset
        if available < seconds:
            raise SystemExit(f"{src}: only {available/60:.1f} min after the {offset}s offset, "
                             f"need {seconds/60:.0f} min for {dst.name}")
        w.setpos(offset * sr)
        frames = w.readframes(seconds * sr)
    with wave.open(str(dst), "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(width)
        o.setframerate(sr)
        o.writeframes(frames)
    return dst.stat().st_size / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="16 kHz mono wav, >= 31 min")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "clips")
    ap.add_argument("--offset", type=int, default=SKIP)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    for name, seconds in LENGTHS.items():
        mb = cut(a.source, a.out / name, seconds, a.offset)
        print(f"{name}: {seconds/60:.0f} min, {mb:.1f} MB")
    (a.out / "SOURCE.txt").write_text(
        f"Cut from {a.source.name} at offset {a.offset}s by prepare_clips.py.\n"
        "Default source: AMI Meeting Corpus (https://groups.inf.ed.ac.uk/ami/corpus/),\n"
        "licensed CC BY 4.0 — attribute the AMI Consortium if you redistribute these clips.\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
