#!/usr/bin/env python
"""Score a published suplime pipeline on one of our benchmark corpora.

    python tools/benchmark.py --model hf --corpus AMI_IHM --out results/AMI_IHM

`--model` is anything `Pipeline.from_pretrained` accepts: the local `hf/` staging
directory, a Hub repo id, or a config.yaml path. Scoring is pyannote.metrics
DiarizationErrorRate with collar 0 and overlapped speech scored, reference
cropped to the UEM — the benchmark convention behind every number in the model
card. Resumable: a uri whose hypothesis RTTM already exists is rescored, not
re-diarized, so a preempted job continues where it stopped.

Corpus paths are this cluster's layout (/mnt/shared/jose/diar_train_data); they
mirror diarizen_eval/run_corpus.sh so both tools score the same files.
"""
import argparse
import json
import os
import time
from pathlib import Path

D = Path(os.environ.get("DIAR_DATA", "/mnt/shared/jose/diar_train_data"))
CORPORA = {
    # name: (uri list, audio dir, audio suffix, rttm dir, uem dir)
    "AISHELL4": (D / "extended/lists/AISHELL4/test.txt", D / "extended/prepped/AISHELL4_test/audio", ".wav",
                 D / "extended/prepped/AISHELL4_test/rttm", D / "extended/uem/AISHELL4"),
    "AliMeeting_far": (D / "extended/prepped/AliMeeting_test_far/lists.txt", D / "extended/prepped/AliMeeting_test_far/audio", ".wav",
                       D / "extended/prepped/AliMeeting_test_far/rttm", D / "extended/prepped/AliMeeting_test_far/uem"),
    "AMI_IHM": (D / "diar_pkg/data/AMI/lists/test.txt", D / "diar_pkg/audio/AMI/audio", ".Mix-Headset.wav",
                D / "diar_pkg/data/AMI/rttm/test", D / "diar_pkg/data/AMI/uem/test"),
    "AMI_SDM": (D / "diar_pkg/data/AMI/lists/test.txt", D / "extended/prepped/AMI_SDM/audio", ".wav",
                D / "extended/prepped/AMI_SDM/rttm", D / "extended/uem/AMI_SDM"),
    "AVA_AVD": (D / "extended/lists/AVA_AVD/test.txt", D / "extended/prepped/AVA_AVD_test/audio", ".wav",
                D / "extended/prepped/AVA_AVD_test/rttm", D / "extended/uem/AVA_AVD"),
    "MSDWild": (D / "extended/lists/MSDWild/test.txt", D / "extended/prepped/MSDWild_val/audio", ".wav",
                D / "extended/prepped/MSDWild_val/rttm", D / "extended/uem/MSDWild"),
    "RAMC": (D / "extended/lists/RAMC/test.txt", D / "extended/prepped/RAMC_test/audio", ".wav",
             D / "extended/prepped/RAMC_test/rttm", D / "extended/uem/RAMC"),
    "VoxConverse": (D / "diar_pkg/data/VoxConverse/lists/test.txt", D / "diar_pkg/audio/VoxConverse/audio_ref", ".wav",
                    D / "diar_pkg/data/VoxConverse/rttm/test", D / "diar_pkg/data/VoxConverse/uem/test"),
    "NOTSOFAR1": (D / "diar_pkg/data/NOTSOFAR1/lists/test.txt", D / "diar_pkg/audio/NOTSOFAR1/audio_ref", ".wav",
                  D / "diar_pkg/data/NOTSOFAR1/rttm/test", D / "diar_pkg/data/NOTSOFAR1/uem/test"),
    "CHiME6": (D / "diar_pkg/data/CHiME6/lists/test.txt", D / "diar_pkg/audio/CHiME6/audio_ref", ".wav",
               D / "diar_pkg/data/CHiME6/rttm/test", D / "diar_pkg/data/CHiME6/uem/test"),
    "DiPCo": (D / "diar_pkg/data/DiPCo/lists/test.txt", D / "diar_pkg/audio/DiPCo/audio_ref", ".wav",
              D / "diar_pkg/data/DiPCo/rttm/test", D / "diar_pkg/data/DiPCo/uem/test"),
    "ICSI": (D / "diar_pkg/data/ICSI/lists/test.txt", D / "diar_pkg/audio/ICSI/audio", ".interaction.wav",
             D / "diar_pkg/data/ICSI/rttm/test", D / "diar_pkg/data/ICSI/uem/test"),
}


def load_rttm(path: Path):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=path.stem)
    for i, ln in enumerate(path.read_text(errors="replace").splitlines()):
        p = ln.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            # one track per line: two speakers sharing an identical segment must not
            # overwrite each other (they would with the default single track)
            ann[Segment(s, s + d), i] = p[7]
    return ann


def load_uem(path: Path):
    from pyannote.core import Segment, Timeline

    tl = Timeline(uri=path.stem)
    for ln in path.read_text(errors="replace").splitlines():
        p = ln.split()
        if len(p) >= 4:
            tl.add(Segment(float(p[2]), float(p[3])))
    return tl


def write_rttm(ann, uri: str, path: Path):
    tmp = path.with_suffix(".rttm.tmp")
    with open(tmp, "w") as f:
        for seg, _, lab in ann.itertracks(yield_label=True):
            f.write(f"SPEAKER {uri} 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {lab} <NA> <NA>\n")
    tmp.rename(path)


def build_pipeline(model: str, threshold, device: str):
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model)
    if pipeline is None:
        raise SystemExit(f"could not load pipeline {model!r}")
    if threshold is not None:
        params = pipeline.parameters(instantiated=True)
        params["clustering"]["threshold"] = threshold
        pipeline.instantiate(params)
    return pipeline.to(torch.device(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hf/ dir, Hub repo id or config.yaml")
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None, help="override clustering.threshold")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from pyannote.metrics.diarization import DiarizationErrorRate

    list_path, audio_dir, suffix, rttm_dir, uem_dir = CORPORA[a.corpus]
    uris = [u for u in list_path.read_text().split() if u]
    if a.limit:
        uris = uris[: a.limit]
    hyp_dir = a.out / "hyp"
    # Hypotheses are reused by filename alone, so without this a re-run with a different
    # --model or --threshold silently rescored the OLD hypotheses and wrote the NEW
    # parameters into the JSON — a threshold sweep in which nothing was re-diarized.
    stamp_path = a.out / "run_params.json"
    stamp = {"model": str(a.model), "threshold": a.threshold}
    if stamp_path.exists():
        previous = json.loads(stamp_path.read_text())
        if previous != stamp and any(hyp_dir.glob("*.rttm")):
            raise SystemExit(
                f"{hyp_dir} holds hypotheses produced with {previous}, but this run asks for "
                f"{stamp}. Re-running here would reuse the old RTTMs and report the new "
                f"parameters. Use a different --out, or delete {hyp_dir}."
            )
    a.out.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    hyp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = None
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    per_file = {}
    t_all = time.time()
    for i, uri in enumerate(uris, 1):
        ref = load_rttm(rttm_dir / f"{uri}.rttm")
        uem = load_uem(uem_dir / f"{uri}.uem")
        hyp_path = hyp_dir / f"{uri}.rttm"
        if hyp_path.exists():
            hyp = load_rttm(hyp_path)
        else:
            if pipeline is None:
                pipeline = build_pipeline(a.model, a.threshold, a.device)
                print(f"[{a.corpus}] pipeline {a.model}: {pipeline.parameters(instantiated=True)}", flush=True)
            t0 = time.time()
            out = pipeline(str(audio_dir / f"{uri}{suffix}"))
            hyp = getattr(out, "speaker_diarization", out)
            hyp.uri = uri
            write_rttm(hyp, uri, hyp_path)
            print(f"[{a.corpus}] {i}/{len(uris)} {uri}: inference {time.time() - t0:.0f}s", flush=True)
        d = metric(ref, hyp, uem=uem, detailed=True)
        per_file[uri] = {
            "der": d["diarization error rate"],
            "miss": d["missed detection"] / max(d["total"], 1e-9),
            "fa": d["false alarm"] / max(d["total"], 1e-9),
            "conf": d["confusion"] / max(d["total"], 1e-9),
        }
        print(f"[{a.corpus}] {uri}: DER={100 * per_file[uri]['der']:.2f}", flush=True)

    total = abs(metric)
    comp = metric[:]
    summary = {
        "corpus": a.corpus,
        "model": a.model,
        "threshold": a.threshold,
        "n": len(uris),
        "der": total,
        "components": {k: comp[k] / max(comp["total"], 1e-9)
                       for k in ("missed detection", "false alarm", "confusion")},
        "per_file": per_file,
        "elapsed_sec": time.time() - t_all,
    }
    (a.out / f"{a.corpus}.json").write_text(json.dumps(summary, indent=2))
    c = summary["components"]
    print(f"== {a.corpus}: DER={100 * total:.2f}% (n={len(uris)}) miss={100 * c['missed detection']:.1f} "
          f"fa={100 * c['false alarm']:.1f} conf={100 * c['confusion']:.1f}", flush=True)


if __name__ == "__main__":
    main()
