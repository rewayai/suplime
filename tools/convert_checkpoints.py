#!/usr/bin/env python
"""Convert the internal pyannote-audio-fork checkpoints into the Hugging Face layout.

    python tools/convert_checkpoints.py \
        --segmentation ~/models/conf_aug_ext_avg5.ckpt \
        --embedding   <fork>/reway/speaker-diarization-vblinkf/embedding/pytorch_model.bin \
        --out hf

Segmentation: the fork stored `wav2vec: "WAVLM_BASE_PLUS"` (a torchaudio bundle
name, which re-downloads 377 MB on every load) plus LSTM/Mamba hparams that the
Conformer checkpoint never uses. We rewrite `hyper_parameters` to the exact
signature of `suplime.models.segmentation.SuplimeSegmentation` with the bundle's
config dict inlined, and re-stamp the architecture pointer and versions.

Embedding: only the architecture pointer and the pyannote.audio version stamp
change (the fork's build stamped 4.0.4, which made upstream's dependency check
fail against the fork's 0.1.devNNNN version string).

Run under the *suplime* environment (plain pyannote.audio, no fork): loading the
originals needs nothing fork-specific, and the strict reload at the end proves
the published files work without it.

The version stamped into each checkpoint is the *current* suplime version, which is
therefore the minimum a consumer needs. Re-converting an unchanged model only to raise
that floor would invalidate published checksums for no gain, so hf/ keeps the 0.1.0 stamp
it was converted with while hf-large/ carries 0.2.0 (the release that added the
normalising wrapper its weights require).

SECURITY: torch.load(..., weights_only=False) below unpickles the checkpoint, which can
execute arbitrary code. Only ever run this on checkpoints you produced or otherwise trust.
"""
import argparse
import hashlib
from pathlib import Path

import torch
import torchaudio

from suplime import __version__ as SUPLIME_VERSION
from suplime.models import SuplimeSegmentation, WeSpeakerSimAMResNet34


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pyannote_version() -> str:
    from pyannote.audio import __version__

    return __version__


def convert_segmentation(src: Path, dst: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    arch = ckpt["pyannote.audio"]["architecture"]
    print(f"[seg] source architecture {arch['module']}.{arch['class']}, hparams keys {sorted(hp)}")

    wav2vec = hp["wav2vec"]
    normalize = bool(hp.get("normalize_waveform", False))
    if isinstance(wav2vec, str):
        bundle = getattr(torchaudio.pipelines, wav2vec)
        if not hasattr(bundle, "_params"):
            raise SystemExit(f"torchaudio.pipelines.{wav2vec} has no _params; this torchaudio "
                             f"({torchaudio.__version__}) is not one suplime was tested against "
                             f"(see requirements-tested.txt)")
        # the *_LARGE bundles wrap the backbone in a waveform-normalising module, so their
        # checkpoints carry `wav2vec.model.*`; suplime rebuilds that wrapper from this flag
        normalize = bool(getattr(bundle, "_normalize_waveform", False))
        wav2vec = dict(bundle._params)
        print(f"[seg] inlined torchaudio.pipelines.{hp['wav2vec']} config ({len(wav2vec)} keys), "
              f"normalize_waveform={normalize}")
    assert hp.get("conformer") is not None, "only Conformer-head checkpoints are supported"
    assert hp.get("mamba") is None

    ckpt["hyper_parameters"] = {
        "wav2vec": wav2vec,
        "wav2vec_layer": hp["wav2vec_layer"],
        "conformer": dict(hp["conformer"]),
        "linear": dict(hp["linear"]),
        "normalize_waveform": normalize,
        "sample_rate": hp.get("sample_rate", 16000),
        "num_channels": hp.get("num_channels", 1),
    }
    ckpt["pyannote.audio"]["architecture"] = {
        "module": "suplime.models.segmentation",
        "class": "SuplimeSegmentation",
    }
    ckpt["pyannote.audio"]["versions"] = {
        "pyannote.audio": pyannote_version(),
        "suplime": SUPLIME_VERSION,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)

    model = SuplimeSegmentation.load_from_checkpoint(dst, map_location="cpu", strict=True, weights_only=False)
    n = sum(p.numel() for p in model.parameters())
    frames = model.num_frames(int(10 * model.hparams.sample_rate))
    assert model.specifications.powerset, "expected a powerset model"
    print(f"[seg] strict reload OK: {n:,} params, {frames} frames / 10 s, "
          f"{model.specifications.num_powerset_classes} powerset classes -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")


def convert_embedding(src: Path, dst: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    arch = ckpt["pyannote.audio"]["architecture"]
    print(f"[emb] source architecture {arch['module']}.{arch['class']}, "
          f"versions {ckpt['pyannote.audio']['versions']}")
    ckpt["pyannote.audio"]["architecture"] = {
        "module": "suplime.models.embedding",
        "class": "WeSpeakerSimAMResNet34",
    }
    ckpt["pyannote.audio"]["versions"] = {
        "pyannote.audio": pyannote_version(),
        "suplime": SUPLIME_VERSION,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)

    model = WeSpeakerSimAMResNet34.load_from_checkpoint(dst, map_location="cpu", strict=True, weights_only=False)
    n = sum(p.numel() for p in model.parameters())
    print(f"[emb] strict reload OK: {n:,} params, dimension {model.dimension} -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation", type=Path, required=True)
    ap.add_argument("--embedding", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("hf"))
    a = ap.parse_args()

    print(f"[src] segmentation sha256 {sha256(a.segmentation)}  {a.segmentation}")
    print(f"[src] embedding    sha256 {sha256(a.embedding)}  {a.embedding}")
    convert_segmentation(a.segmentation, a.out / "segmentation" / "pytorch_model.bin")
    convert_embedding(a.embedding, a.out / "embedding" / "pytorch_model.bin")


if __name__ == "__main__":
    main()
