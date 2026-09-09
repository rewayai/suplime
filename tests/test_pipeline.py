import os
from pathlib import Path

import pytest

from suplime import SuplimeDiarization

HF = Path(__file__).resolve().parents[1] / "hf"
have_weights = (HF / "segmentation" / "pytorch_model.bin").exists() and (HF / "embedding" / "pytorch_model.bin").exists()
needs_weights = pytest.mark.skipif(not have_weights, reason="converted checkpoints not present in hf/")


def test_vbx_is_refused_early():
    with pytest.raises(ValueError, match="PLDA"):
        SuplimeDiarization(clustering="VBxClustering")


@needs_weights
def test_pipeline_from_local_dir_offline(monkeypatch):
    # no token, no network: everything must resolve inside hf/
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(str(HF))
    assert isinstance(pipeline, SuplimeDiarization)
    params = pipeline.parameters(instantiated=True)
    assert params["clustering"] == {"method": "centroid", "min_cluster_size": 12, "threshold": 0.72}
    assert params["segmentation"] == {"min_duration_off": 0.0}
    assert pipeline._plda is None


@needs_weights
def test_pipeline_direct_constructor_matches_config():
    pipeline = SuplimeDiarization(
        segmentation={"checkpoint": str(HF), "subfolder": "segmentation"},
        embedding={"checkpoint": str(HF), "subfolder": "embedding"},
    )
    pipeline.instantiate(pipeline.default_parameters())
    assert pipeline.parameters(instantiated=True)["clustering"]["threshold"] == 0.72


@needs_weights
def test_pipeline_diarizes_synthetic_audio(tmp_path):
    import wave

    import numpy as np

    sr = 16000
    t = np.arange(sr * 12) / sr
    # two "speakers": 200 Hz tone then 400 Hz tone, silence in between
    wav = np.zeros_like(t)
    wav[: sr * 5] = 0.3 * np.sin(2 * np.pi * 200 * t[: sr * 5])
    wav[sr * 7 :] = 0.3 * np.sin(2 * np.pi * 400 * t[sr * 7 :])
    path = tmp_path / "tones.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((wav * 32767).astype("<i2").tobytes())

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(str(HF))
    out = pipeline(str(path))
    ann = getattr(out, "speaker_diarization", out)
    # only the API contract is asserted (tones are not speech); it must run end to end
    assert hasattr(ann, "itertracks")
