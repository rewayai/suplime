from pathlib import Path

import pytest
import torch
import torchaudio
from pyannote.audio.core.task import Problem, Resolution, Specifications

from suplime.models import SuplimeSegmentation, WeSpeakerSimAMResNet34

HF = Path(__file__).resolve().parents[1] / "hf"
SEG = HF / "segmentation" / "pytorch_model.bin"
EMB = HF / "embedding" / "pytorch_model.bin"


def test_segmentation_builds_from_config_without_download():
    model = SuplimeSegmentation(wav2vec=dict(torchaudio.pipelines.WAVLM_BASE_PLUS._params))
    model.specifications = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=10.0,
        classes=["speaker#1", "speaker#2", "speaker#3", "speaker#4"],
        powerset_max_classes=2,
        permutation_invariant=True,
    )
    model.build()
    assert model.num_frames(160000) == 499
    with torch.inference_mode():
        scores = model(torch.zeros(1, 1, 160000))
    assert scores.shape == (1, 499, 11)


def test_segmentation_rejects_bundle_name():
    with pytest.raises(TypeError):
        SuplimeSegmentation(wav2vec="WAVLM_BASE_PLUS")


@pytest.mark.skipif(not SEG.exists(), reason="converted segmentation checkpoint not present")
def test_segmentation_checkpoint_loads_strict():
    from pyannote.audio import Model

    model = Model.from_pretrained(SEG, strict=True)
    assert type(model) is SuplimeSegmentation
    assert model.specifications.powerset
    assert model.specifications.duration == 10.0
    assert model.num_frames(160000) == 499
    assert isinstance(model.hparams.wav2vec, dict)


@pytest.mark.skipif(not EMB.exists(), reason="converted embedding checkpoint not present")
def test_embedding_checkpoint_loads_strict():
    from pyannote.audio import Model

    model = Model.from_pretrained(EMB, strict=True)
    assert type(model) is WeSpeakerSimAMResNet34
    assert model.dimension == 256
    with torch.inference_mode():
        emb = model(torch.zeros(2, 1, 16000 * 3))
    assert emb.shape == (2, 256)


def test_large_wrapper_normalises_and_nests_the_backbone():
    """WavLM-Large checkpoints were trained through torchaudio's normalising bundle wrapper:
    the backbone sits under `wav2vec.model.*` and the waveform is layer-normalised. Getting
    either half wrong loads the weights into the wrong module or feeds the encoder an input
    distribution it never saw (2026-09-09)."""
    cfg = dict(torchaudio.pipelines.WAVLM_LARGE._params)
    assert torchaudio.pipelines.WAVLM_LARGE._normalize_waveform is True
    m = SuplimeSegmentation(wav2vec=cfg, normalize_waveform=True).eval()  # WavLM has dropout+LayerDrop
    assert hasattr(m.wav2vec, "model"), "backbone must be nested for wav2vec.model.* keys"
    assert any(k.startswith("wav2vec.model.") for k in m.state_dict())
    assert m._feature_extractor is m.wav2vec.model.feature_extractor
    # the wrapper must apply exactly torchaudio's normalisation: layer_norm over the whole
    # waveform tensor, then the backbone (not scale-invariance — layer_norm's eps breaks that)
    x = torch.randn(1, 32000) * 0.05
    with torch.inference_mode():
        wrapped, _ = m.wav2vec.extract_features(x, num_layers=1)
        manual, _ = m.wav2vec.model.extract_features(
            torch.nn.functional.layer_norm(x, x.shape), None, 1
        )
        raw, _ = m.wav2vec.model.extract_features(x, None, 1)
    assert torch.allclose(wrapped[0], manual[0], atol=1e-6), "wrapper must normalise the waveform"
    assert not torch.allclose(wrapped[0], raw[0], atol=1e-3), "normalisation must actually change the input"


def test_base_plus_stays_unwrapped():
    """WAVLM_BASE_PLUS has the flag off; its published checkpoint stores `wav2vec.*`."""
    m = SuplimeSegmentation(wav2vec=dict(torchaudio.pipelines.WAVLM_BASE_PLUS._params))
    assert not hasattr(m.wav2vec, "model")
    assert any(k.startswith("wav2vec.feature_extractor.") for k in m.state_dict())
