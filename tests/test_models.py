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
