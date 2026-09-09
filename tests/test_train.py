"""CPU-only tests of the training path: synthetic 2-speaker corpus, a tiny WavLM config,
a few optimizer steps, checkpoint soup, and re-loading the soup through pyannote."""

import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from suplime import soup, train
from suplime.models import SuplimeSegmentation

SR = 16000


def _tiny_wavlm() -> dict:
    cfg = dict(torchaudio.pipelines.WAVLM_BASE_PLUS._params)
    cfg.update(encoder_embed_dim=32, encoder_num_layers=2, encoder_num_heads=2,
               encoder_ff_interm_features=64, extractor_conv_layer_config=[(8, 10, 5), (8, 8, 4), (8, 4, 2), (8, 4, 2)],
               encoder_projection_dropout=0.0)
    return cfg


def _write_wav(path: Path, seconds: float):
    x = (np.random.RandomState(0).randn(int(seconds * SR)) * 3000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Two 40 s files with two alternating speakers; train == dev (it is a smoke test)."""
    root = tmp_path_factory.mktemp("corpus")
    (root / "audio").mkdir()
    (root / "rttm").mkdir()
    (root / "uem").mkdir()
    uris = ["a", "b"]
    for uri in uris:
        _write_wav(root / "audio" / f"{uri}.wav", 40.0)
        with open(root / "rttm" / f"{uri}.rttm", "w") as f:
            for i in range(10):
                spk = f"{uri}_spk{i % 2}"
                f.write(f"SPEAKER {uri} 1 {4 * i:.3f} 4.500 <NA> <NA> {spk} <NA> <NA>\n")
        (root / "uem" / f"{uri}.uem").write_text(f"{uri} NA 0.000 40.000\n")
    (root / "train.txt").write_text("\n".join(uris) + "\n")
    yml = root / "database.yml"
    yml.write_text(f"""Databases:
  Synth: {root}/audio/{{uri}}.wav
Protocols:
  Synth:
    SpeakerDiarization:
      tiny:
        train:
          uri: {root}/train.txt
          annotation: {root}/rttm/{{uri}}.rttm
          annotated: {root}/uem/{{uri}}.uem
        development:
          uri: {root}/train.txt
          annotation: {root}/rttm/{{uri}}.rttm
          annotated: {root}/uem/{{uri}}.uem
""")
    return yml


def _argv(out: Path, corpus: Path, *extra):
    return ["--out", str(out), "--database", str(corpus), "--protocol", "Synth.SpeakerDiarization.tiny",
            "--duration", "5", "--max-spk-chunk", "2", "--batch-size", "2", "--num-workers", "0",
            "--augment", "none", "--accelerator", "cpu", "--precision", "32", "--conformer-layers", "1",
            "--conformer-ffn", "32", "--conformer-heads", "2", *extra]


@pytest.fixture(autouse=True)
def tiny_wavlm(monkeypatch):
    monkeypatch.setattr(train, "load_wavlm", lambda name: (_tiny_wavlm(), None, False))


def test_fast_dev_runs_two_batches(tmp_path, corpus):
    train.main(_argv(tmp_path / "run", corpus, "--fast-dev", "2"))


def test_fast_dev_bf16_mixed_on_cpu(tmp_path, corpus):
    # exercises the float32 cast in permutate and the skipped validation figure
    argv = _argv(tmp_path / "run", corpus, "--fast-dev", "2")
    argv[argv.index("--precision") + 1] = "bf16-mixed"
    train.main(argv)


def test_train_checkpoints_soup_and_reload(tmp_path, corpus):
    out = tmp_path / "run"
    trainer = train.run(train.parse_args(_argv(out, corpus, "--max-epochs", "2", "--limit-train-batches", "2",
                                               "--limit-val-batches", "1", "--ckpt-minutes", "0")))
    ckpts = sorted((out / "checkpoints").glob("*.ckpt"))
    names = [p.name for p in ckpts]
    assert "last.ckpt" in names
    scored = soup.best_checkpoints(out / "checkpoints", 5)
    assert len(scored) == 2
    assert (out / "logs").exists()  # CSVLogger
    assert trainer.current_epoch == 2

    # a finished run is not resumed (DONE marker), nothing new is written
    assert (out / "DONE").read_text().startswith("best checkpoint:")
    assert train.run(train.parse_args(_argv(out, corpus, "--max-epochs", "2", "--limit-train-batches", "2",
                                            "--limit-val-batches", "1", "--ckpt-minutes", "0"))) is None
    assert sorted((out / "checkpoints").glob("*.ckpt")) == ckpts

    avg = out / "avg.ckpt"
    soup.main([str(out / "checkpoints"), "-k", "2", "-o", str(avg)])
    from pyannote.audio import Model

    model = Model.from_pretrained(avg, strict=True)
    assert type(model) is SuplimeSegmentation
    assert model.specifications.powerset
    with torch.inference_mode():
        scores = model(torch.zeros(1, 1, 5 * SR))
    assert scores.shape[0] == 1 and scores.shape[2] == model.specifications.num_powerset_classes


def test_build_model_wraps_the_backbone_when_the_bundle_normalises(tmp_path, corpus):
    """The *_LARGE bundles hand back a normalising wrapper, so build_model has to be told:
    without the flag the model is built unwrapped and the strict load of the bundle weights
    fails (`wav2vec.model.*` against `wav2vec.*`). Regression for the WAVLM_LARGE recipe."""
    args = train.parse_args(_argv(tmp_path / "run", corpus))
    cfg = _tiny_wavlm()
    cfg["encoder_max_distance"] = 800  # what makes it a WavLM rather than a wav2vec2 config

    plain = train.build_model(args, None, cfg)
    wrapped = train.build_model(args, None, cfg, normalize_waveform=True)
    assert not hasattr(plain.wav2vec, "model")
    assert wrapped.hparams.normalize_waveform is True
    assert any(k.startswith("wav2vec.model.") for k in wrapped.state_dict())

    # the wrapped model accepts exactly the state_dict a normalising bundle produces
    nested = {f"model.{k}": v for k, v in plain.wav2vec.state_dict().items()}
    wrapped.wav2vec.load_state_dict(nested, strict=True)
    with pytest.raises(RuntimeError):
        plain.wav2vec.load_state_dict(nested, strict=True)


LARGE_CACHED = Path(torch.hub.get_dir()) / "checkpoints" / "wavlm_large.pth"


@pytest.mark.skipif(not LARGE_CACHED.exists(), reason="WAVLM_LARGE not in the torch hub cache")
def test_wavlm_large_bundle_loads_into_the_model_it_builds(tmp_path, corpus, monkeypatch):
    """End to end on the real bundle (1.2 GB, cache only): load_wavlm reports the flag and the
    weights load strictly into the model build_model makes from it."""
    monkeypatch.undo()  # this one wants the real load_wavlm, not the tiny stub
    cfg, state, normalize = train.load_wavlm("WAVLM_LARGE")
    assert normalize is True
    assert next(iter(state)).startswith("model.")
    args = train.parse_args(_argv(tmp_path / "run", corpus))
    model = train.build_model(args, None, cfg, state, normalize)
    assert model.hparams.wav2vec["encoder_embed_dim"] == 1024


def test_augment_never_writes_into_the_callers_batch(tmp_path):
    """torch_audiomentations returns the input tensor itself when no element is selected for
    mixing (~6% of batches at B=4), and reshape() is a view — so reverb/noise used to write
    straight into the caller's tensor, which also made the assertion below vacuous."""
    root = tmp_path / "aug"
    (root / "musan" / "noise" / "free").mkdir(parents=True)
    (root / "RIRS_NOISES" / "simulated_rirs" / "small").mkdir(parents=True)
    _write_wav(root / "musan" / "noise" / "free" / "n1.wav", 2.0)
    _write_wav(root / "RIRS_NOISES" / "simulated_rirs" / "small" / "r1.wav", 0.3)
    from suplime.augment import RamAugment

    aug = RamAugment(root, max_speakers=2, n_rirs=1)
    aug.P_REVERB = aug.P_NOISE = 1.0
    for seed in range(24):          # >> 1/0.5**4, so the no-mix path is certainly exercised
        torch.manual_seed(seed)
        x = torch.randn(4, 1, 3 * SR)
        before = x.clone()
        out = aug(samples=x, sample_rate=SR, targets=torch.zeros(4, 1, 150, 2))
        assert torch.equal(x, before), f"input mutated in place (seed {seed})"
        assert not torch.allclose(out.samples, x)


def test_soup_takes_version_suffixed_checkpoints_and_rejects_foreign_ones(tmp_path):
    """A preempted run re-validating to the same DER writes EE-D.DDDD-v1.ckpt; those were
    silently skipped, so `-k 5` could average fewer. And name-only matching let checkpoints
    from a different run broadcast into a wrong average."""
    d = tmp_path / "checkpoints"
    d.mkdir()

    def ckpt(path, w, shape=(3,), hp={"a": 1}):
        torch.save({"state_dict": {"w": torch.full(shape, float(w))},
                    "hyper_parameters": hp, "pyannote.audio": {"x": 1}}, path)

    ckpt(d / "05-0.1000.ckpt", 1.0)
    ckpt(d / "05-0.1000-v1.ckpt", 2.0)     # same epoch+DER, written after a resume
    ckpt(d / "06-0.2000.ckpt", 3.0)
    picked = soup.best_checkpoints(d, 2)
    assert [p.name for p in picked] == ["05-0.1000-v1.ckpt", "06-0.2000.ckpt"], picked

    ckpt(d / "07-0.3000.ckpt", 4.0, shape=(1,))          # broadcastable but wrong
    with pytest.raises(SystemExit, match="different runs"):
        soup.average_checkpoints([d / "05-0.1000.ckpt", d / "07-0.3000.ckpt"], tmp_path / "o.ckpt")

    ckpt(d / "08-0.4000.ckpt", 5.0, hp={"a": 2})         # same shapes, different run
    with pytest.raises(SystemExit, match="different runs"):
        soup.average_checkpoints([d / "05-0.1000.ckpt", d / "08-0.4000.ckpt"], tmp_path / "o.ckpt")

    with pytest.raises(SystemExit, match="-k must be at least 2"):
        soup.best_checkpoints(d, 1)


def test_warm_start_refuses_a_foreign_checkpoint(tmp_path, corpus):
    """--init-ckpt from another architecture matched almost nothing and degraded to a
    from-scratch run at a warm-start learning rate, discovered days later."""
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {"nothing.like.this": torch.zeros(3)}}, foreign)
    # not --fast-dev: that path builds a bare Trainer with no callbacks, so _WarmStart
    # (and therefore --init-ckpt) is not attached at all
    argv = _argv(tmp_path / "run", corpus, "--max-epochs", "1", "--limit-train-batches", "1",
                 "--limit-val-batches", "1", "--ckpt-minutes", "0", "--init-ckpt", str(foreign))
    with pytest.raises(SystemExit, match="not a checkpoint of this architecture"):
        train.run(train.parse_args(argv))


def test_missing_init_ckpt_fails_before_any_work(tmp_path, corpus):
    with pytest.raises(SystemExit, match="does not exist"):
        train.run(train.parse_args(_argv(tmp_path / "run", corpus, "--init-ckpt", "/no/such.ckpt")))


def test_soup_averages_float_tensors(tmp_path):
    def ckpt(v):
        return {"state_dict": {"w": torch.full((3,), float(v)), "n": torch.tensor(7)},
                "optimizer_states": [1], "pyannote.audio": {"x": 1}}

    a, b = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    torch.save(ckpt(1.0), a)
    torch.save(ckpt(3.0), b)
    out = soup.average_checkpoints([a, b], tmp_path / "avg.ckpt")
    assert torch.allclose(out["state_dict"]["w"], torch.full((3,), 2.0))
    assert out["state_dict"]["n"].item() == 7
    assert "optimizer_states" not in out and out["pyannote.audio"] == {"x": 1}


def test_ram_augment_on_synthetic_data(tmp_path):
    root = tmp_path / "aug"
    (root / "musan" / "noise" / "free").mkdir(parents=True)
    (root / "RIRS_NOISES" / "simulated_rirs" / "small").mkdir(parents=True)
    _write_wav(root / "musan" / "noise" / "free" / "n1.wav", 2.0)
    _write_wav(root / "RIRS_NOISES" / "simulated_rirs" / "small" / "r1.wav", 0.3)
    _write_wav(root / "RIRS_NOISES" / "simulated_rirs" / "small" / "r2.wav", 0.2)
    from suplime.augment import RamAugment

    aug = RamAugment(root, max_speakers=2, n_rirs=1)
    aug.P_REVERB = aug.P_NOISE = 1.0
    x = torch.randn(4, 1, 3 * SR)
    y = torch.zeros(4, 1, 150, 2)  # (batch, channel, frame, class), as the task passes it
    out = aug(samples=x, sample_rate=SR, targets=y)
    assert out.samples.shape == x.shape and out.targets.shape[0] == 4
    assert not torch.allclose(out.samples, x)
    assert torch.isfinite(out.samples).all()
    aug.eval()
    assert torch.equal(aug(samples=x, sample_rate=SR, targets=y).samples, x)
