# MIT License
#
# Copyright (c) 2026 Re:WayAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Train the suplime segmentation model (WavLM > Conformer > powerset) with stock
pyannote.audio 4.x. Defaults reproduce the published checkpoint::

    suplime-train --out runs/suplime --database training/database.yml

Re-running the same command resumes from ``<out>/checkpoints/last.ckpt`` (rewritten every
``--ckpt-minutes``), so a preempted job is simply relaunched. Each epoch writes
``<out>/checkpoints/EE-D.DDDD.ckpt`` (validation DER, 5 best kept); average them with
``suplime-soup`` and convert with ``tools/convert_checkpoints.py``. See training/README.md.
"""

import argparse
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torchaudio
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pyannote.database import FileFinder, registry
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from suplime.models import SuplimeSegmentation
from suplime.task import SuplimeSpeakerDiarization

warnings.filterwarnings("ignore", module=r"^torchaudio(\.|$)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the suplime segmentation model (defaults = the published recipe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", required=True, help="run directory (checkpoints/, logs/, task cache)")
    p.add_argument("--database", default=None, help="pyannote.database yml declaring --protocol "
                   "(training/database.example.yml); omit if ~/.pyannote/database.yml declares it")
    p.add_argument("--protocol", default="X.SpeakerDiarization.TrainAllExt2")
    p.add_argument("--cache", default=None, help="task cache .npz (default <out>/task_cache.npz); "
                   "an existing file skips the protocol scan")
    # model
    p.add_argument("--wavlm", default="WAVLM_BASE_PLUS", help="torchaudio bundle: WAVLM_BASE_PLUS | WAVLM_LARGE")
    p.add_argument("--wavlm-layer", type=int, default=-1, help="-1 = learned mixture of all layers")
    p.add_argument("--conformer-layers", type=int, default=4)
    p.add_argument("--conformer-ffn", type=int, default=256)
    p.add_argument("--conformer-heads", type=int, default=4)
    p.add_argument("--conformer-dropout", type=float, default=0.1)
    # task
    p.add_argument("--duration", type=float, default=10.0, help="chunk length in seconds")
    p.add_argument("--max-spk-chunk", type=int, default=4)
    p.add_argument("--max-spk-frame", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32, help="per device")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--augment", choices=["none", "full"], default="full", help="full = speaker mix + RIR + MUSAN")
    p.add_argument("--aug-root", default=None, help="dir with musan/ and RIRS_NOISES/ (or SUPLIME_AUG_ROOT)")
    p.add_argument("--n-rirs", type=int, default=5000)
    # optimisation
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--pct-start", type=float, default=0.1, help="OneCycleLR warm-up fraction")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10, help="early stopping on validation DER")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--devices", type=int, default=1, help=">1 = DDP; effective batch = batch-size x devices")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--accelerator", default="gpu")
    # bookkeeping
    p.add_argument("--ckpt-minutes", type=float, default=10.0, help="wall-clock interval of last.ckpt; 0 = epoch end only")
    p.add_argument("--init-ckpt", default=None, help="warm-start weights from this checkpoint (fresh optimizer)")
    p.add_argument("--wandb", default=None, metavar="PROJECT", help="also log to Weights & Biases")
    p.add_argument("--tensorboard", action="store_true", help="also log to TensorBoard")
    p.add_argument("--limit-train-batches", type=int, default=None, help="smoke tests")
    p.add_argument("--limit-val-batches", type=int, default=None, help="smoke tests")
    p.add_argument("--fast-dev", type=int, default=0, help="run N train+val batches without logging/checkpoints")
    return p.parse_args(argv)


def load_wavlm(name: str):
    """(config dict, pretrained state_dict, normalize_waveform) of a torchaudio WavLM bundle.

    The ``*_LARGE`` bundles set ``_normalize_waveform``: ``get_model()`` returns the backbone
    inside a wrapper that layer-normalises the waveform, so its state_dict is nested under
    ``model.`` and the segmentation model has to be built with the same wrapper. Forgetting the
    flag makes the strict load below fail outright — see ``_NormalizedWav2Vec2``.
    """
    bundle = getattr(torchaudio.pipelines, name)
    for attr in ("_params", "get_model"):
        if not hasattr(bundle, attr):
            raise TypeError(f"torchaudio.pipelines.{name} has no {attr}; this torchaudio "
                            f"({torchaudio.__version__}) is not one suplime was tested against")
    return (dict(bundle._params), bundle.get_model().state_dict(),
            bool(getattr(bundle, "_normalize_waveform", False)))


def build_task(args, augmentation=None):
    if args.database:
        registry.load_database(args.database)
    protocol = registry.get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
    cache = args.cache or str(Path(args.out) / "task_cache.npz")
    return SuplimeSpeakerDiarization(
        protocol,
        cache=cache,
        duration=args.duration,
        max_speakers_per_chunk=args.max_spk_chunk,
        max_speakers_per_frame=args.max_spk_frame,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        augmentation=augmentation,
    )


def build_model(args, task, wav2vec: dict, wav2vec_state=None, normalize_waveform: bool = False):
    model = SuplimeSegmentation(
        wav2vec=wav2vec,
        wav2vec_layer=args.wavlm_layer,
        normalize_waveform=normalize_waveform,
        conformer={"num_heads": args.conformer_heads, "ffn_dim": args.conformer_ffn,
                   "num_layers": args.conformer_layers, "depthwise_conv_kernel_size": 31,
                   "dropout": args.conformer_dropout},
        task=task,
    )
    if wav2vec_state is not None:
        model.wav2vec.load_state_dict(wav2vec_state, strict=True)

    def configure_optimizers():
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, pct_start=args.pct_start,
                               anneal_strategy="cos",
                               total_steps=model.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # Set on the instance so the checkpoint keeps pointing at SuplimeSegmentation itself.
    model.configure_optimizers = configure_optimizers
    return model


class _WarmStart(Callback):
    """Load weights (only) from another checkpoint once the classifier exists.

    ``--init-ckpt`` and ``last.ckpt`` resume both unpickle (``weights_only=False``): a
    checkpoint can execute code on load, so point them only at checkpoints you trust.
    """

    def __init__(self, path):
        self.path = path

    def on_fit_start(self, trainer, pl_module):
        state = torch.load(self.path, map_location="cpu", weights_only=False)["state_dict"]
        own = pl_module.state_dict()
        kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        pl_module.load_state_dict(kept, strict=False)
        print(f"warm start from {self.path}: {len(kept)} tensors loaded, "
              f"{len(state) - len(kept)} skipped", flush=True)


def build_trainer(args, task, work_dir: Path):
    if args.fast_dev:
        return Trainer(accelerator=args.accelerator, devices=args.devices, precision=args.precision,
                       logger=False, fast_dev_run=args.fast_dev)
    monitor, mode = task.val_monitor  # ("DiarizationErrorRate", "min")
    ckpt_dir = str(work_dir / "checkpoints")
    periodic = args.ckpt_minutes > 0
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:02d}-{DiarizationErrorRate:.4f}",
                        monitor=monitor, mode=mode, save_top_k=5, save_last=not periodic,
                        auto_insert_metric_name=False),
        EarlyStopping(monitor=monitor, mode=mode, patience=args.patience),
        LearningRateMonitor(logging_interval="step"),
    ]
    if periodic:
        # last.ckpt on a wall clock, independent of epoch length: preemption insurance.
        # enable_version_counter must stay False or a resumed run with a different
        # --ckpt-minutes starts writing last-v1.ckpt while resume keeps reading last.ckpt.
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, monitor=None, save_top_k=0, save_last=True,
                                         train_time_interval=timedelta(minutes=args.ckpt_minutes),
                                         enable_version_counter=False))
    if args.init_ckpt and not (work_dir / "checkpoints" / "last.ckpt").exists():
        callbacks.append(_WarmStart(args.init_ckpt))
    loggers = [CSVLogger(str(work_dir), name="logs")]
    if args.tensorboard:
        from lightning.pytorch.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(str(work_dir), name="tb"))
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(WandbLogger(project=args.wandb, name=work_dir.name, id=work_dir.name,
                                   resume="allow", save_dir=str(work_dir)))
    return Trainer(
        accelerator=args.accelerator, devices=args.devices, num_nodes=args.num_nodes,
        precision=args.precision, max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.grad_clip or None, gradient_clip_algorithm="norm",
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        callbacks=callbacks, logger=loggers, log_every_n_steps=50,
    )


def run(args):
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    work_dir = Path(args.out)
    work_dir.mkdir(parents=True, exist_ok=True)
    done = work_dir / "DONE"
    if done.exists() and not args.fast_dev:
        # early stopping is not part of the checkpoint state, so a finished run must not
        # be resumed by a job scheduler that relaunches it
        print(f"{work_dir} already finished:\n{done.read_text()}", flush=True)
        return None
    print("config:", vars(args), flush=True)

    augmentation = None
    if args.augment == "full":
        from suplime.augment import RamAugment

        augmentation = RamAugment(args.aug_root, max_speakers=args.max_spk_chunk, n_rirs=args.n_rirs,
                                  seed=args.seed)
    task = build_task(args, augmentation)
    wav2vec, state, normalize = load_wavlm(args.wavlm)
    model = build_model(args, task, wav2vec, state, normalize)
    trainer = build_trainer(args, task, work_dir)

    resume = None
    if not args.fast_dev:
        last = work_dir / "checkpoints" / "last.ckpt"
        if last.exists():
            resume = str(last)
            print(f"resuming from {resume}", flush=True)
    trainer.fit(model, ckpt_path=resume, weights_only=False)

    ckpt = trainer.checkpoint_callback
    if ckpt is not None and ckpt.best_model_path:
        summary = f"best checkpoint: {ckpt.best_model_path}  ({ckpt.monitor} {float(ckpt.best_model_score):.4f})"
        print(summary, flush=True)
        if trainer.is_global_zero:
            done.write_text(summary + "\n")
    return trainer


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
