# Training SUPlime

Everything needed to retrain the published segmentation model on stock
pyannote.audio 4.x: the exact data splits, the augmentation, the command, and the
post-processing that turns a run into the Hub checkpoint. The embedding model
(WeSpeaker `voxblink2_samresnet34_ft`) is not trained here; it is converted as is.

## How it works

`suplime-train` builds `suplime.models.SuplimeSegmentation` (WavLM-Base+ initialised
from torchaudio's `WAVLM_BASE_PLUS` bundle → 4-layer Conformer → 2 linear layers →
powerset classifier) and fine-tunes it end to end with pyannote's
`SpeakerDiarization` task (10 s chunks, up to 4 speakers per chunk, 2 per frame,
powerset cross-entropy). Chunks are sampled from the 12 training splits in
proportion to their annotated duration; each batch goes through in-batch speaker
mixing, simulated-room reverberation and MUSAN noise before the model sees it.
Validation DER on a held-out development set picks the 5 best epochs; their weights
are averaged (`suplime-soup`) and converted to the Hub layout.

## 1. Data

| Corpus | Train files | Audio used | Annotation | License |
|---|--:|---|---|---|
| AMI | 154 | headset mix (`Mix-Headset`) | [AMI-diarization-setup](https://github.com/BUT-FIT/AMI-diarization-setup) `only_words` | CC BY 4.0 |
| AMI-SDM | 152 | single distant microphone (`Array1-01`) | same RTTMs as AMI | CC BY 4.0 |
| ICSI | 72 | `interaction` mix | pyannote ICSI setup | CC BY 4.0 |
| VoxConverse | 216 | v0.3 audio | official RTTMs | CC BY 4.0 |
| CHiME-6 | 18 | reference-array channel | official transcripts → RTTM | CHiME license (research) |
| DiPCo | 5 | reference-array channel | official | CDLA-Permissive |
| NOTSOFAR-1 | 108 | single-channel devices | official (80-session split) | MIT (data: research use) |
| AISHELL-4 | 191 | 8-channel array downmixed to mono; train L+M+S rooms | TextGrid → RTTM | CC BY-SA 4.0 |
| AliMeeting | 209 | far-field array, channel 1 | TextGrid → RTTM | CC BY-SA 4.0 |
| MSDWild | 2476 | `few.train` / `few.val` | official RTTMs | research only |
| RAMC | 289 | official train split | transcript timestamps → RTTM | non-commercial |
| AVA-AVD | 297 | official train / test clips | official RTTMs | research only |
| **total** | **4187** | | | |

All audio is 16 kHz mono WAV. Training uses each corpus' official training split
(file counts above). Checkpoint selection and early stopping need a `development`
subset per corpus as well: hold out files of your own choosing that are not used for
training, and keep it small and balanced across corpora so that no single corpus
dominates the validation DER.

Corpus downloads and RTTM conversions are the corpora's own tooling; put the results
into the layout described in [`database.example.yml`](database.example.yml), copy it to
`training/database.yml` and replace `/DATA`. Sanity check:

```python
from pyannote.database import registry
registry.load_database("training/database.yml")
p = registry.get_protocol("X.SpeakerDiarization.TrainAllExt2")
print(sum(1 for _ in p.train()), "train files;", sum(1 for _ in p.development()), "development files")  # 4187 train
```

### Augmentation data (~27 GB, public)

```bash
mkdir -p aug_data && cd aug_data
wget https://www.openslr.org/resources/17/musan.tar.gz && tar xzf musan.tar.gz          # CC BY 4.0
wget https://www.openslr.org/resources/28/rirs_noises.zip && unzip -q rirs_noises.zip  # Apache 2.0
```

`suplime-train --aug-root aug_data` (or `SUPLIME_AUG_ROOT`). Only `musan/noise` and
`RIRS_NOISES/simulated_rirs` are read; a random 5000-file subset of the RIRs and all
930 noise files are held in RAM.

## 2. Train

```bash
pip install "suplime[train]"      # or pip install -e . from a checkout
suplime-train --out runs/suplime --database training/database.yml --aug-root aug_data
```

That command *is* the published recipe; the defaults are:

| | |
|---|---|
| model | WavLM-Base+ (all 12 layers, learned softmax mixture), Conformer 4 layers × 4 heads × FFN 256, kernel 31, dropout 0.1; linear 2 × 128; powerset 4 speakers / ≤ 2 per frame (11 classes) |
| chunks | 10 s, sampled by annotated duration |
| batch | 32 chunks on one GPU (`--batch-size 32 --devices 1`) |
| optimiser | AdamW, lr 5e-5, weight decay 0.01, OneCycleLR (10 % warm-up, cosine), gradient clipping 0.5 |
| precision | bf16-mixed |
| schedule | 60 epochs, early stopping patience 10 on validation DER |
| augmentation | speaker mix (p 0.5, SNR 0–5 dB) → RIR reverb (p 0.5) → MUSAN noise (p 0.5, SNR 5–15 dB) |
| seed | 42 |

An epoch is one pass over the annotated training duration (~283,000 chunks, ~8,800
optimizer steps): about 45–50 min on one RTX 5090 / RTX Pro 5000 / A100-class GPU
(fits in 32 GB), so a full run is ~48 GPU-hours; early stopping usually ends it
between epochs 50 and 60.

The run directory holds `checkpoints/EE-D.DDDD.ckpt` (epoch, validation DER; 5 best
kept), `checkpoints/last.ckpt` and `logs/version_0/metrics.csv`. Add `--tensorboard`
or `--wandb <project>` for live curves.

**Interrupted?** Run the same command again. `last.ckpt` is rewritten every
`--ckpt-minutes` (default 10) of training wall-clock, and a run whose `last.ckpt`
exists resumes from it, optimizer and schedule included. A finished run leaves a
`DONE` file and is never resumed, so a scheduler may relaunch the command blindly.
This is how the released model was trained, on preemptible cloud GPUs.

Other knobs: `--wavlm WAVLM_LARGE` (never at the default batch 32 — it OOMs; see the
SUPlime-L recipe below for the per-card batch sizes), `--devices N` for DDP (batch is per device; keep the product at 32 to stay
on-recipe), `--init-ckpt` to warm-start weights, `--cache` to reuse a prepared task
cache between runs, `--fast-dev 5` for a smoke test.

### SUPlime-L (WavLM-Large)

[SUPlime-L](https://huggingface.co/rewayai/suplime-large) is the same recipe with a
WavLM-Large backbone, trained in two stages because the large model keeps improving
past the point where the single-GPU batch becomes the bottleneck:

```bash
# stage 1: 15 epochs, effective batch 32, one GPU
suplime-train --out runs/suplime-l --database training/database.yml --aug-root aug_data \
    --wavlm WAVLM_LARGE --batch-size 8 --accumulate-grad-batches 4 --max-epochs 15
# stage 2: 20 epochs, effective batch 192, eight GPUs, fresh OneCycle at the same peak lr
suplime-train --out runs/suplime-l-b192 --database training/database.yml --aug-root aug_data \
    --wavlm WAVLM_LARGE --devices 8 --batch-size 12 --accumulate-grad-batches 2 \
    --max-epochs 20 --init-ckpt runs/suplime-l/checkpoints/last.ckpt
suplime-soup runs/suplime-l-b192/checkpoints -k 5 -o suplime_l_avg5.ckpt
```

Roughly 3× the compute of the base recipe per epoch, and it does not fit at the default
batch 32. Measured per-rank batch sizes: **8 on 24–32 GB, 12 on 40 GB** (about 22 GB in
use; 24 measured 39.3 GB and does not fit), **24 on 80 GB**. Keep the effective batch —
per-rank batch × `--accumulate-grad-batches` × `--devices` — at 32 for stage 1 and 192 for
stage 2, whichever card you land on.

## 3. Soup, convert, evaluate

```bash
suplime-soup runs/suplime/checkpoints -k 5 -o runs/suplime/suplime_avg5.ckpt
python tools/convert_checkpoints.py \
    --segmentation runs/suplime/suplime_avg5.ckpt \
    --embedding <wespeaker_samresnet34_pyannote_checkpoint> \
    --out my_hf_dir                       # + copy hf/config.yaml, README.md, LICENSE
python tools/benchmark.py --model my_hf_dir --corpus AMI_IHM --out results/AMI_IHM
```

The soup is a plain pyannote checkpoint (`Model.from_pretrained` loads it); the
converter re-stamps versions and inlines the WavLM config, and its strict reload is
the proof that the file works without this repository. `tools/benchmark.py` scores
with pyannote's benchmark convention (collar 0, overlap scored, UEM-cropped); the
clustering threshold 0.72 in `hf/config.yaml` was chosen as the single best value
across the 8 pyannote-benchmark corpora and is worth re-tuning for a new model.

## Differences from the internal training code

The model was trained with a private fork of pyannote.audio. This recipe reproduces
it on the released package: same model class, same task, same hyper-parameters. Two
fork fixes are carried in `suplime.task.SuplimeSpeakerDiarization` (padded chunk
cropping at file ends; no validation figure under bf16). The fork's multi-GPU epoch
sharding and shared task cache are not needed for the single-GPU recipe.
