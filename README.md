<p align="center"><img src="assets/logo.png" alt="SUPlime" width="280"></p>

# SUPlime speaker diarization

*SUP as in "what's up": who is speaking, and when.*

The most accurate open-weights speaker diarization we know of, on both public suites:
**15.85 macro DER on pyannote's 8-corpus benchmark** — ahead of pyannoteAI's commercial
`precision-2` (16.06) — and **20.94 across all 12 corpora**. From the research team at
[Re:WayAI](https://rewayai.ai), packaged for
[pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.x.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results_dark.svg">
  <img src="assets/results.svg" alt="Macro-average DER of SUPlime, SUPlime-L, DiariZen-L-s80-v2, pyannoteAI precision-2 and pyannote community-1 on the 8-corpus benchmark and on all 12 corpora" width="100%">
</picture>

Two models on one recipe: [**SUPlime**](https://huggingface.co/rewayai/suplime) on a
WavLM-Base+ backbone (114 M params) and
[**SUPlime-L**](https://huggingface.co/rewayai/suplime-large) on WavLM-Large (349 M, ≈ 3×
the inference cost). SUPlime-L is the better model on meeting and conversational audio;
the base model wins on far-field and dinner-party recordings (CHiME-6, DiPCo, NOTSOFAR-1)
and costs a third as much to run. Per-corpus numbers for all 12 corpora are in
[Results](#results) below.

Both model cards and the weights live on the Hugging Face Hub; this package is the code
those checkpoints point at, and without it pyannote.audio cannot instantiate them. Weights
are CC BY-NC 4.0 (non-commercial) — see [License](#license).

## Install

```bash
pip install suplime            # pulls pyannote.audio>=4.0.7,<5
```

## Use

```python
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("rewayai/suplime").to(torch.device("cuda"))
# the WavLM-Large variant is a drop-in replacement, same API and same threshold:
# pipeline = Pipeline.from_pretrained("rewayai/suplime-large").to(torch.device("cuda"))
output = pipeline("meeting.wav")
for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f} {turn.end:.2f} {speaker}")
```

No Hugging Face token is needed: the repository is not gated.

The same system without `config.yaml`:

```python
from suplime import SuplimeDiarization
pipeline = SuplimeDiarization()                       # defaults = rewayai/suplime
pipeline.instantiate(pipeline.default_parameters())   # threshold 0.72, min_cluster_size 12

L = "rewayai/suplime-large"                           # the same, for SUPlime-L
pipeline = SuplimeDiarization(segmentation={"checkpoint": L, "subfolder": "segmentation"},
                              embedding={"checkpoint": L, "subfolder": "embedding"})
```

Individual models load through the usual API:

```python
from pyannote.audio import Model
seg = Model.from_pretrained("rewayai/suplime", subfolder="segmentation")
emb = Model.from_pretrained("rewayai/suplime", subfolder="embedding")
```

Both repositories have the same layout, so every call above works with
`rewayai/suplime-large`; the embedding model is byte-identical between them.

Set `SUPLIME_FP16=0` to run the WavLM backbone in fp32 (default: fp16 autocast
on CUDA during inference — the published numbers were produced this way).

## Results

Diarization error rate (%), lower is better:

| Corpus (test unless noted) | **SUPlime** | **SUPlime-L** | DiariZen-L-s80-v2 | pyannoteAI precision-2 | pyannote community-1 |
|---|--:|--:|--:|--:|--:|
| AISHELL-4 | 11.55 | 11.21 | **10.1** | 11.4 | 11.7 |
| AliMeeting (far, ch1) | 14.45 | 14.55 | **10.8** | 15.2 | 20.3 |
| AMI (IHM, Mix-Headset) | **12.59** | 11.91 | 25.69 † | 12.9 | 17.0 |
| AMI (SDM) | 15.14 | 14.84 | **13.9** | 15.6 | 19.9 |
| AVA-AVD | 38.09 | 36.78 | 42.62 † | **37.1** | 44.6 |
| MSDWild (few.val) | 17.81 | 17.48 | **15.8** | 17.3 | 22.8 |
| RAMC | 10.86 | 11.10 | 11.0 | **10.5** | 20.8 |
| VoxConverse (v0.3) | 9.21 | 8.93 | 9.1 | **8.5** | 11.2 |
| **macro average (8)** | **16.21** | **15.85** | 17.38 | 16.06 | 21.04 |
| NOTSOFAR-1 (80-session split) | 20.04 | 22.68 | **18.86 †** | — | 27.67 † |
| ICSI | **22.97** | 22.77 | 26.54 † | — | 30.84 † |
| CHiME-6 | **48.11** | 48.88 | 48.39 † | — | 51.98 † |
| DiPCo | **30.44** | 30.92 | 37.56 † | — | 34.38 † |
| **macro average (12)** | **20.94** | 21.00 | 22.53 | — | 26.10 |

SUPlime-L leads the 8-corpus benchmark at 15.85 macro DER — ahead of `precision-2`
(16.06), the base model (16.21) and DiariZen-L-s80-v2 (17.38) — while across all 12 corpora
the base model takes it back by 0.06, the four extra sets being far-field and dinner-party
audio where the larger backbone does not pay off. Both beat DiariZen by ~1.5 DER on the
12-corpus average, with the largest margins on close-talk AMI and on CHiME-6 / DiPCo, while
DiariZen stays clearly better on the Mandarin meeting corpora and MSDWild. Per-corpus
hypothesis RTTMs ship with each model; the cards cover threshold behaviour and known
limitations.

**Scoring conditions.** Collar 0 s, overlapped speech scored, no oracle speaker count,
reference cropped to each corpus' UEM, one clustering threshold (0.72) everywhere. The
other systems' columns are their authors' published numbers
([DiariZen](https://github.com/BUT-FIT/DiariZen),
[pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1)), except where
† marks our own re-run of their open-source pipeline, unchanged and on the same files —
and for NOTSOFAR-1 the DiariZen authors report 16.7 on a different session split.

## Architecture

* **Segmentation**: WavLM (learned mixture of all layers) → 4-layer Conformer → powerset
  classifier (4 speakers / 10 s window, up to 2 simultaneous). WavLM-Base+, 114 M params
  for SUPlime; WavLM-Large, 349 M for SUPlime-L. This is the only difference between them.
* **Embedding**: WeSpeaker SimAM-ResNet34 with attentive statistics pooling
  (`voxblink2_samresnet34_ft`, 25 M params, 256-dim). Shared, byte for byte.
* **Clustering**: agglomerative (centroid linkage), threshold 0.72, `min_cluster_size` 12,
  overlap excluded from embeddings. Shared.

## Training

The full recipe is in [`training/README.md`](training/README.md): the training
splits of the 11 public corpora, the augmentation data, and
one command that reproduces the published checkpoint on stock pyannote.audio:

```bash
pip install "suplime[train]"
suplime-train --out runs/suplime --database training/database.yml --aug-root aug_data
suplime-soup runs/suplime/checkpoints -k 5 -o suplime_avg5.ckpt
```

About 48 GPU-hours on one 32 GB card; interrupted runs resume from `last.ckpt`
with the same command. Add `--wavlm WAVLM_LARGE` for the SUPlime-L variant
(roughly 3× the compute, and it needs more than one card at the same batch size).

## Development

```bash
pip install -e ".[test]"
pytest                          # checkpoint tests skip unless hf/ holds the converted weights
python tools/convert_checkpoints.py --segmentation <ckpt> --embedding <ckpt> --out hf
python tools/upload_hf.py --repo rewayai/suplime
python tools/upload_hf.py --repo rewayai/suplime-large --dir hf-large
```

## License

Code: MIT (`src/suplime/models/samresnet.py`: Apache-2.0, ported from
[WeSpeaker](https://github.com/wenet-e2e/wespeaker)). Model weights: CC BY-NC 4.0.
Non-commercial because several training corpora (RAMC, MSDWild, AVA-AVD) are
licensed for research use only and others (AISHELL-4, AliMeeting) are share-alike;
the most restrictive terms apply to the derived model. Details in the model card. Built on pyannote.audio (MIT), WavLM (MIT) and WeSpeaker (Apache-2.0).

## Diarization is not enough?

SUPlime tells you *who* spoke and *when*. If you also want to know *what* they said,
and *who they actually are*, that is our day job: [Re:WayAI](https://rewayai.ai) is
the speech API that knows who's talking.

- Transcription in 25 European languages (40 for real-time streaming), with
  overlap-robust diarization from the same team that built SUPlime
- Voice enrollment: speakers get their real names, not `SPEAKER_03`
- Real-time streaming over WebSocket, transcript Q&A, one-click DOCX / PDF exports
- On-premise deployment, audio deleted right after processing, ISO 27001

Pay-as-you-go, 50 free hours to start, no subscription. The lime approves.
