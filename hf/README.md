---
license: cc-by-nc-4.0
library_name: pyannote-audio
pipeline_tag: automatic-speech-recognition
tags:
  - pyannote
  - pyannote-audio
  - pyannote-audio-pipeline
  - audio
  - voice
  - speech
  - speaker
  - speaker-diarization
  - speaker-segmentation
  - speaker-embedding
  - wavlm
  - conformer
  - wespeaker
datasets:
  - AMI
  - ICSI
  - VoxConverse
  - CHiME-6
  - DiPCo
  - NOTSOFAR-1
  - AISHELL-4
  - AliMeeting
  - MSDWild
  - RAMC
  - AVA-AVD
inference: false
---

<p align="center"><img src="logo.png" alt="SUPlime" width="280"></p>

# SUPlime — speaker diarization (WavLM-Base+)

*SUP as in "what's up": who is speaking, and when.*

Open-weights speaker diarization by [Re:WayAI](https://rewayai.ai), packaged for
[pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.x. One 16 kHz mono
recording in, who-spoke-when out; no speaker count needed.

This is the **WavLM-Base+** model: 114 M parameters, the cheaper of the two and the better one across all 12 corpora we score. Its sibling [SUPlime-L](https://huggingface.co/rewayai/suplime-large) swaps in a WavLM-Large backbone and leads pyannote's 8-corpus benchmark — **15.85 macro DER, ahead of pyannoteAI's commercial `precision-2` (16.06)** — at roughly 3× the inference cost. Same recipe, same embedding, same clustering, same threshold; pick by domain, not by size, see [Which variant](#which-variant).

**Weights are released under CC BY-NC 4.0 (non-commercial)** because part of the
training data is licensed for research use only, which rules out commercial use of
models derived from it — see [LICENSE](LICENSE) and the rationale below.

## Quick start

```bash
pip install suplime      # installs pyannote.audio>=4.0.7 and the model code
```

```python
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("rewayai/suplime").to(torch.device("cuda"))
output = pipeline("meeting.wav")
for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f} {turn.end:.2f} {speaker}")
```

No Hugging Face token or gated access is required. The `suplime` package is
needed because the checkpoints reference model classes that do not exist in
pyannote.audio itself; without it pyannote raises `MissingDependency`.

Optional knobs:

```python
pipeline("meeting.wav", num_speakers=3)          # or min_speakers= / max_speakers=
pipeline.instantiate({"clustering": {"threshold": 0.68, "min_cluster_size": 12, "method": "centroid"},
                      "segmentation": {"min_duration_off": 0.0}})   # lower threshold = more speakers
```

`SUPLIME_FP16=0` runs the WavLM backbone in fp32 (default: fp16 autocast on CUDA
during inference — all numbers below were produced with the default).

## Architecture

| Stage | Model | Details |
|---|---|---|
| Segmentation | WavLM-Base+ → Conformer → powerset | 12 WavLM layers mixed with learned softmax weights; 4 Conformer layers (4 heads, FFN 256, depthwise kernel 31, dropout 0.1); 2 linear layers (128); powerset output over 4 speakers with ≤ 2 simultaneous (11 classes); 10 s windows, 20 ms frames, window step 1 s. 114 M parameters. |
| Embedding | WeSpeaker SimAM-ResNet34 + attentive statistics pooling | `voxblink2_samresnet34_ft` (VoxBlink2 pre-training, VoxCeleb2 fine-tuning), converted to a pyannote model; 256-dim, cosine metric; overlapping speech excluded from pooling. 25 M parameters, shared with SUPlime-L byte for byte. |
| Clustering | Agglomerative, centroid linkage | threshold 0.72, `min_cluster_size` 12 (pyannote's standard AHC). |

The segmentation model was fine-tuned end to end (WavLM unfrozen) on the training
splits of 11 public corpora — AMI (IHM and SDM), ICSI, VoxConverse, CHiME-6, DiPCo,
NOTSOFAR-1, AISHELL-4, AliMeeting, MSDWild, RAMC and AVA-AVD (4,187 files) — with
in-batch speaker mixing, simulated-RIR reverberation and MUSAN noise; AdamW, lr 5e-5,
weight decay 0.01, OneCycle schedule, gradient clipping 0.5, bf16 mixed precision,
10 s chunks. Batch 32, 60 epochs on a single GPU with early stopping (patience 10). The released weights are the average of the 5 best
validation checkpoints. The complete recipe — training splits, augmentation data, one
`suplime-train` command on stock pyannote.audio — is published in
[`training/`](https://github.com/rewayai/suplime/tree/main/training) of the GitHub
repository. Repository layout follows pyannote's conventions:
`config.yaml` (pipeline), `segmentation/pytorch_model.bin`, `embedding/pytorch_model.bin`.

## Results

<img src="results.png" alt="Macro-average DER of SUPlime, SUPlime-L, DiariZen-L-s80-v2, pyannoteAI precision-2 and pyannote community-1 on the 8-corpus benchmark and on all 12 corpora" width="100%">

Diarization error rate (%), lower is better:

| Corpus (test unless noted) | **SUPlime** | **SUPlime-L** | DiariZen-L-s80-v2 | pyannoteAI precision-2 | pyannote community-1 |
|---|--:|--:|--:|--:|--:|
| AISHELL-4 | 11.55 | 11.21 | **10.1** | 11.4 | 11.7 |
| AliMeeting (far, ch1) | 14.45 | 14.55 | **10.8** | 15.2 | 20.3 |
| AMI (IHM, Mix-Headset) | 12.59 | **11.91** | 25.69 † | 12.9 | 17.0 |
| AMI (SDM) | 15.14 | 14.84 | **13.9** | 15.6 | 19.9 |
| AVA-AVD | 38.09 | **36.78** | 42.62 † | 37.1 | 44.6 |
| MSDWild (few.val) | 17.81 | 17.48 | **15.8** | 17.3 | 22.8 |
| RAMC | 10.86 | 11.10 | 11.0 | **10.5** | 20.8 |
| VoxConverse (v0.3) | 9.21 | 8.93 | 9.1 | **8.5** | 11.2 |
| **macro average (8)** | 16.21 | **15.85** | 17.38 | 16.06 | 21.04 |
| NOTSOFAR-1 (80-session split) | 20.04 | 22.68 | **18.86 †** | — | 27.67 † |
| ICSI | 22.97 | **22.77** | 26.54 † | — | 30.84 † |
| CHiME-6 | **48.11** | 48.88 | 48.39 † | — | 51.98 † |
| DiPCo | **30.44** | 30.92 | 37.56 † | — | 34.38 † |
| **macro average (12)** | **20.94** | 21.00 | 22.53 | — | 26.10 |

SUPlime-L leads the 8-corpus benchmark at 15.85 macro DER — ahead of `precision-2`
(16.06), SUPlime (16.21) and DiariZen-L-s80-v2 (17.38) — while across all 12 corpora
SUPlime takes it back by 0.06, the four extra sets being far-field and dinner-party audio
where the larger backbone does not pay off. Both beat DiariZen by ~1.5 DER on the 12-corpus
average, with the largest margins on close-talk AMI and on CHiME-6 / DiPCo, while DiariZen
stays clearly better on the Mandarin meeting corpora and MSDWild. The per-corpus hypothesis
RTTMs behind every number are in [`reproducible_research/`](reproducible_research/).

**Scoring conditions.** Collar 0 s, overlapped speech scored, no oracle speaker count,
reference cropped to each corpus' UEM, one clustering threshold (0.72) everywhere. The
other systems' columns are their authors' published numbers
([DiariZen](https://github.com/BUT-FIT/DiariZen),
[pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1)), except where
† marks our own re-run of their open-source pipeline, unchanged and on the same files —
and for NOTSOFAR-1 the DiariZen authors report 16.7 on a different session split.

## Which variant

| | **SUPlime** (this model) | [SUPlime-L](https://huggingface.co/rewayai/suplime-large) |
|---|--:|--:|
| Backbone | WavLM-Base+ | WavLM-Large |
| Parameters (segmentation) | 114 M | 349 M |
| macro DER, 8-corpus benchmark | 16.21 | **15.85** |
| macro DER, all 12 corpora | **20.94** | 21.00 |
| Relative inference cost | 1× | ≈ 3× |

Take SUPlime-L for meeting and conversational audio of the kind the 8-corpus benchmark
covers, and when accuracy matters more than throughput. Take SUPlime for far-field and
dinner-party audio (CHiME-6, DiPCo, NOTSOFAR-1), for CPU or edge deployment, or when you
want the cheaper model that stays within 0.4 DER of the larger one almost everywhere. They
share the embedding, the clustering and the operating threshold, so switching is a one-line
change.

## Limitations

- **Threshold.** 0.72 is a DER-optimal compromise across corpora. AISHELL-4 and
  AliMeeting want 0.68 (10.00 and 13.48) and degrade sharply above 0.74 (AISHELL-4:
  19.11 at 0.80); AMI-IHM wants 0.84 and CHiME-6 gains 4.5 DER at 0.80 (46.07 →
  41.56). Tune on your own data. A DER-optimal threshold tends to over-merge speakers;
  if your downstream metric is speaker-attributed WER, a slightly higher threshold is
  usually better.
- A third of SUPlime-L's segmentation compute and memory, and ahead of it on the
  12-corpus macro average — but 0.36 DER behind it on the 8-corpus benchmark.
- The powerset head models at most 2 simultaneous speakers per frame and 4 speakers
  per 10 s window; recordings with heavy 3-way overlap are under-served.
- Trained on 16 kHz meeting / conversational / broadcast / movie audio; telephone
  (8 kHz) speech is out of domain and is resampled, not modelled.
- The embedding model is trained on VoxBlink2 / VoxCeleb2 (largely English YouTube
  speech); no fairness evaluation across languages or demographics has been done.
- English and Mandarin dominate the training data.

## License and attribution

**Model weights** (`segmentation/`, `embedding/`): [CC BY-NC 4.0](LICENSE).
The segmentation model was trained on corpora with mixed terms: several are
licensed for research / non-commercial use only (RAMC, MSDWild, AVA-AVD) and
others are share-alike (AISHELL-4, AliMeeting, CC BY-SA 4.0). A model derived
from that data is bound by the most restrictive of those terms, so the weights
cannot be offered for commercial use and are released under CC BY-NC 4.0. The embedding weights are a conversion of WeSpeaker's
`voxblink2_samresnet34_ft` and remain subject to WeSpeaker's and the VoxBlink2 /
VoxCeleb2 data terms.

**Code** (`suplime` package): MIT; the SimAM-ResNet backbone is Apache-2.0,
ported from [WeSpeaker](https://github.com/wenet-e2e/wespeaker).

Built on [pyannote.audio](https://github.com/pyannote/pyannote-audio) (Bredin et al.),
[WavLM](https://arxiv.org/abs/2110.13900) (Chen et al., Microsoft, MIT) and
[WeSpeaker](https://arxiv.org/abs/2210.17016) (Wang et al.). The powerset
segmentation formulation follows Plaquet & Bredin, *Powerset multi-class cross
entropy loss for neural speaker diarization* (Interspeech 2023); the WavLM +
Conformer segmentation design is in the spirit of
[DiariZen](https://github.com/BUT-FIT/DiariZen) (Han et al., BUT).

## Citation

```bibtex
@misc{suplime2026,
  title  = {SUPlime: open-weights speaker diarization for pyannote.audio},
  author = {Re:WayAI},
  year   = {2026},
  url    = {https://huggingface.co/rewayai/suplime}
}
```

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
