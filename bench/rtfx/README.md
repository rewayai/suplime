# RTFx benchmark

How many times faster than real time each system diarizes, measured the way people actually
run them: the published pipeline, downloaded from the Hub, at its own default batch sizes.

**RTFx = seconds of audio / seconds of wall clock.** Higher is faster — 50x means an hour of
audio in 72 seconds. `rtfx_results.json` also carries the reciprocal (RTF) and raw timings.

Four systems:

| id | model |
|---|---|
| `suplime` | [rewayai/suplime](https://huggingface.co/rewayai/suplime) — WavLM-Base+ |
| `suplime-large` | [rewayai/suplime-large](https://huggingface.co/rewayai/suplime-large) — WavLM-Large |
| `diarizen` | [BUT-FIT/diarizen-wavlm-large-s80-md-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) |
| `community-1` | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) — **gated** |

## Run it

```bash
docker build -t rtfx bench/rtfx
docker run --rm --gpus all -v $HOME/.cache/hf:/cache/hf -e HF_TOKEN=hf_... rtfx
```

Mounting `/cache/hf` keeps the ~4 GB of weights across runs; without it every container
re-downloads them. `HF_TOKEN` is needed **only** for community-1, which is gated: the token
must belong to an account that has accepted its conditions. Without one, that system is
skipped and the other three still run.

Narrow the run by appending arguments:

```bash
docker run --rm --gpus all rtfx --systems suplime suplime-large --repeats 5
docker run --rm --gpus all rtfx --clips /bench/clips/clip_30min.wav
```

Outside Docker, `python rtfx.py` works too if the systems are importable — it falls back to
the current interpreter when `/opt/venv/*` is absent.

## What is measured

- **Timed:** the single call that turns a file into speaker turns — segmentation, embedding
  and clustering. Median of `--repeats` runs (default 3) after one warm-up run.
- **Not timed:** model download and load (reported separately as `load s`), CUDA context
  creation, and audio decoding of the warm-up clip.
- Each system keeps its own shipped defaults — batch sizes, window step, clustering. That is
  the honest comparison: it is what a user gets, not a matched-hyper-parameter study.

Three clip lengths, because RTFx is not a single number — a 1-minute file is dominated by
fixed costs, a 30-minute one shows steady-state throughput:

| clip | length |
|---|---|
| `clip_01min.wav` | 1 min |
| `clip_05min.wav` | 5 min |
| `clip_30min.wav` | 30 min |

They are cut from AMI ES2004b (Mix-Headset, 16 kHz mono, four speakers), which is
**CC BY 4.0** — attribute the AMI Consortium if you redistribute them. `prepare_clips.py`
regenerates them from any long 16 kHz mono wav.

## Why two virtualenvs

They cannot share one. SUPlime and community-1 need pyannote.audio 4.x with numpy 2;
DiariZen vendors a pyannote.audio **3.1.1** fork pinned to numpy 1.26. The image therefore
carries `/opt/venv/suplime` and `/opt/venv/diarizen`, and `rtfx.py` dispatches each system to
the right interpreter. Torch versions are the ones each system is known to work with —
cu130 for pyannote 4.x, cu128 for DiariZen (2.8 is the first release with sm_120 kernels, so
both work on Blackwell and on Hopper).

## Where the time goes

`--profile` times the pipeline's internal stages via pyannote's hook, on the longest clip:

```bash
docker run --rm --gpus all rtfx --systems suplime suplime-large --profile
```

On CPU the split for SUPlime is **82% embedding extraction, 17% segmentation** — the WavLM
backbone is the minority of the work, which is why the Base+ and Large variants land within
~20% of each other despite a 3.3x difference in backbone size. Confirm the split on your GPU
before drawing conclusions from it. DiariZen is skipped (its pipeline takes no hook).

## GPUs to cover

RTX 5090, RTX PRO 5000 Blackwell, RTX PRO 6000 Blackwell, H100. Run the same image on each
and keep the JSON; the GPU name and memory are recorded in every result file. On vast.ai,
rent with the Docker image directly rather than a VM — several GPU types have no KVM
enabled, and the image needs only the driver.

Rough expectations from our fleet logs, for a sanity check rather than a target: SUPlime
around 33x on a 5090 and 22x on a PRO 5000; DiariZen roughly 2x faster than SUPlime;
community-1 roughly 2.8x faster. The open question this benchmark settles is SUPlime-L
versus SUPlime — the model cards claim about 3x the inference cost, and same-GPU data so
far suggests it is closer to 1.2x.
