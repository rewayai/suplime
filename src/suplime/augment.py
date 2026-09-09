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


"""In-RAM training augmentation: in-batch speaker mixing (pyannote), room reverberation with
simulated RIRs, and MUSAN background noise.

RIRs and noise are loaded into memory once (shared with dataloader workers through fork
copy-on-write), so augmentation does no per-batch disk I/O. Reverb is a batched FFT
convolution with RMS preserved. Speaker mixing updates the diarization targets; reverb
and noise are label-preserving.

Data layout (``--aug-root`` / ``SUPLIME_AUG_ROOT``)::

    <root>/musan/noise/**/*.wav              MUSAN, https://www.openslr.org/17  (CC BY 4.0)
    <root>/RIRS_NOISES/simulated_rirs/**/*.wav  https://www.openslr.org/28  (Apache 2.0)

Both are 16 kHz mono 16-bit WAV as distributed; anything else is resampled/downmixed on load.
"""

import glob
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from pyannote.audio.augmentation import MixSpeakerDiarization

SAMPLE_RATE = 16000


class _Out:
    __slots__ = ("samples", "targets")

    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets


def _load_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # (channels, time)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav[0].contiguous()


class RamAugment:
    """speaker mix -> reverb (p=0.5) -> MUSAN noise (p=0.5, SNR 5-15 dB), all in RAM.

    Implements the interface pyannote's task expects from an ``augmentation``:
    ``.train(mode)`` and ``__call__(samples, sample_rate, targets)`` returning an object
    with ``.samples`` (batch, channel, time) and ``.targets``.

    Parameters
    ----------
    root : path
        Directory holding ``musan/`` and ``RIRS_NOISES/`` (see module docstring).
    max_speakers : int
        ``max_speakers_per_chunk`` of the task; mixing never exceeds it.
    n_rirs : int
        Random subset of RIRs to preload (the full set is ~60k files; 5000 is plenty).
    seed : int
        Seed of the RIR subset selection (the per-batch draws use ``random``).
    """

    SNR_DB = (5.0, 15.0)
    P_REVERB = 0.5
    P_NOISE = 0.5

    def __init__(self, root: Optional[os.PathLike] = None, max_speakers: int = 4,
                 n_rirs: int = 5000, seed: int = 0):
        root = Path(root or os.environ.get("SUPLIME_AUG_ROOT", "aug_data"))
        noise_dir, rir_dir = root / "musan" / "noise", root / "RIRS_NOISES" / "simulated_rirs"
        for p in (noise_dir, rir_dir):
            if not p.is_dir():
                raise FileNotFoundError(f"augmentation data not found: {p} (see suplime/augment.py)")
        rng = random.Random(seed)
        rir_paths = sorted(glob.glob(str(rir_dir / "**" / "*.wav"), recursive=True))
        if len(rir_paths) > n_rirs:
            rir_paths = rng.sample(rir_paths, n_rirs)
        noise_paths = sorted(glob.glob(str(noise_dir / "**" / "*.wav"), recursive=True))
        print(f"[RamAugment] preloading {len(rir_paths)} RIRs + {len(noise_paths)} noise files "
              f"from {root} ...", flush=True)
        self.rirs = [_load_mono(p) for p in rir_paths]
        self.noises = [_load_mono(p) for p in noise_paths]
        if not self.rirs or not self.noises:
            raise FileNotFoundError(f"no RIR / noise WAVs under {root}")
        self.mix = MixSpeakerDiarization(
            max_num_speakers=max_speakers, min_snr_in_db=0.0, max_snr_in_db=5.0,
            p=0.5, output_type="dict",
        )
        self.training = True
        mb = sum(t.numel() for t in self.rirs + self.noises) * 4 / 1e6
        print(f"[RamAugment] ready ({mb:.0f} MB in RAM)", flush=True)

    def train(self, mode: bool = True):
        self.training = mode
        self.mix.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @torch.no_grad()
    def _reverb(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        idx = [random.randrange(len(self.rirs)) for _ in range(B)]
        R = max(self.rirs[i].numel() for i in idx)
        h = x.new_zeros(B, R)
        for b, i in enumerate(idx):
            h[b, : self.rirs[i].numel()] = self.rirs[i]
        n = 1
        while n < T + R - 1:
            n <<= 1
        y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)[:, :T]
        rms_in = x.pow(2).mean(1, keepdim=True).sqrt()
        rms_out = y.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        return y * (rms_in / rms_out)

    @torch.no_grad()
    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        nb = x.new_empty(B, T)
        for b in range(B):
            nz = self.noises[random.randrange(len(self.noises))]
            if nz.numel() < T:
                nz = nz.repeat(T // nz.numel() + 1)
            s = random.randrange(0, nz.numel() - T + 1)
            nb[b] = nz[s : s + T]
        sig_rms = x.pow(2).mean(1, keepdim=True).sqrt()
        noi_rms = nb.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        snr = x.new_empty(B, 1).uniform_(*self.SNR_DB)
        return x + nb * (sig_rms / (noi_rms * 10 ** (snr / 20)))

    @torch.no_grad()
    def __call__(self, samples, sample_rate=SAMPLE_RATE, targets=None):
        if not self.training:
            return _Out(samples, targets)
        out = self.mix(samples=samples, sample_rate=sample_rate, targets=targets)
        x, targets = out.samples, out.targets  # (B, 1, T)
        if x is samples:
            # torch_audiomentations returns the input tensor itself when no element was
            # selected for mixing (p=0.5 per element, so ~6% of batches at B=4). reshape()
            # below is a view, so reverb/noise would then write into the caller's batch.
            x = x.clone()
        B, C, T = x.shape
        flat = x.reshape(B, T)
        m = torch.rand(B) < self.P_REVERB
        if m.any():
            flat[m] = self._reverb(flat[m])
        m = torch.rand(B) < self.P_NOISE
        if m.any():
            flat[m] = self._noise(flat[m])
        return _Out(flat.reshape(B, C, T), targets)
