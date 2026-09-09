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


"""Speaker-diarization training task for suplime: upstream pyannote.audio's powerset task
plus two robustness fixes that the published model was trained with.

1. ``Audio.crop(..., mode="pad")`` while cutting training/validation chunks: a chunk whose
   end lands a millisecond past the audio (UEM rounding) is zero-padded instead of raising
   and killing the run.
2. Under ``precision="bf16-mixed"`` the model output is bfloat16, which NumPy cannot
   convert: upstream's ``permutate`` (Hungarian matching of targets to predictions, in
   both training and validation steps) and its first-validation-batch figure both fail.
   The matching cost is computed in float32 instead (only the resulting integer
   assignment is used, so this is exact) and the figure, a TensorBoard convenience, is
   skipped; losses and metrics are unaffected.
"""

from functools import partial

import torch
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.utils.permutation import permutate, permutate_torch

_LOW = (torch.bfloat16, torch.float16)


def _permutate_torch_fp32(y1: torch.Tensor, y2: torch.Tensor, *args, **kwargs):
    """``permutate_torch`` with the matching cost computed in float32.

    ``*args`` matters: upstream's signature takes ``cost_func`` positionally, and this
    handler replaces it for every caller — narrowing it to keyword-only would break any
    caller that passes it positionally.
    """
    return permutate_torch(y1.float(), y2.float() if y2.dtype in _LOW else y2, *args, **kwargs)


# `permutate` is a functools.singledispatch function used by the task's training and
# validation steps and by pyannote's DiarizationErrorRate metric; re-registering its
# torch.Tensor handler fixes every call site at once.
permutate.register(torch.Tensor, _permutate_torch_fp32)


class SuplimeSpeakerDiarization(SpeakerDiarization):
    """``pyannote.audio.tasks.SpeakerDiarization`` with padded chunk cropping, bf16-safe
    permutation and no validation figure. Constructor arguments are unchanged."""

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        # the task only sees its model once training starts, hence the lazy patch
        audio = self.model.audio
        if not getattr(audio, "_suplime_pad", False):
            audio.crop = partial(audio.crop, mode="pad")
            audio._suplime_pad = True
        return super().prepare_chunk(file_id, start_time, duration)

    def validation_step(self, batch, batch_idx: int):
        # upstream returns before drawing its figure whenever batch_idx > 0; the index is
        # not used for anything else in validation_step.
        return super().validation_step(batch, batch_idx + 1)
