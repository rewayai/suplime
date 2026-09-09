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

"""WeSpeaker SimAM-ResNet34 speaker embedding as a pyannote.audio Model.

Fbank extraction, forward pass and receptive-field helpers come from
pyannote.audio's ``BaseWeSpeakerResNet``; only the backbone differs
(SimAM attention + attentive statistics pooling, see ``samresnet.py``).
"""

from typing import Optional

from pyannote.audio.core.task import Task
from pyannote.audio.models.embedding.wespeaker import BaseWeSpeakerResNet

from .samresnet import SimAMResNet34


class WeSpeakerSimAMResNet34(BaseWeSpeakerResNet):
    """WeSpeaker's SimAM-ResNet34 with attentive statistics pooling (256-dim).

    Weight-compatible with WeSpeaker's ``voxblink2_samresnet34`` /
    ``voxblink2_samresnet34_ft`` checkpoints once their classification head is
    dropped.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        round_to_power_of_two: bool = True,
        snip_edges: bool = True,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        fbank_centering_span: Optional[float] = None,
        task: Optional[Task] = None,
    ):
        # Every fbank parameter the base class saves has to appear here: Lightning's
        # load_from_checkpoint filters saved hyper-parameters through THIS signature, so a
        # narrower one silently drops snip_edges / fbank_centering_span / round_to_power_of_two
        # on reload and rebuilds the feature extractor with different settings.
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            round_to_power_of_two=round_to_power_of_two,
            snip_edges=snip_edges,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            fbank_centering_span=fbank_centering_span,
            task=task,
        )
        self.resnet = SimAMResNet34(num_mel_bins, 256)
