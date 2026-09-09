# MIT License
#
# Copyright (c) 2023- CNRS
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

"""Suplime segmentation model: WavLM > Conformer > feed-forward > powerset classifier.

Derived from pyannote.audio's ``SSeRiouSS`` (wav2vec > LSTM > ...), with the LSTM
replaced by a torchaudio Conformer. The state_dict layout (``wav2vec.*``,
``wav2vec_weights``, ``conformer.*``, ``linear.*``, ``classifier.*``) is what the
published checkpoint contains, so attribute names here must not change.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)


class _NormalizedWav2Vec2(nn.Module):
    """torchaudio's large-bundle wrapper, reimplemented so suplime does not depend on a private
    torchaudio class (``pipelines._wav2vec2.utils._Wav2Vec2Model``).

    The ``*_LARGE`` bundles set ``_normalize_waveform=True``: ``bundle.get_model()`` returns the
    backbone inside a wrapper that layer-normalises the waveform before the encoder and holds the
    backbone under ``.model``. Checkpoints trained that way store ``wav2vec.model.*``, and dropping
    the normalisation would silently feed the encoder a different input distribution from the one
    it was trained on. WAVLM_BASE_PLUS has the flag off, so its checkpoints are unwrapped.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def extract_features(self, waveforms, lengths=None, num_layers=None):
        waveforms = F.layer_norm(waveforms, waveforms.shape)
        return self.model.extract_features(waveforms, lengths, num_layers)

    def forward(self, waveforms, lengths=None):
        waveforms = F.layer_norm(waveforms, waveforms.shape)
        return self.model(waveforms, lengths)


class SuplimeSegmentation(Model):
    """WavLM (learned layer mixture) > Conformer > linear > classifier.

    Parameters
    ----------
    wav2vec : dict
        Keyword arguments for ``torchaudio.models.wav2vec2_model`` (e.g.
        ``torchaudio.pipelines.WAVLM_BASE_PLUS._params``). Only the config
        is used: weights come from the checkpoint, nothing is downloaded.
    wav2vec_layer : int, optional
        Index of the wav2vec layer to feed the Conformer. Defaults (-1) to a
        softmax-weighted mixture of all layers with learnable weights.
    conformer : dict, optional
        Keyword arguments for ``torchaudio.models.Conformer`` (``num_heads``,
        ``ffn_dim``, ``num_layers``, ``depthwise_conv_kernel_size``, ``dropout``).
    linear : dict, optional
        ``{"hidden_size": 128, "num_layers": 2}`` by default.
    sample_rate : int, optional
        Defaults to 16 kHz. num_channels : int, optional. Defaults to mono.
    """

    CONFORMER_DEFAULTS = {
        "num_heads": 4,
        "ffn_dim": 256,
        "num_layers": 4,
        "depthwise_conv_kernel_size": 31,
        "dropout": 0.1,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        wav2vec: Optional[dict] = None,
        wav2vec_layer: int = -1,
        conformer: Optional[dict] = None,
        linear: Optional[dict] = None,
        normalize_waveform: bool = False,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if not isinstance(wav2vec, dict):
            raise TypeError(
                "`wav2vec` must be a wav2vec2_model config dict "
                "(e.g. torchaudio.pipelines.WAVLM_BASE_PLUS._params); "
                f"got {type(wav2vec).__name__}"
            )
        wav2vec = dict(wav2vec)
        # WavLM configs carry the gated relative position bias keys, which only
        # torchaudio's wavlm_model factory understands (this is what the WAVLM_*
        # bundles' get_model() calls); plain wav2vec2/HuBERT configs do not.
        if "encoder_max_distance" in wav2vec:
            backbone = torchaudio.models.wavlm_model(**wav2vec)
        else:
            backbone = torchaudio.models.wav2vec2_model(**wav2vec)
        # WavLM-Large was trained through torchaudio's normalising bundle wrapper; see
        # _NormalizedWav2Vec2. The flag rides in the checkpoint's hyper-parameters.
        self.wav2vec = _NormalizedWav2Vec2(backbone) if normalize_waveform else backbone
        wav2vec_dim = wav2vec["encoder_embed_dim"]
        wav2vec_num_layers = wav2vec["encoder_num_layers"]

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )

        conformer = merge_dict(self.CONFORMER_DEFAULTS, conformer)
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters(
            "wav2vec", "wav2vec_layer", "conformer", "linear", "normalize_waveform"
        )

        self.conformer = torchaudio.models.Conformer(
            input_dim=wav2vec_dim,
            num_heads=conformer["num_heads"],
            ffn_dim=conformer["ffn_dim"],
            num_layers=conformer["num_layers"],
            depthwise_conv_kernel_size=conformer["depthwise_conv_kernel_size"],
            dropout=conformer["dropout"],
        )
        # Conformer preserves the feature dimension
        self._head_out_features = wav2vec_dim

        if linear["num_layers"] < 1:
            return

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [wav2vec_dim] + [linear["hidden_size"]] * linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("SuplimeSegmentation does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self._head_out_features

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @property
    def _feature_extractor(self):
        """The conv frontend, one level deeper when the backbone is wrapped for normalisation."""
        w = self.wav2vec
        return w.feature_extractor if hasattr(w, "feature_extractor") else w.model.feature_extractor

    def num_frames(self, num_samples: int) -> int:
        """Number of output frames for `num_samples` input samples."""
        cache = self.__dict__.setdefault("_num_frames_cache", {})
        if num_samples in cache:
            return cache[num_samples]
        cache[num_samples] = _n = self._num_frames(num_samples)
        return _n

    def _num_frames(self, num_samples: int) -> int:
        num_frames = num_samples
        for conv_layer in self._feature_extractor.conv_layers:
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        receptive_field_size = num_frames
        for conv_layer in reversed(self._feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        receptive_field_center = frame
        for conv_layer in reversed(self._feature_extractor.conv_layers):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_center

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms : (batch, channel, sample) -> scores : (batch, frame, classes)"""

        num_layers = (
            None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        )

        # The WavLM backbone dominates inference cost and runs under fp16 autocast
        # on CUDA inside torch.inference_mode() (i.e. the Inference path, never
        # training). The published benchmark numbers were produced this way;
        # SUPLIME_FP16=0 switches it off. Everything downstream stays fp32.
        use_fp16 = (
            torch.is_inference_mode_enabled()
            and waveforms.device.type == "cuda"
            and os.environ.get("SUPLIME_FP16", "1") != "0"
        )
        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = self.wav2vec.extract_features(
                    waveforms.squeeze(1), num_layers=num_layers
                )
            outputs = [output.float() for output in outputs]
        else:
            outputs, _ = self.wav2vec.extract_features(
                waveforms.squeeze(1), num_layers=num_layers
            )

        if num_layers is None:
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.wav2vec_weights, dim=0
            )
        else:
            outputs = outputs[-1]

        lengths = torch.full((outputs.shape[0],), outputs.shape[1], device=outputs.device)
        outputs, _ = self.conformer(outputs, lengths)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
