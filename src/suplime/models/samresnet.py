# Copyright (c) 2024 XiaoyiQin, Yuke Lin (linyuke0609@gmail.com)
#               2024 Shuai Wang (wsstriving@gmail.com)
#               2026 Re:WayAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SimAM-ResNet34 speaker embedding backbone, ported from WeSpeaker.

Weight-compatible with WeSpeaker's ``SimAM_ResNet34_ASP`` checkpoints
(e.g. ``voxblink2_samresnet34_ft`` a.k.a. the python package's ``vblinkf``):
module names (``front``, ``pooling.attention``, ``bottleneck``) and parameter
shapes match, so a WeSpeaker ``avg_model.pt`` loads with ``strict=True`` once
the classification head (``projection.*``) is dropped.

Differences from the WeSpeaker original:
- the attentive statistics pooling accepts optional per-frame ``weights``
  (masked, renormalized softmax) so the diarization pipeline can pool
  per-speaker embeddings from overlapping speech, mirroring ``StatsPool``;
- ``num_frames`` / ``receptive_field_*`` helpers mirror ``resnet.py`` so the
  model integrates with pyannote's receptive-field machinery.
"""

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SimAMBasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        return multi_conv_num_frames(
            num_samples,
            kernel_size=[3, 3],
            stride=[self.stride, 1],
            padding=[1, 1],
            dilation=[1, 1],
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=[3, 3],
            stride=[self.stride, 1],
            padding=[1, 1],
            dilation=[1, 1],
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        return multi_conv_receptive_field_center(
            frame,
            kernel_size=[3, 3],
            stride=[self.stride, 1],
            padding=[1, 1],
            dilation=[1, 1],
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class SimAMResNetFront(nn.Module):
    """The convolutional front-end of WeSpeaker's SimAM ResNet."""

    def __init__(self, in_planes, block, num_blocks, in_ch=1):
        super(SimAMResNetFront, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        num_frames = conv1d_num_frames(
            num_samples, kernel_size=3, stride=1, padding=1, dilation=1
        )
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                num_frames = layer.num_frames(num_frames)
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        receptive_field_size = num_frames
        for layers in reversed([self.layer1, self.layer2, self.layer3, self.layer4]):
            for layer in reversed(layers):
                receptive_field_size = layer.receptive_field_size(receptive_field_size)
        return conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        receptive_field_center = frame
        for layers in reversed([self.layer1, self.layer2, self.layer3, self.layer4]):
            for layer in reversed(layers):
                receptive_field_center = layer.receptive_field_center(
                    frame=receptive_field_center
                )
        return conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ASP(nn.Module):
    """Attentive statistics pooling with optional per-frame weights.

    Same parameters (and state-dict keys ``attention.{0,2,3}.*``) as
    WeSpeaker's ``ASP``: the final ``Softmax`` module carried no parameters,
    so it is applied functionally here to make room for masking.
    """

    def __init__(self, in_planes: int, acoustic_dim: int):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.feature_dim = in_planes * 8 * outmap_size
        self.out_dim = self.feature_dim * 2

        self.attention = nn.Sequential(
            nn.Conv1d(self.feature_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, self.feature_dim, kernel_size=1),
        )

    def _pool(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x: (batch, features, frames), w: (batch, features, frames) normalized"""
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        return torch.cat([mu, sg], dim=1)

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : (batch, channel, freq, frames) torch.Tensor
            Output of the convolutional front-end.
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Per-frame weights, linearly interpolated to the pooled resolution
            when needed. The attention distribution is multiplied by the
            weights and renormalized, so ``weights=None`` reproduces
            WeSpeaker's unmasked attentive pooling exactly.

        Returns
        -------
        stats : (batch, 2 * features) or (batch, speakers, 2 * features)
        """
        x = x.reshape(x.size(0), -1, x.size(-1))
        # (batch, features, frames)

        w = F.softmax(self.attention(x), dim=2)

        if weights is None:
            return self._pool(x, w)

        if weights.dim() == 2:
            has_speaker_dimension = False
            weights = weights.unsqueeze(dim=1)
        else:
            has_speaker_dimension = True

        num_frames = x.size(-1)
        if num_frames != weights.size(-1):
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        output = []
        for speaker in range(weights.size(1)):
            masked = w * weights[:, speaker, :].unsqueeze(dim=1)
            masked = masked / (masked.sum(dim=2, keepdim=True) + 1e-8)
            output.append(self._pool(x, masked))
        output = torch.stack(output, dim=1)

        if not has_speaker_dimension:
            return output.squeeze(dim=1)

        return output


class SimAMResNet(nn.Module):
    """WeSpeaker ``SimAM_ResNet*_ASP``, with the same interface as ``resnet.ResNet``."""

    def __init__(
        self,
        num_blocks,
        in_planes: int = 64,
        embed_dim: int = 256,
        acoustic_dim: int = 80,
    ):
        super(SimAMResNet, self).__init__()
        self.embed_dim = embed_dim

        self.front = SimAMResNetFront(in_planes, SimAMBasicBlock, num_blocks)
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        return self.front.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return self.front.receptive_field_size(num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        return self.front.receptive_field_center(frame=frame)

    def forward_frames(self, fbank: torch.Tensor) -> torch.Tensor:
        """Extract frame-wise embeddings

        Parameters
        ----------
        fbank : (batch, frames, features) torch.Tensor
            Batch of fbank features

        Returns
        -------
        embeddings : (batch, channel, freq, embedding_frames) torch.Tensor
        """
        fbank = fbank.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        fbank = fbank.unsqueeze_(1)
        return self.front(fbank)

    def forward_embedding(
        self, frames: torch.Tensor, weights: Optional[torch.Tensor] = None
    ):
        """Extract speaker embeddings from frame-wise embeddings

        Parameters
        ----------
        frames : (batch, channel, freq, embedding_frames) torch.Tensor
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
        """
        stats = self.pooling(frames, weights=weights)
        return torch.tensor(0.0), self.bottleneck(stats)

    def forward(self, fbank: torch.Tensor, weights: Optional[torch.Tensor] = None):
        """Extract speaker embeddings

        Parameters
        ----------
        fbank : (batch, frames, features) torch.Tensor
            Batch of features
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights

        Returns
        -------
        embedding : (batch, embedding_dim) torch.Tensor
        """
        return self.forward_embedding(self.forward_frames(fbank), weights=weights)


def SimAMResNet34(feat_dim, embed_dim, in_planes: int = 64):
    return SimAMResNet(
        [3, 4, 6, 3], in_planes=in_planes, embed_dim=embed_dim, acoustic_dim=feat_dim
    )


def SimAMResNet100(feat_dim, embed_dim, in_planes: int = 64):
    return SimAMResNet(
        [6, 16, 24, 3], in_planes=in_planes, embed_dim=embed_dim, acoustic_dim=feat_dim
    )
