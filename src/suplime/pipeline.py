# MIT License
#
# Copyright (c) 2026 Re:WayAI
# Copyright (c) 2021- CNRS (constructor derived from pyannote.audio's SpeakerDiarization)
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

"""Suplime speaker diarization pipeline.

``pyannote.audio.pipelines.SpeakerDiarization`` with two differences:

* it never loads a PLDA. Upstream 4.x fetches one unconditionally from the gated
  ``pyannote/speaker-diarization-community-1`` repo even when clustering is
  agglomerative, which forces a Hugging Face token on every user; suplime ships
  agglomerative clustering only.
* its defaults are the published ``rewayai/suplime`` models and hyper-parameters,
  so ``SuplimeDiarization()`` alone is the benchmarked system.

Everything after construction (``apply``, embedding extraction, clustering,
reconstruction) is inherited unchanged.
"""

from pathlib import Path
from typing import Optional, Text, Union

from pyannote.audio import Audio, Inference, Model
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.pipeline.parameter import ParamDict, Uniform

HF_REPO = "rewayai/suplime"


class SuplimeDiarization(SpeakerDiarization):
    """Speaker diarization with the suplime segmentation + embedding models.

    Parameters
    ----------
    segmentation, embedding : Model, str or dict, optional
        Pretrained models; default to the ``segmentation`` / ``embedding``
        subfolders of ``rewayai/suplime`` on the Hugging Face Hub.
    segmentation_step : float, optional
        Sliding-window step as a fraction of the window duration. Defaults to 0.1.
    embedding_exclude_overlap : bool, optional
        Exclude overlapping speech from embedding extraction. Defaults to True.
    clustering : {"AgglomerativeClustering", "KMeansClustering", "OracleClustering"}
        Defaults to "AgglomerativeClustering". VBx is not available (no PLDA).
    embedding_batch_size, segmentation_batch_size : int, optional
        Default to 32.
    der_variant : dict, optional
        Diarization error rate variant used for optimization. Defaults to
        ``{"collar": 0.0, "skip_overlap": False}``.
    token : str, optional
        Hugging Face token (only needed for private/gated repos).
    cache_dir : Path or str, optional
        Hugging Face cache directory.

    Hyper-parameters (``instantiate``)
    ----------------------------------
    segmentation.min_duration_off, clustering.method, clustering.min_cluster_size,
    clustering.threshold — see ``default_parameters()`` for the published values.
    """

    def __init__(
        self,
        segmentation: Optional[PipelineModel] = None,
        segmentation_step: float = 0.1,
        embedding: Optional[PipelineModel] = None,
        embedding_exclude_overlap: bool = True,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 32,
        segmentation_batch_size: int = 32,
        der_variant: Optional[dict] = None,
        token: Union[Text, None] = None,
        cache_dir: Union[Path, Text, None] = None,
    ):
        # skip SpeakerDiarization.__init__ (it loads a PLDA unconditionally)
        super(SpeakerDiarization, self).__init__()

        if clustering == "VBxClustering":
            raise ValueError(
                "SuplimeDiarization ships no PLDA, so VBxClustering is not available; "
                "use AgglomerativeClustering (default), KMeansClustering or OracleClustering."
            )

        if segmentation is None:
            segmentation = {"checkpoint": HF_REPO, "subfolder": "segmentation"}
        if embedding is None:
            embedding = {"checkpoint": HF_REPO, "subfolder": "embedding"}

        self.legacy = False

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, token=token, cache_dir=cache_dir)

        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.plda = None
        self._plda = None

        self.klustering = clustering

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        segmentation_duration = model.specifications.duration
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
        )

        if self._segmentation.model.specifications.powerset:
            self.segmentation = ParamDict(min_duration_off=Uniform(0.0, 1.0))
        else:
            self.segmentation = ParamDict(
                threshold=Uniform(0.1, 0.9),
                min_duration_off=Uniform(0.0, 1.0),
            )

        # pyannote 3.x skipped the embedding model for OracleClustering; 4.0.7's
        # SpeakerDiarization.apply calls get_embeddings unconditionally and dereferences
        # self._embedding / self._audio, so skipping them here made OracleClustering raise
        # AttributeError on the first call. Always build them.
        self._embedding = PretrainedSpeakerEmbedding(
            self.embedding, token=token, cache_dir=cache_dir
        )
        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")
        metric = "not_applicable" if self.klustering == "OracleClustering" else self._embedding.metric

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f"clustering must be one of [{', '.join(list(Clustering.__members__))}]"
            )
        self.clustering = Klustering.value(metric=metric)

        self._expects_num_speakers = self.clustering.expects_num_clusters

    def default_parameters(self):
        """Hyper-parameters the published benchmark numbers were produced with.

        Only defined for the default AgglomerativeClustering: the other clustering modes
        take different parameters entirely (KMeans and Oracle define no ``method`` or
        ``threshold``), so returning these would hand ``instantiate`` a dict it must reject.
        """
        if self.klustering != "AgglomerativeClustering":
            raise NotImplementedError(
                f"default_parameters() describes the published AgglomerativeClustering setup; "
                f"{self.klustering} takes different parameters. Instantiate it explicitly, e.g. "
                f'pipeline.instantiate({{"segmentation": {{"min_duration_off": 0.0}}, '
                f'"clustering": {{...}}}}).'
            )
        return {
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.72},
        }
