"""Suplime: speaker diarization models loadable through pyannote.audio 4.x.

The classes here are what the checkpoints on https://huggingface.co/rewayai/suplime
point at (``checkpoint["pyannote.audio"]["architecture"]`` and ``config.yaml``'s
``pipeline.name``), so ``pip install suplime`` is all that is needed for
``Pipeline.from_pretrained("rewayai/suplime")`` to work.
"""

from importlib.metadata import PackageNotFoundError, version

from .pipeline import SuplimeDiarization

try:
    __version__ = version("suplime")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = ["SuplimeDiarization", "__version__"]
