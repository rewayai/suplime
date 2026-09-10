# RTFx benchmark: SUPlime, SUPlime-L, DiariZen and pyannote community-1 on one GPU.
#
#   docker build -t rtfx bench/rtfx
#   docker run --rm --gpus all -e HF_TOKEN=hf_... rtfx
#
# Two virtualenvs, because the systems cannot share one: SUPlime and community-1 need
# pyannote.audio 4.x with numpy 2, while DiariZen vendors a 3.1.1 fork pinned to numpy 1.26.
# Each venv brings its own CUDA runtime in the torch wheel, so the host needs only a driver.
# Torch versions are the ones each system was measured with: cu130 for pyannote 4.x (Blackwell
# and Hopper), cu128 for DiariZen (2.8 is the first release that compiles for sm_120).
# Python 3.11: DiariZen is developed on 3.10/3.11 and pins numpy 1.26, and every torch
# wheel pinned below has a cp311 build. suplime itself is happy on anything >= 3.10.
FROM python:3.11-slim

# ffmpeg: pyannote 4.x decodes audio through torchcodec, which dlopens libav* at runtime.
# git: DiariZen installs from source, it is not on PyPI.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- pyannote 4.x side
RUN python -m venv /opt/venv/suplime && \
    /opt/venv/suplime/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/suplime/bin/pip install --no-cache-dir \
        torch==2.14.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
    /opt/venv/suplime/bin/pip install --no-cache-dir suplime==0.2.0

# ---------------------------------------------------------------- DiariZen side
ARG DIARIZEN_COMMIT=844f555
RUN python3 -m venv /opt/venv/diarizen && \
    /opt/venv/diarizen/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/diarizen/bin/pip install --no-cache-dir \
        torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
    git clone https://github.com/BUTSpeechFIT/DiariZen /opt/DiariZen && \
    cd /opt/DiariZen && git checkout ${DIARIZEN_COMMIT} && \
    /opt/venv/diarizen/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/diarizen/bin/pip install --no-cache-dir -e . && \
    cd pyannote-audio && \
    /opt/venv/diarizen/bin/pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip

# Models are downloaded on first run into this cache; mount it to keep them across runs.
ENV HF_HOME=/cache/hf
VOLUME /cache/hf

WORKDIR /bench
COPY rtfx.py bench_one.py prepare_clips.py ./
COPY clips ./clips

# `docker run rtfx` benchmarks everything; append arguments to narrow it, e.g.
#   docker run --rm --gpus all rtfx --systems suplime suplime-large --repeats 5
ENTRYPOINT ["/opt/venv/suplime/bin/python", "/bench/rtfx.py"]
