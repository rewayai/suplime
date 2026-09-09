#!/usr/bin/env bash
# One corpus of the suplime verification benchmark on one GPU. Env from run_benchmark.sh:
#   C (corpus), MODEL (hf dir / repo), OUT (results dir), EXTRA (extra benchmark.py args)
exec bash -l -c '
set -eo pipefail
source /mnt/shared/jose/Github/pyannote-audio/reway/slurm/common.sh
diar_require_gpu                      # requeues (and flags the node) if cuInit fails
export HF_HOME=/mnt/shared/envs/hf_cache
export HF_HUB_OFFLINE=1               # nothing may be fetched on nodes; the model is a local dir
ENV=/mnt/shared/envs/root/envs/suplime
export LD_LIBRARY_PATH=$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # conda ffmpeg libs for torchcodec
PY=$ENV/bin/python
echo "[suplime-$C] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
$PY -c "import pyannote.audio, suplime, torch; print(\"[suplime-$C]\", pyannote.audio.__file__, suplime.__version__, torch.__version__, torch.cuda.get_device_name(0))"
$PY /mnt/shared/jose/Github/suplime/tools/benchmark.py --model "$MODEL" --corpus "$C" --out "$OUT/$C" $EXTRA
'
