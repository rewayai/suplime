#!/usr/bin/env bash
# Verify a published SUPlime pipeline on several corpora IN PARALLEL on one whole-machine slot:
# one tools/benchmark.py process per GPU, pinned with CUDA_VISIBLE_DEVICES. Written because the
# fleet often has a single free 1-GPU chunk but a whole 8-GPU machine idle (2026-09-09).
# Env: MODEL (hf dir or repo id), OUT, LANES (";"-separated, each a comma-separated corpus list),
# EXTRA (extra benchmark.py args, e.g. "--threshold 0.80" — forwarded exactly as benchmark.sbatch does).
exec bash -l -c '
set -eo pipefail
source /mnt/shared/jose/Github/pyannote-audio/reway/slurm/common.sh
diar_require_gpu
ENV=/mnt/shared/envs/root/envs/suplime
export LD_LIBRARY_PATH=$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # conda ffmpeg for torchcodec
export HF_HOME=/mnt/shared/envs/hf_cache HF_HUB_OFFLINE=1
R=/mnt/shared/jose/Github/suplime; mkdir -p "$OUT/logs"
NLANES=$(awk -F";" "{print NF}" <<< "$LANES")
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NLANES" -gt "$NGPU" ]; then
  echo "[bench] $NLANES lanes but only $NGPU GPUs; lanes past the last GPU would get an "\
       "invalid CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
THREADS=$(( $(nproc) / NLANES )); [ "$THREADS" -lt 2 ] && THREADS=2
echo "[bench] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) lanes=$NLANES threads/lane=$THREADS model=$MODEL"
gpu=0; pids=""
IFS=";" read -ra LANE_ARR <<< "$LANES"
for lane in "${LANE_ARR[@]}"; do
  (
    lane_rc=0
    export CUDA_VISIBLE_DEVICES=$gpu
    # one torch process per lane: cap the intra-op pool or 8 lanes oversubscribe the machine
    export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
    export NUMEXPR_NUM_THREADS=$THREADS TORCH_NUM_THREADS=$THREADS
    IFS="," read -ra CS <<< "$lane"
    for C in "${CS[@]}"; do
      echo "[bench] gpu $CUDA_VISIBLE_DEVICES -> $C"
      if ! $ENV/bin/python "$R/tools/benchmark.py" --model "$MODEL" --corpus "$C" --out "$OUT/$C" \
             ${EXTRA:+$EXTRA} > "$OUT/logs/$C.out" 2>&1; then
        echo "[bench] FAILED: $C (gpu $CUDA_VISIBLE_DEVICES)"
        lane_rc=1
      fi
    done
    # the subshell status is the status of its LAST command, so this exit is what makes a
    # failed corpus reach `wait` below; without it every lane exited 0 and rc stayed 0
    exit $lane_rc
  ) &
  pids="$pids $!"; gpu=$((gpu + 1))
done
rc=0; for p in $pids; do wait $p || rc=1; done
echo "[bench] all lanes finished (rc=$rc)"; exit $rc
'
