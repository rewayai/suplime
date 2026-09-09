#!/usr/bin/env bash
# Submit the suplime verification benchmark: one 1-GPU slurm job per corpus.
#   tools/run_benchmark.sh [corpus ...] [-- extra benchmark.py args]
# Defaults to the 8 pyannote-benchmark corpora. MODEL / OUT can be overridden via env.
set -euo pipefail
R=/mnt/shared/jose/Github/suplime
MODEL="${MODEL:-$R/hf}"
OUT="${OUT:-$R/results}"
CORPORA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do CORPORA+=("$1"); shift; done
[ $# -gt 0 ] && shift
EXTRA="$*"
[ ${#CORPORA[@]} -eq 0 ] && CORPORA=(AISHELL4 AliMeeting_far AMI_IHM AMI_SDM AVA_AVD MSDWild RAMC VoxConverse)
mkdir -p "$OUT/logs"
source /mnt/shared/jose/Github/pyannote-audio/reway/slurm/common.sh   # DIAR_EXCLUDE_NODES
export MODEL OUT EXTRA
for C in "${CORPORA[@]}"; do
  export C
  sbatch --parsable --job-name="suplime-$C" --gres=gpu:1 --constraint="vram24|vram32|vram40|vram48" \
         --exclude="$DIAR_EXCLUDE_NODES" --time=06:00:00 --requeue \
         -o "$OUT/logs/$C-%j.out" --export=ALL \
         "$R/tools/benchmark.sbatch" | sed "s/^/  suplime-$C: job /"
done
