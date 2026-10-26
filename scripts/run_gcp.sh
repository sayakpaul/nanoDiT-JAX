#!/bin/bash
# Run inside the provisioned TPU VM; output and checkpoints stay on its disk.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="$PWD/.venv/bin/python"
OUTPUT="$PWD/runs/full"
mkdir -p "$OUTPUT"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export MPLCONFIGDIR="$PWD/runs/matplotlib"
if [ -f "$OUTPUT/COMPLETE" ]; then
  echo 'Training already completed.'
  exit 0
fi

"$PYTHON" -u -c "from huggingface_hub import snapshot_download; snapshot_download('sayakpaul/butteflies_with_classes', repo_type='dataset', local_dir='butterflies', max_workers=32); print('Dataset ready', flush=True)" &
DOWNLOAD_PID=$!
if [ ! -f "$OUTPUT/benchmark.json" ]; then
  "$PYTHON" -u scripts/benchmark.py --output "$OUTPUT/benchmark.json"
fi
wait "$DOWNLOAD_PID"

mapfile -t CONFIG < <("$PYTHON" -c "import json; c=json.load(open('runs/full/benchmark.json'))['best']; print(c['microbatch_size']); print('--remat' if c['remat'] else '--no-remat')")
RESUME=()
if [ -f "$OUTPUT/latest.msgpack" ]; then
  RESUME=(--resume "$OUTPUT/latest.msgpack")
fi
"$PYTHON" -u train.py --require-tpu --epochs 600 --batch-size 128 \
  --image-size 64 --patch-size 2 --hidden-size 768 --depth 12 --num-heads 12 --num-classes 75 \
  --microbatch-size "${CONFIG[0]}" "${CONFIG[1]}" --learning-rate 1e-4 \
  --warmup-steps 0 --weight-decay 0.01 --log-every 50 \
  --sample-every 500 --checkpoint-every 1250 --ode-steps 150 --cfg-scale 5 \
  --sample-ema --ema-decay 0.999 --output-dir "$OUTPUT" "${RESUME[@]}" "$@"
touch "$OUTPUT/COMPLETE"
sync
# Leave time to retrieve the final checkpoint, then release compute. Disk persists.
if [ "${NANODIT_POWEROFF:-0}" = 1 ]; then
  shutdown -h +15
fi
