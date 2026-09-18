#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
config=configs/handeye_checkerboard_eye_in_hand.json
session="${1:-he-checker-$(date +%Y%m%d-%H%M%S)}"
if [ "$#" -gt 0 ]; then shift; fi
python3 -m piper_capture.cli --config "$config" handeye sample --session "$session" "$@"
python3 -m piper_capture.cli --config "$config" handeye solve --session "$session"
