#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
config=configs/handeye_official_eye_in_hand.json
session="${1:-he-$(date +%Y%m%d-%H%M%S)}"
python3 -m piper_capture.cli --config "$config" handeye sample --session "$session"
python3 -m piper_capture.cli --config "$config" handeye solve --session "$session"
