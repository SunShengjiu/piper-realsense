#!/usr/bin/env python3
"""Install the published AgileX/Intel nominal mount transform as an explicit, unverified calibration record."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
source = root / "configs" / "official_piper_realsense_mid_stand_nominal.json"
dest = root / "dataset" / "calibrations" / "handeye" / "handeye-official-piper-realsense-mid-stand-nominal.json"
dest.parent.mkdir(parents=True, exist_ok=True)
payload = json.loads(source.read_text(encoding="utf-8"))
dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(dest)
print("status:", payload["status"], "valid:", payload["valid"])
