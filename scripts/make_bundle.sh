#!/usr/bin/env bash
# Build the offline bundle for a machine with no internet (the hospital workstation).
# GitHub gets you to your laptop; this gets you the last hop.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/scte_bundle}
rm -rf "$OUT"; mkdir -p "$OUT/env"
echo "== code + shipped weights"
tar czf "$OUT/scte_code.tgz" --exclude='__pycache__' \
    scte_r scripts configs baselines public*.pt runs/protocol_rplhr_5mm.json \
    PRIVATE_RUNBOOK.md STAGE1_执行报告.md 2>/dev/null || true
echo "== TotalSegmentator weights (the single most common on-site blocker)"
if [ -d "$HOME/.totalsegmentator" ]; then
  tar czf "$OUT/totalsegmentator_weights.tgz" -C "$HOME" .totalsegmentator
else
  echo "!! ~/.totalsegmentator not found - run TotalSegmentator once here first,"
  echo "   otherwise it will try to download weights on a machine with no internet."
fi
echo "== python wheels"
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu128 -d "$OUT/env" -q || \
  echo "!! wheel download failed - do this on a machine with internet"
pip download numpy scipy scikit-image SimpleITK pydicom PyYAML einops timm \
    TotalSegmentator nnunetv2 -d "$OUT/env" -q || true
cat > "$OUT/INSTALL.txt" <<'TXT'
On the offline machine:

  tar xzf scte_code.tgz
  tar xzf totalsegmentator_weights.tgz -C ~          # -> ~/.totalsegmentator
  pip install --no-index --find-links env/ torch torchvision numpy scipy \
      scikit-image SimpleITK pydicom PyYAML einops timm
  pip install --no-index --find-links env/ TotalSegmentator

Verify before touching any patient data:
  python -c "import torch, torchvision, totalsegmentator; print('ok')"
  # torchvision MUST come from the same CUDA channel as torch, or TotalSegmentator
  # dies with "torchvision::nms does not exist".

Then:
  bash scripts/run_private.sh dryrun /data/hebei hebei        # 15 min, must pass
  EXT_CAL=50 bash scripts/run_private.sh full /data/hebei hebei /data/henan
TXT
du -sh "$OUT"/* 2>/dev/null
echo "bundle ready: $OUT"
