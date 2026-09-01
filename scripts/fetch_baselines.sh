#!/usr/bin/env bash
# I3Net ships without a LICENSE file, so it is fetched at setup time instead of
# being redistributed inside this repository. TVSRN and CTHNet are Apache-2.0 and
# ARE vendored (with their LICENSE files) so the comparison stays reproducible
# even offline. Run this once, on a machine with internet, before travelling.
set -euo pipefail
cd "$(dirname "$0")/.."
D=baselines/i3net
if [ -f "$D/basic_model.py" ]; then echo "I3Net already present"; exit 0; fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "fetching I3Net (Song et al., IEEE TMI 2024) ..."
# Retry, and verify the file we need actually arrived. A half-finished clone on a flaky
# hospital link otherwise fails later, in an import, with a message that says nothing
# about the network.
ok=0
for try in 1 2 3; do
  rm -rf "$TMP/I3Net"
  if git clone --depth 1 -q https://github.com/eeeric-code/I3Net "$TMP/I3Net" \
     && [ -f "$TMP/I3Net/model_zoo/i3net/basic_model.py" ]; then ok=1; break; fi
  echo "  attempt $try failed, retrying in 10s"; sleep 10
done
[ "$ok" = 1 ] || { echo "!! could not fetch I3Net. Everything else works without it;"
                   echo "   only the I3Net baseline column needs this."; exit 1; }
mkdir -p "$D"
cp "$TMP/I3Net/model_zoo/i3net/basic_model.py" \
   "$TMP/I3Net/model_zoo/i3net/dct_util.py" \
   "$TMP/I3Net/model_zoo/i3net/utils_win.py" \
   "$TMP/I3Net/opt/i3net.json" "$D/"
touch "$D/__init__.py"
echo "I3Net installed into $D (not redistributed; see baselines/README.md)"
