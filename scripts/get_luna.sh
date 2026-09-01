#!/usr/bin/env bash
# LUNA16 (LIDC-IDRI subset): 888 CT scans with 1186 radiologist-annotated nodules >=3 mm.
# This is the only public route to a REAL lesion-level detection number - every result
# so far counts connected components, which are mostly vessel cross-sections.
set -uo pipefail
cd /home/prinlab/SCTE/DATA/luna16
Z=https://zenodo.org/records/3723295/files
for f in annotations.csv candidates_V2.zip seg-lungs-LUNA16.zip; do
  [ -e "$f" ] || curl -sL -o "$f" "$Z/$f?download=1" && echo "got $f"
done
for s in "$@"; do
  [ -d "subset$s" ] && [ "$(ls subset$s/*.mhd 2>/dev/null | wc -l)" -gt 50 ] && \
      { echo "subset$s present"; continue; }
  # Resume and verify: a plain curl can return a 1 MB error page and the pipeline then
  # fails in unzip with a message that says nothing about the download. Retry on a
  # partial file (-C -), then check the archive before trusting it.
  for try in 1 2 3; do
    curl -sL -C - --retry 5 --retry-delay 10 -o "subset$s.zip" "$Z/subset$s.zip?download=1"
    sz=$(du -m "subset$s.zip" 2>/dev/null | cut -f1)
    if [ "${sz:-0}" -gt 4000 ] && unzip -tq "subset$s.zip" >/dev/null 2>&1; then break; fi
    echo "subset$s attempt $try incomplete (${sz:-0} MB), retrying"; sleep 15
  done
  if unzip -tq "subset$s.zip" >/dev/null 2>&1; then
    unzip -q -o "subset$s.zip" && rm -f "subset$s.zip" \
      && echo "subset$s ready: $(ls subset$s/*.mhd | wc -l) scans"
  else
    echo "!! subset$s download failed after 3 attempts"; rm -f "subset$s.zip"
  fi
done
