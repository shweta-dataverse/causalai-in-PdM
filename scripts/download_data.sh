#!/usr/bin/env bash
# Downloads the causRCA dataset into data/causrca/raw/.
# Safe to re-run: it skips the download if the data is already there.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${PROJECT_ROOT}/data/causrca/raw"
ARCHIVE_URL="https://zenodo.org/api/records/15876410/files/dataset_causRCA.zip/content"

if [[ -d "${TARGET}/expert_graph" ]]; then
    echo "causRCA is already in data/causrca/raw/ - nothing to do."
    exit 0
fi

echo "Downloading causRCA (about 3.6 MB)..."
mkdir -p "${TARGET}"
curl -fSL --retry 3 -o "${TARGET}/causRCA.zip" "${ARCHIVE_URL}"

echo "Unpacking..."
unzip -qo "${TARGET}/causRCA.zip" -d "${TARGET}"
rm -f "${TARGET}/causRCA.zip"

echo "Done. Now run:  python scripts/explore_data.py"
