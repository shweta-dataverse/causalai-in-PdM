#!/usr/bin/env bash
# =============================================================================
# download_data.sh
#
# Downloads the three candidate datasets for the thesis dataset review into
# the data/ directory. Idempotent: files that already exist are skipped.
#
# Datasets
#   1. causRCA  - CNC vertical lathe, causal RCA benchmark (Zenodo, Apache-2.0)
#   2. C-MAPSS  - NASA turbofan degradation simulation (NASA PCoE, public)
#   3. SCANIA   - Component X truck fleet dataset (Research Data Sweden, CC BY 4.0)
#
# Usage
#   bash scripts/download_data.sh            # download everything
#   bash scripts/download_data.sh causrca    # download a single dataset
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

CAUSRCA_URL="https://zenodo.org/api/records/15876410/files/dataset_causRCA.zip/content"
CMAPSS_URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SCANIA_API="https://api.researchdata.se/dataset/2024-34/3/file"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

fetch() {
    # fetch <url> <destination>: resumable curl download, skip if present.
    local url="$1" dest="$2"
    if [[ -s "${dest}" ]]; then
        log "SKIP  $(basename "${dest}") (already exists)"
        return 0
    fi
    log "GET   $(basename "${dest}")"
    curl -fSL --retry 3 --retry-delay 5 -C - -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
}

download_causrca() {
    local dir="${DATA_DIR}/causrca/raw"
    mkdir -p "${dir}"
    fetch "${CAUSRCA_URL}" "${dir}/dataset_causRCA.zip"
    if [[ ! -d "${dir}/dataset_causRCA" ]]; then
        log "UNZIP dataset_causRCA.zip"
        unzip -qo "${dir}/dataset_causRCA.zip" -d "${dir}"
    fi
    log "DONE  causRCA"
}

download_cmapss() {
    local dir="${DATA_DIR}/cmapss/raw"
    mkdir -p "${dir}"
    fetch "${CMAPSS_URL}" "${dir}/CMAPSS.zip"
    if [[ ! -f "${dir}/train_FD001.txt" ]]; then
        log "UNZIP CMAPSS.zip"
        unzip -qo "${dir}/CMAPSS.zip" -d "${dir}"
        # The NASA archive nests the actual data inside a second zip.
        local inner
        inner="$(find "${dir}" -name 'CMAPSSData.zip' | head -1 || true)"
        if [[ -n "${inner}" ]]; then
            unzip -qo "${inner}" -d "${dir}"
        fi
    fi
    log "DONE  C-MAPSS"
}

download_scania() {
    local dir="${DATA_DIR}/scania/raw"
    mkdir -p "${dir}"
    local data_files=(
        train_operational_readouts.csv
        train_specifications.csv
        train_tte.csv
        validation_operational_readouts.csv
        validation_specifications.csv
        validation_labels.csv
        test_operational_readouts.csv
        test_specifications.csv
        test_labels.csv
    )
    local doc_files=(
        Scania_Component_X.pdf
        2024_IDA_challenge_v2.pdf
    )
    for f in "${data_files[@]}"; do
        fetch "${SCANIA_API}/data?filePath=${f}" "${dir}/${f}"
    done
    for f in "${doc_files[@]}"; do
        fetch "${SCANIA_API}/documentation?filePath=${f}" "${dir}/${f}"
    done
    log "DONE  SCANIA"
}

main() {
    local target="${1:-all}"
    case "${target}" in
        causrca) download_causrca ;;
        cmapss)  download_cmapss ;;
        scania)  download_scania ;;
        all)
            download_causrca
            download_cmapss
            download_scania
            ;;
        *)
            echo "Unknown dataset '${target}'. Use: causrca | cmapss | scania | all" >&2
            exit 1
            ;;
    esac
    log "All requested downloads complete."
}

main "$@"
