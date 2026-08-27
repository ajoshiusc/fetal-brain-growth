#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
MODEL_DIR="${PROJECT_DIR}/models"
MODEL_PATH="${MODEL_DIR}/KISPI-all_fss.ckpt"
SOURCE_DIR="${PROJECT_DIR}/third_party/FetalSynthSeg"
FSS_REPO="https://github.com/Medical-Image-Analysis-Laboratory/FetalSynthSeg.git"
FSS_COMMIT="03c439edef02fc830e31a38169c5aa09ca98eeb4"
MODEL_ID="1zCHN8OS2gSmVPF_S408BOqDxFbZqEHKb"
MODEL_SHA256="d991edbb1295797f59973867ec3724ba6ebaddb44387f09da38bc5ab98b2fe52"

if [[ "${1:-}" != "--accept-license" ]]; then
  echo "FetalSynthSeg is public source, not public domain."
  echo "It is licensed for academic, non-commercial research."
  echo "Read: https://github.com/Medical-Image-Analysis-Laboratory/FetalSynthSeg/blob/main/LICENSE"
  echo "Rerun with --accept-license only if those terms apply and you accept them."
  exit 2
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --editable "${PROJECT_DIR}[all]"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 "${FSS_REPO}" "${SOURCE_DIR}"
fi
git -C "${SOURCE_DIR}" fetch --depth 1 origin "${FSS_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${FSS_COMMIT}"

mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  curl -L --fail --show-error \
    "https://drive.usercontent.google.com/download?id=${MODEL_ID}&export=download&confirm=t" \
    -o "${MODEL_PATH}"
fi
echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum --check

echo "Installed: ${VENV_DIR}/bin/fbg"
echo "Checkpoint: ${MODEL_PATH}"
echo "Pinned FetalSynthSeg source: ${SOURCE_DIR} @ ${FSS_COMMIT}"
