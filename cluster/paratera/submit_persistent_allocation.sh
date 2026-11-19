#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: bash $0 <a100|h100>" >&2
    exit 64
}

PLATFORM="${1:-}"
case "${PLATFORM}" in
    a100)
        PARTITION="a100x"
        GPU_TYPE="a100"
        MEMORY_PER_GPU="120G"
        ;;
    h100)
        PARTITION="h100x"
        GPU_TYPE="h100"
        MEMORY_PER_GPU="240G"
        ;;
    *)
        usage
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RAGLEASE_ROOT="${RAGLEASE_ROOT:-${HOME}/Performance01/dyh/raglease}"
CPUS="${RAGLEASE_CPUS_PER_GPU:-12}"
TIME_LIMIT="${RAGLEASE_ALLOCATION_TIME:-08:00:00}"
ALLOCATION_KEY="${RAGLEASE_ALLOCATION_KEY:-}"
ALLOCATION_DEPENDENCY="${RAGLEASE_ALLOCATION_DEPENDENCY:-}"
DEPENDENCY_ARGS=()
if [[ -n "${ALLOCATION_DEPENDENCY}" ]]; then
    if [[ ! "${ALLOCATION_DEPENDENCY}" =~ ^afterany:[0-9]+$ || -z "${ALLOCATION_KEY}" ]]; then
        echo "A sequential dependency requires afterany:JOB_ID and a unique allocation key." >&2
        exit 64
    fi
    DEPENDENCY_ARGS=(--dependency="${ALLOCATION_DEPENDENCY}")
fi
if [[ -n "${ALLOCATION_KEY}" && ! "${ALLOCATION_KEY}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RAGLEASE_ALLOCATION_KEY must contain only letters, digits, '_' or '-'." >&2
    exit 64
fi
JOB_NAME="raglease-${PLATFORM}-persistent${ALLOCATION_KEY:+-${ALLOCATION_KEY}}"
LOG_DIR="${RAGLEASE_ROOT}/logs"
STATE_DIR="${RAGLEASE_ROOT}/run/allocations"
LATEST_FILE="${STATE_DIR}/latest-${PLATFORM}${ALLOCATION_KEY:+-${ALLOCATION_KEY}}.jobid"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

existing_job="$({
    squeue -h -u "${USER}" -n "${JOB_NAME}" -o '%A' 2>/dev/null || true
} | head -n 1)"
if [[ -n "${existing_job}" ]]; then
    printf '%s\n' "${existing_job}" > "${LATEST_FILE}"
    echo "Reusing ${PLATFORM} allocation job ${existing_job}." >&2
    printf '%s\n' "${existing_job}"
    exit 0
fi

submit_output="$(
    sbatch --parsable \
        "${DEPENDENCY_ARGS[@]}" \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --gres="gpu:${GPU_TYPE}:1" \
        --cpus-per-task="${CPUS}" \
        --mem-per-gpu="${MEMORY_PER_GPU}" \
        --time="${TIME_LIMIT}" \
        --export="ALL,RAGLEASE_ROOT=${RAGLEASE_ROOT},RAGLEASE_PLATFORM=${PLATFORM}" \
        --output="${LOG_DIR}/${JOB_NAME}.%j.out" \
        --error="${LOG_DIR}/${JOB_NAME}.%j.err" \
        "${SCRIPT_DIR}/hold_gpu_allocation.sh"
)"

job_id="${submit_output%%;*}"
printf '%s\n' "${job_id}" > "${LATEST_FILE}"
echo "Submitted ${PLATFORM} persistent allocation job ${job_id}." >&2
printf '%s\n' "${job_id}"
