#!/usr/bin/env bash

set -euo pipefail

RAGLEASE_ROOT="${RAGLEASE_ROOT:-${HOME}/Performance01/dyh/raglease}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
JOB_ID="${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
STATE_DIR="${RAGLEASE_ROOT}/run/allocations/${JOB_ID}"

mkdir -p "${STATE_DIR}"

cleanup() {
    rm -f "${STATE_DIR}/READY"
    date -Iseconds > "${STATE_DIR}/STOPPED"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

{
    echo "job_id=${JOB_ID}"
    echo "host=$(hostname)"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodelist=${SLURM_JOB_NODELIST:-unknown}"
    echo "started=$(date -Iseconds)"
} > "${STATE_DIR}/allocation.env"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv \
    > "${STATE_DIR}/gpus.csv"
touch "${STATE_DIR}/READY"

echo "RAGLease allocation ${JOB_ID} is ready on $(hostname)"
echo "Attach with: srun --jobid=${JOB_ID} --overlap --pty bash -l"
echo "Release with: touch ${STATE_DIR}/STOP"

while [[ ! -f "${STATE_DIR}/STOP" ]]; do
    temporary="${STATE_DIR}/heartbeat.tmp"
    date -Iseconds > "${temporary}"
    mv -f "${temporary}" "${STATE_DIR}/heartbeat"
    sleep "${HEARTBEAT_SECONDS}" &
    wait $!
done
