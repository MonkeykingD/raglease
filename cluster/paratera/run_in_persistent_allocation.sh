#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: bash $0 <a100|h100> -- '<experiment command>'

Set RAGLEASE_JOB_ID to override the recorded allocation job.
EOF
    exit 64
}

[[ $# -eq 3 && "${2:-}" == "--" ]] || usage

PLATFORM="$1"
COMMAND="$3"
case "${PLATFORM}" in
    a100|h100) ;;
    *) usage ;;
esac

RAGLEASE_ROOT="${RAGLEASE_ROOT:-${HOME}/Performance01/dyh/raglease}"
STATE_DIR="${RAGLEASE_ROOT}/run/allocations"
LATEST_FILE="${STATE_DIR}/latest-${PLATFORM}.jobid"
JOB_NAME="raglease-${PLATFORM}-persistent"
LOCK_WAIT_SECONDS="${RAGLEASE_LOCK_WAIT_SECONDS:-28800}"
LOCK_HEARTBEAT_SECONDS="${RAGLEASE_LOCK_HEARTBEAT_SECONDS:-15}"
LOCK_STALE_SECONDS="${RAGLEASE_LOCK_STALE_SECONDS:-300}"
ALLOW_STALE_LOCK_REMOVAL="${RAGLEASE_ALLOW_STALE_LOCK_REMOVAL:-0}"

case "${PLATFORM}" in
    a100) DEFAULT_COMPUTE_ENV="${RAGLEASE_ROOT}/envs/raglease" ;;
    h100) DEFAULT_COMPUTE_ENV="${RAGLEASE_ROOT}/envs/raglease-h100" ;;
esac
COMPUTE_ENV="${RAGLEASE_COMPUTE_ENV:-${DEFAULT_COMPUTE_ENV}}"
CUDA_TARGET="${COMPUTE_ENV}/targets/x86_64-linux"
if [[ -d "${CUDA_TARGET}/lib" ]]; then
    export PATH="${COMPUTE_ENV}/bin:${PATH}"
    export CUDA_HOME="${RAGLEASE_CUDA_HOME_OVERRIDE:-${COMPUTE_ENV}}"
    export LD_LIBRARY_PATH="${CUDA_TARGET}/lib:${COMPUTE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${CUDA_TARGET}/lib:${COMPUTE_ENV}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

# SGLang's server, profiler, and benchmark communicate over loopback. Preserve
# any site proxy while ensuring local health checks never leave the node.
LOCAL_NO_PROXY="127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

running_job="$({
    squeue -h -u "${USER}" -n "${JOB_NAME}" -t R -o '%A' 2>/dev/null || true
} | head -n 1)"

if [[ -n "${RAGLEASE_JOB_ID:-}" ]]; then
    job_id="${RAGLEASE_JOB_ID}"
elif [[ -n "${running_job}" ]]; then
    job_id="${running_job}"
elif [[ -s "${LATEST_FILE}" ]]; then
    job_id="$(tr -d '[:space:]' < "${LATEST_FILE}")"
else
    echo "No ${PLATFORM} persistent allocation is recorded." >&2
    echo "Submit one with: bash cluster/paratera/submit_persistent_allocation.sh ${PLATFORM}" >&2
    exit 69
fi

# A per-allocation guard can prevent a stale supervisor from relaunching a
# Wikipedia build after that build has been migrated to another allocation.
WIKIPEDIA_BLOCK_FILE="${STATE_DIR}/block-wikipedia-${job_id}"
if [[ -f "${WIKIPEDIA_BLOCK_FILE}" && "${COMMAND}" == *wikipedia* ]]; then
    echo "Wikipedia work is blocked in allocation ${job_id} by ${WIKIPEDIA_BLOCK_FILE}." >&2
    exit 75
fi

job_state="$(squeue -h -j "${job_id}" -o '%T' 2>/dev/null | head -n 1)"
if [[ "${job_state}" != "RUNNING" ]]; then
    echo "Allocation ${job_id} is not running (state=${job_state:-not-in-queue})." >&2
    squeue -j "${job_id}" -o '%.18i %.10P %.2t %.10M %.19S %R' >&2 || true
    exit 75
fi

LOCK_DIR="${RAGLEASE_ROOT}/run/locks"
RUN_ROOT="${RAGLEASE_ROOT}/run/commands/${job_id}"
RUN_ID="${RAGLEASE_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOCK_FILE="${LOCK_DIR}/${PLATFORM}-${job_id}.lock"
LOCK_GUARD_DIR="${LOCK_FILE}.d"
mkdir -p "${LOCK_DIR}" "${RUN_DIR}"

{
    echo "platform=${PLATFORM}"
    echo "job_id=${job_id}"
    echo "run_id=${RUN_ID}"
    echo "submitted=$(date -Iseconds)"
    printf 'command=%q\n' "${COMMAND}"
} > "${RUN_DIR}/metadata.env"

echo "Waiting for the serial allocation lock: ${LOCK_GUARD_DIR}"
lock_wait_started="$(date +%s)"
last_active_step_warning=0
while ! mkdir "${LOCK_GUARD_DIR}" 2>/dev/null; do
    owner_host="$(cat "${LOCK_GUARD_DIR}/host" 2>/dev/null || true)"
    owner_pid="$(cat "${LOCK_GUARD_DIR}/pid" 2>/dev/null || true)"
    heartbeat_epoch="$(cat "${LOCK_GUARD_DIR}/heartbeat" 2>/dev/null || true)"
    now="$(date +%s)"
    stale_reason=""
    if [[ "${owner_host}" == "$(hostname)" \
        && "${owner_pid}" =~ ^[0-9]+$ \
        && ! -d "/proc/${owner_pid}" ]]; then
        stale_reason="dead local owner pid=${owner_pid}"
    elif [[ "${heartbeat_epoch}" =~ ^[0-9]+$ ]] \
        && (( now - heartbeat_epoch >= LOCK_STALE_SECONDS )); then
        stale_reason="heartbeat age=$((now - heartbeat_epoch))s"
    elif [[ ! "${heartbeat_epoch}" =~ ^[0-9]+$ ]]; then
        lock_mtime="$(stat -c '%Y' "${LOCK_GUARD_DIR}" 2>/dev/null || true)"
        if [[ "${lock_mtime}" =~ ^[0-9]+$ ]] \
            && (( now - lock_mtime >= LOCK_STALE_SECONDS )); then
            stale_reason="legacy lock age=$((now - lock_mtime))s"
        fi
    fi
    if [[ -n "${stale_reason}" ]]; then
        active_step="$({
            squeue --steps -h -j "${job_id}" -o '%i' 2>/dev/null || true
        } | awk -v prefix="${job_id}." '
            index($1, prefix) == 1 && $1 !~ /\.batch$/ && $1 !~ /\.extern$/ {
                print $1
                exit
            }
        ')"
        if [[ -n "${active_step}" ]]; then
            if (( now - last_active_step_warning >= 60 )); then
                echo "Lock appears stale (${stale_reason}), but active step ${active_step} remains; waiting instead of overlapping."
                last_active_step_warning="${now}"
            fi
            stale_reason=""
        fi
    fi
    if [[ -n "${stale_reason}" && "${ALLOW_STALE_LOCK_REMOVAL}" != "1" ]]; then
        if (( now - last_active_step_warning >= 60 )); then
            echo "Lock appears stale (${stale_reason}); automatic removal is disabled. Verify the allocation before clearing it."
            last_active_step_warning="${now}"
        fi
        stale_reason=""
    fi
    if [[ -n "${stale_reason}" ]]; then
        echo "Removing stale allocation lock (${stale_reason})."
        rm -f "${LOCK_GUARD_DIR}/host" "${LOCK_GUARD_DIR}/pid" \
            "${LOCK_GUARD_DIR}/heartbeat" "${LOCK_GUARD_DIR}"/heartbeat.*.tmp
        rmdir "${LOCK_GUARD_DIR}" 2>/dev/null || true
        continue
    fi
    if (( now - lock_wait_started >= LOCK_WAIT_SECONDS )); then
        echo "Timed out waiting for allocation ${job_id}." >&2
        exit 75
    fi
    sleep 2
done
printf '%s\n' "$(hostname)" > "${LOCK_GUARD_DIR}/host"
printf '%s\n' "$$" > "${LOCK_GUARD_DIR}/pid"
date +%s > "${LOCK_GUARD_DIR}/heartbeat"

lock_heartbeat_pid=""

refresh_lock_heartbeat() {
    while true; do
        temporary="${LOCK_GUARD_DIR}/heartbeat.$$.tmp"
        date +%s > "${temporary}"
        mv -f "${temporary}" "${LOCK_GUARD_DIR}/heartbeat"
        sleep "${LOCK_HEARTBEAT_SECONDS}"
    done
}

release_serial_lock() {
    if [[ "${lock_heartbeat_pid}" =~ ^[0-9]+$ ]]; then
        kill "${lock_heartbeat_pid}" 2>/dev/null || true
        wait "${lock_heartbeat_pid}" 2>/dev/null || true
    fi
    lock_owner_pid="$(cat "${LOCK_GUARD_DIR}/pid" 2>/dev/null || true)"
    lock_owner_host="$(cat "${LOCK_GUARD_DIR}/host" 2>/dev/null || true)"
    if [[ "${lock_owner_pid}" != "$$" \
        || "${lock_owner_host}" != "$(hostname)" ]]; then
        return
    fi
    rm -f "${LOCK_GUARD_DIR}/host" "${LOCK_GUARD_DIR}/pid" \
        "${LOCK_GUARD_DIR}/heartbeat" "${LOCK_GUARD_DIR}"/heartbeat.*.tmp
    rmdir "${LOCK_GUARD_DIR}" 2>/dev/null || true
}
trap release_serial_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

refresh_lock_heartbeat &
lock_heartbeat_pid="$!"

echo "Running command ${RUN_ID} in allocation ${job_id}."
date -Iseconds > "${RUN_DIR}/started"
set +e
srun --overlap \
    --jobid="${job_id}" \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    --export=ALL \
    bash -lc "${COMMAND}" 2>&1 | tee "${RUN_DIR}/output.log"
run_status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${run_status}" > "${RUN_DIR}/exit_code"
date -Iseconds > "${RUN_DIR}/finished"

exit "${run_status}"
