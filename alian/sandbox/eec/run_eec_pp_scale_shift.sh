#!/bin/bash
# Run EEC with various scale shifts to test sensitivity to jet energy scale

LOGDIR="logs_escale"
mkdir -p "${LOGDIR}"
JOBLOG="${LOGDIR}/parallel_joblog.txt"

run_one() {
    nev=50000
    # nev=20000
    # nev=5000
    local scale_shift=$1
    local logfile="logs_escale/run_scale_shift_${scale_shift}.log"
    echo "[$(date '+%H:%M:%S')] START scale_shift=${scale_shift}" | tee "${logfile}"
    ./eec_pp_escale_test.py --py-pthatmin 18 --jet-pt-min 20 --jet-pt-max 40 --py-hardQCD --nev ${nev} --scale-shift ${scale_shift} \
        >> "${logfile}" 2>&1
    local rc=$?
    echo "[$(date '+%H:%M:%S')] DONE  scale_shift=${scale_shift}  exit=${rc}" | tee -a "${logfile}"
    return ${rc}
}
export -f run_one

echo "Logs → ${LOGDIR}/  |  job log → ${JOBLOG}"
parallel --joblog "${JOBLOG}" --tag --line-buffer run_one ::: -2 -1 -0.6 0.6 1 2
# parallel --joblog "${JOBLOG}" --tag --line-buffer run_one ::: -5 -1 -0.6 0.6 1 5
# parallel --joblog "${JOBLOG}" --tag --line-buffer run_one ::: 5

echo ""
echo "=== Summary ==="
awk 'NR>1 {printf "scale_shift=%-6s  exit=%s  runtime=%ss\n", $NF, $7, $4}' "${JOBLOG}"