#!/bin/bash
# Run overexposure_filter.py every INTERVAL seconds during a collection, so the collectors (parallel mode re-reads the kept
# count before every round) replace what it moves out. Stops when WORKERS workers of queue QTAG have finished, or after
# MAX_HOURS. env: DATASET_ROOT [PROFILE QTAG WORKERS INTERVAL MAX_HOURS PYTHON]
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${DATASET_ROOT:?set DATASET_ROOT}"
PROFILE=${PROFILE:-mixed}; Q=$DATASET_ROOT/_queue/${PROFILE}_${QTAG:-main}; L=$DATASET_ROOT/_queue/overexposure_filter_loop.log
T0=$(date +%s); MAX=$(( ${MAX_HOURS:-72} * 3600 ))
while [ $(( $(date +%s) - T0 )) -lt $MAX ]; do
  ${PYTHON:-python3} $HERE/overexposure_filter.py --dataset-root $DATASET_ROOT --profile $PROFILE >> $L 2>&1
  [ "$(grep -c WORKER_DONE $Q/status.txt 2>/dev/null)" -ge "${WORKERS:-20}" ] 2>/dev/null && { echo "$(date +%F_%T) collection finished, loop ends" >> $L; break; }
  sleep ${INTERVAL:-180}
done
