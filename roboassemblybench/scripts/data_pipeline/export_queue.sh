#!/bin/bash
# Export every task of DATASET_ROOT to LeRobot v3 + ourdata-diversity-1.1, one task at a time (mkdir claims shared by every
# worker on every host), then validate it with the evaluator's export_features.py if VALIDATOR_DIR is set.
# usage: export_queue.sh <worker name>
# env: DATASET_ROOT EXPORT_ROOT PRODUCTION [PROFILE ROBOT_TYPE PYTHON EXPORT_PYTHONPATH VALIDATOR_DIR] + the exporter's
#      AUTOMATE_VISUAL_ASSETS (or SHARED_ROOT)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${DATASET_ROOT:?set DATASET_ROOT}" "${EXPORT_ROOT:?set EXPORT_ROOT}" "${PRODUCTION:?set PRODUCTION}"
PROFILE=${PROFILE:-mixed}; ROBOT_TYPE=${ROBOT_TYPE:-franka_panda_single_arm}; PY=${PYTHON:-python3}; SCRATCH=${TMPDIR:-/tmp}
OUT=$EXPORT_ROOT/$ROBOT_TYPE; Q=$EXPORT_ROOT/_queue; mkdir -p $OUT $Q/claims $Q/logs; umask 000
for t in $(ls $DATASET_ROOT | grep -E '^[0-9]{5}$'); do
  mkdir $Q/claims/$t 2>/dev/null || continue
  echo "$1 $(date +%F_%T)" > $Q/claims/$t/owner
  PYTHONPATH=${EXPORT_PYTHONPATH:-} $PY $HERE/export_lerobot_v3.py --raw $DATASET_ROOT/$t/$PROFILE --out $OUT/$t/$PROFILE \
    --dataset-id $ROBOT_TYPE-$PRODUCTION-$t-$PROFILE --visual-randomization $HERE/visual_randomization_mixed.json < /dev/null > $Q/logs/$t.export.log 2>&1
  v="validate=skipped"
  if [ -n "${VALIDATOR_DIR:-}" ]; then
    ( cd $VALIDATOR_DIR && PYTHONPATH=${EXPORT_PYTHONPATH:-} $PY export_features.py --dataset $OUT/$t/$PROFILE --out $SCRATCH/automate_validate_$t \
        --max-episodes 0 --require-mount < /dev/null > $Q/logs/$t.validate.log 2>&1 )
    rm -rf $SCRATCH/automate_validate_$t; v="validate=$(grep -o '"status": "[A-Z_]*"' $Q/logs/$t.validate.log | head -1)"
  fi
  echo "[$(date +%F_%T)] $1 $t export=$(tail -1 $Q/logs/$t.export.log | grep -o '"episodes": [0-9]*\|"structural_check": "[a-z]*"' | tr '\n' ' ') $v" >> $Q/status.txt
done
echo "[$(date +%F_%T)] $1 WORKER_DONE" >> $Q/status.txt
