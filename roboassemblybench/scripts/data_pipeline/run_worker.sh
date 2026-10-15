#!/bin/bash
# One collection worker: collect_queue.sh over the task list, with the settings of the environment (config.example.env).
# usage: run_worker.sh <activate.sh> <GPU> [<kit portable root seed dir>]
#   env: SHARED_ROOT DATASET_ROOT PRODUCTION [PROFILE TARGET NENV SLOTS QTAG TASK_LIST POLICY_ROOT KIT_LOCAL_ROOT]
#   a file <DATASET_ROOT>/_queue/STOP keeps new workers from starting
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${DATASET_ROOT:?set DATASET_ROOT}"
export PROFILE=${PROFILE:-mixed} NENV=${NENV:-16} SLOTS=${SLOTS:-3} TARGET=${TARGET:-1200} QTAG=${QTAG:-main}
TASK_LIST=${TASK_LIST:-$DATASET_ROOT/_queue/task_list_$QTAG.txt}
[ -e $DATASET_ROOT/_queue/STOP ] && exit 0
bash $HERE/collect_queue.sh "$1" "$2" "$TASK_LIST" "${3:-}"
