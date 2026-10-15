#!/bin/bash
# Train one task as task_variants.json specifies: its training stages run in order, each resuming from the previous
# one (train_resume.sh continues from the newest checkpoint of the experiment), with the stage's hydra overrides.
# Tasks without an entry train one stage of 1000 iterations. The curriculum level is env state: it restarts at level 0
# at every stage.   usage: [TAG=evalsig DATE=... NENV=128 MPS=1 SEED=0] train_task.sh <activate.sh> <task> <GPU>
ACT=$1; ID=$2; GPU=$3; SC=$(dirname $(readlink -f $0))
set +u; source $ACT >/dev/null 2>&1
export REPO_ROOT=$(cd "$SC/../../.." && pwd)  # this checkout, whatever activate.sh set
V=${ROBOASSEMBLYBENCH_AUTOMATE_TASK_VARIANTS:-$REPO_ROOT/roboassemblybench/library/automate/task_variants.json}
STAGES=$(python3 - "$V" "$ID" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))["tasks"].get(sys.argv[2], {}) if sys.argv[1] else {}
for st in t.get("training", {}).get("stages", [{"iterations": 1000}]):
    print(st["iterations"], " ".join(st.get("hydra_overrides", [])))
PY
)
echo "$STAGES" | while read IT OV; do
  echo "[$(date +%F_%T)] $ID stage to $IT iterations ${OV:+(overrides: $OV)}"
  ITERS=$IT EXTRA_HYDRA="$OV" $SC/train_resume.sh $ACT $ID $GPU || exit 1
done
