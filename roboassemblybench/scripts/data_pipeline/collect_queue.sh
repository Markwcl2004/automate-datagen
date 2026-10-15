#!/bin/bash
# Collection queue: one collector process per call, claiming tasks from the task list (atomic mkdir claims) and collecting
# each until TARGET kept episodes. Settings come from the environment (see config.example.env).
#
# Task list, one line per task:  <task> <max rounds per run> <experiment> <iteration> [<grasp raise m | ->] [<deeper_ok 0|1>]
#   policy checkpoint = $POLICY_ROOT/<experiment>/nn/last_Assembly_ep_<iteration>_*.pth
#   column 5: ROBOASSEMBLYBENCH_AUTOMATE_GRASP_RAISE for that task ("-" = none); column 6: 1 = COLLECT_DEEPER_OK=1
#
# usage: collect_queue.sh <activate.sh> <GPU> <task list> [<kit portable root seed dir>]
#   <activate.sh> sets up Isaac Lab (defines ISAACLAB) and is sourced; required env: SHARED_ROOT, DATASET_ROOT, PRODUCTION
umask 000
ACT=$1; GPU=$2; LIST=$3; KR=${4:-}; TARGET=${TARGET:-1200}; PROFILE=${PROFILE:-mixed}; NENV=${NENV:-16}
: "${SHARED_ROOT:?set SHARED_ROOT}" "${DATASET_ROOT:?set DATASET_ROOT}" "${PRODUCTION:?set PRODUCTION}"
POLICY_ROOT=${POLICY_ROOT:-$SHARED_ROOT/outputs/isaaclab_logs/rl_games/Assembly}
SCRATCH=${TMPDIR:-/tmp}
# SLOTS: how many workers may collect the same task at once; slot 1 claims <task>, slot k claims <task>.s<k>.
# SLOTS > 1 turns on the collector's parallel mode (shared episode directory, atomic index claims, per-run attempt logs)
SLOTS=${SLOTS:-1}; [ "$SLOTS" -gt 1 ] && export COLLECT_PARALLEL=1
# a run whose log shows one of these could not load an asset (S3 content server, intermittent): its episodes are
# moved to <dataset root>/_quarantine/asset_load_failed/ and the task is retried with the next run's seed
ASSET_FAIL='Unable to open the usd file|\[ASSET_LOAD_FAILED\]|Could not open asset'
set +u; source $ACT
export REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)  # this checkout, whatever activate.sh set
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$GPU
# QTAG: a later pass over the same task list (e.g. raising TARGET) gets its own claims / logs, same dataset root
ROOT=$DATASET_ROOT; Q=$ROOT/_queue/$PROFILE${QTAG:+_$QTAG}; mkdir -p $Q/claims $Q/logs $Q/ckpt
chmod a+rwx $ROOT 2>/dev/null  # hosts may run as different users: every host must be able to create task dirs
# seed per (profile, task, run), run numbered in the global seed ledger: no two episodes share a layout, so every
# episode is a new motion. mixed 1e6, lighting 2e6, table_color 3e6, object_distractors 4e6, texture 5e6, scene 6e6;
# + task * 100 + run (< 100)
case $PROFILE in mixed) PB=1000000;; lighting) PB=2000000;; table_color) PB=3000000;; object_distractors) PB=4000000;;
  texture) PB=5000000;; scene) PB=6000000;; *) echo "!! unknown profile $PROFILE"; exit 1;; esac
LEDGER=$SHARED_ROOT/outputs/seed_ledger  # <profile>/<task>.<run>/seed: one directory per run, shared by every dataset root
mkdir -p $LEDGER/$PROFILE; chmod a+rwx $LEDGER $LEDGER/$PROFILE 2>/dev/null
# Kit's portable root must be on a local disk: Kit holds POSIX locks in it, and on NFS v3 the unlock at exit never
# returns (processes hang forever, holding GPU memory). The KR on the shared disk is only a seed, copied to a
# per-process directory under KIT_LOCAL_ROOT (needs >= 2 GB free) and removed when the worker ends.
KITARGS=(); KRL=""
if [ -n "$KR" ]; then
  KRL=${KIT_LOCAL_ROOT:-$SCRATCH/automate_kitroot}/$(basename $KR); mkdir -p $(dirname $KRL)
  avail=$(df -Pm $(dirname $KRL) | awk 'NR==2 {print $4}')
  [ "${avail:-0}" -ge 2048 ] || { echo "!! only ${avail} MB free under $(dirname $KRL) for the Kit portable root"; exit 1; }
  mkdir -p $KR $KRL; rsync -a $KR/ $KRL/ 2>/dev/null
  KITARGS=("--kit_args=--portable-root $KRL")
  trap 'rm -rf "$KRL"' EXIT
fi
WHO="$(hostname | cut -c1-20)_gpu$GPU"
while read ID R EXPX ITX RAISE DEEPER; do
  [ "$RAISE" = - ] && RAISE=""
  # a task excluded by hand (claims/<ID>/owner starting with "excluded") is skipped by every slot
  grep -qs '^excluded' $Q/claims/$ID/owner && continue
  CL=""
  for s in $(seq 1 $SLOTS); do c=$ID; [ $s -gt 1 ] && c=$ID.s$s; mkdir $Q/claims/$c 2>/dev/null && CL=$c && break; done
  [ -n "$CL" ] || continue
  echo "$WHO $(date +%F_%T)" > $Q/claims/$CL/owner
  SRC=$(ls $POLICY_ROOT/$EXPX/nn/last_Assembly_ep_${ITX}_*.pth | grep -v cpu_resume | head -1)
  [ -n "$SRC" ] || { echo "[$(date +%F_%T)] $WHO $ID: no checkpoint $POLICY_ROOT/$EXPX/nn/last_Assembly_ep_${ITX}_*.pth" >> $Q/status.txt; continue; }
  python -c "import torch,sys; torch.save(torch.load(sys.argv[1], map_location='cpu', weights_only=False), sys.argv[2])" $SRC $Q/ckpt/$CL.cpu.pth < /dev/null
  # a start that cannot fetch an asset from the S3 content server (intermittent: the ground plane, or a warehouse prop
  # that loads incompletely) is retried with the next run's seed
  for try in 1 2 3; do
    # the run number comes from the global seed ledger (all dataset roots, smoke tests included): the first n whose
    # directory mkdir creates (atomic on NFS, no file locks: NLM locks hang on this NFS) is this run's, so a seed is
    # never used twice anywhere
    SEED=""
    for n in $(seq 0 99); do
      if mkdir $LEDGER/$PROFILE/$ID.$n 2>/dev/null; then
        SEED=$(( PB + 10#$ID * 100 + n )); echo "$SEED $ROOT $WHO $(date +%F_%T)" > $LEDGER/$PROFILE/$ID.$n/seed; break
      fi
    done
    [ -n "$SEED" ] || { echo "!! $ID: 100 runs"; break; }
    echo "[$(date +%F_%T)] $WHO claimed $ID seed=$SEED" >> $Q/status.txt
    ( if [ -n "$RAISE" ]; then export ROBOASSEMBLYBENCH_AUTOMATE_GRASP_RAISE=$RAISE; else unset ROBOASSEMBLYBENCH_AUTOMATE_GRASP_RAISE; fi
      if [ "$DEEPER" = 1 ]; then export COLLECT_DEEPER_OK=1; else unset COLLECT_DEEPER_OK; fi
      cd $ISAACLAB && ./isaaclab.sh -p $REPO_ROOT/roboassemblybench/scripts/collect_automate_single_arm.py --assembly-id $ID \
        --checkpoint $Q/ckpt/$CL.cpu.pth --checkpoint-source $SRC --out-dir $ROOT --target-successes $TARGET --max-rounds $R \
        --num_envs $NENV --seed $SEED --production-id $PRODUCTION --visual-profile $PROFILE --headless --enable_cameras --device cuda:0 "${KITARGS[@]}" \
        agent.params.config.device=cuda:0 hydra.run.dir=$SCRATCH/automate_hydra/${WHO}_${ID}_${PROFILE}_${SEED} ) < /dev/null > $Q/logs/$ID.$SEED.log 2>&1
    # retried: an asset that did not load, or a run that died without its [COLLECT] summary (crash, killed)
    if ! grep -aqE "$ASSET_FAIL" $Q/logs/$ID.$SEED.log && grep -aq '\[COLLECT\] {' $Q/logs/$ID.$SEED.log; then break; fi
    grep -aq '\[COLLECT\] {' $Q/logs/$ID.$SEED.log || echo "[$(date +%F_%T)] $WHO $ID seed=$SEED died without a summary, retrying" >> $Q/status.txt
    # after an asset load failure the episodes this run wrote are not kept (a missing sub-asset may be in their
    # images); after a crash its finished episodes are fine and the next run resumes after them
    grep -aqE "$ASSET_FAIL" $Q/logs/$ID.$SEED.log && python3 - "$ROOT" "$ID" "$PROFILE" "$SEED" <<'PY' >> $Q/status.txt
import glob, json, os, sys
root, task, prof, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
dst = f"{root}/_quarantine/asset_load_failed/{task}/{prof}"; n = 0
for d in glob.glob(f"{root}/{task}/{prof}/episode_*_cartesian_raw"):
    try:
        if json.load(open(d + "/metadata.json"))["seed"] != seed: continue
    except (OSError, ValueError, KeyError):
        continue
    os.makedirs(dst, exist_ok=True); os.rename(d, f"{dst}/{os.path.basename(d)}_seed{seed}"); n += 1
print(f"asset load failed: task {task} seed {seed}, {n} episodes moved to _quarantine/asset_load_failed")
PY
  done
  rm -f $Q/ckpt/$CL.cpu.pth
  echo "[$(date +%F_%T)] $WHO $ID finished $(grep -a '\[COLLECT\]' $Q/logs/$ID.$SEED.log | grep -o '"successes": [0-9]*, "attempts": [0-9]*')" >> $Q/status.txt
done < $LIST
echo "[$(date +%F_%T)] $WHO WORKER_DONE" >> $Q/status.txt
