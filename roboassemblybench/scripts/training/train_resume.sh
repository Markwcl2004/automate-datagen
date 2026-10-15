#!/bin/bash
# RL training of one task; after a kill / crash it resumes from the newest checkpoint; a last_*.pth every SAVE_FREQ epochs
# EXTRA_HYDRA: hydra overrides (space separated, e.g. env.tasks.insertion.curriculum_success_thresh=0.6) appended to the training command
# usage: [ITERS=1000 SAVE_FREQ=300 TAG=evalsig DATE=YYYYMMDD NENV=128 MPS=1 SEED=0 EXTRA_HYDRA=...] train_resume.sh <activate.sh> <assembly_id> <GPU>
# env: SHARED_ROOT; TRAJ_ROOT (disassembly reference trajectories <TRAJ_ROOT>/<task>/<task>_disassemble_traj.json) or TRAJ
# MPS=1: several training processes on one GPU through NVIDIA MPS (without MPS they time-slice and barely add throughput)
umask 000   # hosts may run as different uids on the shared disk: new directories must be writable by all
ACT=$1; ID=$2; GPU=$3
ITERS=${ITERS:-1000}; SAVE_FREQ=${SAVE_FREQ:-300}; TAG=${TAG:-evalsig}; NENV=${NENV:-128}; DATE=${DATE:-$(date +%Y%m%d)}
set +u; source $ACT
export REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)  # this checkout, whatever activate.sh set
# expose only the assigned GPU: upstream warp uses wp.get_preferred_device() (always cuda:0) and would put the SDF work on another GPU
DEV=cuda:0
if [ -n "${MPS:-}" ]; then
  # one MPS daemon per GPU that sees only that GPU (a daemon seeing all GPUs creates contexts on every GPU, including
  # GPUs other people use); client device ids are relative to the daemon's visible GPUs, hence 0
  export CUDA_MPS_PIPE_DIRECTORY=${TMPDIR:-/tmp}/automate-mps-gpu$GPU CUDA_MPS_LOG_DIRECTORY=${TMPDIR:-/tmp}/automate-mps-gpu$GPU-log CUDA_VISIBLE_DEVICES=0
  mps_up() { echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 && return
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d; sleep 2; }
else
  export CUDA_VISIBLE_DEVICES=$GPU; mps_up() { :; }
fi
: "${SHARED_ROOT:?set SHARED_ROOT (shared disk with outputs/)}"
TRAJ=${TRAJ:-${TRAJ_ROOT:?set TRAJ_ROOT or TRAJ}/$ID/${ID}_disassemble_traj.json}
[ -s $TRAJ ] || { echo "!! reference trajectory $TRAJ missing"; exit 1; }
exp=${DATE}_${ID}_${TAG}
LOGD=$SHARED_ROOT/outputs/isaaclab_logs/rl_games/Assembly/$exp      # every host links IsaacLab/logs to the shared disk
OUT=$SHARED_ROOT/outputs/${DATE}_${TAG}; mkdir -p $OUT/logs
[ "$(readlink -f $ISAACLAB/logs)" = "$(readlink -f $SHARED_ROOT/outputs/isaaclab_logs)" ] || { echo "!! $ISAACLAB/logs does not point to $SHARED_ROOT/outputs/isaaclab_logs"; exit 1; }
cd $ISAACLAB
for attempt in $(seq 1 20); do
  if ls $LOGD/nn/last_Assembly_ep_${ITERS}_*.pth >/dev/null 2>&1; then echo "[$(date +%F_%T)] TRAIN_DONE $exp"; exit 0; fi
  ck=$(ls -t $LOGD/nn/*.pth 2>/dev/null | head -1)
  RES=()
  if [ -n "$ck" ]; then
    # rl_games loads checkpoints without map_location (back onto the GPU they were saved from): resume from a CPU copy
    case "$ck" in *.cpu_resume.pth) rck=$ck ;; *) rck=${ck%.pth}.cpu_resume.pth
      python -c "import torch,sys; torch.save(torch.load(sys.argv[1], map_location='cpu', weights_only=False), sys.argv[2])" $ck $rck || rck=$ck ;; esac
    RES=(--checkpoint $rck)
  fi
  mps_up
  echo "  traj: $TRAJ"
  echo "  code: $(cd $REPO_ROOT && git log --oneline -1 | cut -c1-7) + $(cd $REPO_ROOT && git diff --stat | tail -1)   NENV=$NENV   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES   MPS=${CUDA_MPS_PIPE_DIRECTORY:-off}"
  echo "[$(date +%F_%T)] $exp attempt $attempt on $(hostname | cut -c1-20) gpu$GPU resume=${ck:-none}"
  ./isaaclab.sh -p $REPO_ROOT/roboassemblybench/library/automate/run_assembly.py --mode train --assembly-id $ID \
    --disassembly-json $TRAJ --num-envs $NENV --max-iterations $ITERS --seed ${SEED:-0} "${RES[@]}" --headless --device $DEV ${EXTRA_HYDRA:-} \
    agent.params.config.device=$DEV agent.params.config.full_experiment_name=$exp \
    agent.params.config.save_frequency=$SAVE_FREQ > $OUT/logs/$exp.a$attempt.log 2>&1
  echo "  exit code $?; last error: $(grep -a -A3 "^Traceback" $OUT/logs/$exp.a$attempt.log | tail -1 | cut -c1-160)"
  sleep 60
done
echo "!! $exp not finished after 20 attempts"; exit 1
