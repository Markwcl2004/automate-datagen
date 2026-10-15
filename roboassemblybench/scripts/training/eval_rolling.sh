#!/bin/bash
# Rolling final evaluation (final checkpoint, 100 episodes, 50 envs, curriculum off, deterministic, rollout_logger.py) that
# does not wait for a whole training group: whichever task produces last_Assembly_ep_${IT}_*.pth first is evaluated first;
# exits after STALL seconds without a new task. A start without a result (usually a PhysX GPU memory failure when sharing a
# GPU with training) is moved to _failed/ and retried later, at most 3 times.
# usage: TASKS="..." EXPFMT=<date>_%s_<tag> OUTD=<output dir> [QUEUE=... N=2 IT=1000 STALL=21600] eval_rolling.sh <activate.sh> <GPU>
# env: SHARED_ROOT
umask 000
ACT=$1; GPU=$2; N=${N:-2}; IT=${IT:-1000}; QUEUE=${QUEUE:-$OUTD}; STALL=${STALL:-21600}
set +u; source $ACT
: "${SHARED_ROOT:?set SHARED_ROOT (shared disk with outputs/)}"
export REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)  # this checkout, whatever activate.sh set
A=$SHARED_ROOT/outputs/isaaclab_logs/rl_games/Assembly
export CUDA_MPS_PIPE_DIRECTORY=${TMPDIR:-/tmp}/automate-mps-gpu$GPU CUDA_MPS_LOG_DIRECTORY=${TMPDIR:-/tmp}/automate-mps-gpu$GPU-log
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 || { mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d; sleep 2; }
export CUDA_VISIBLE_DEVICES=0
Q=$SHARED_ROOT/outputs/queues/$QUEUE; mkdir -p $Q/claims
worker() {
  local WHO="$(hostname | cut -c1-20)_gpu${GPU}_w$1" last=$(date +%s)
  local KRL=${KIT_LOCAL_ROOT:-${TMPDIR:-/tmp}/automate_kitroot}/eval_${QUEUE}_gpu${GPU}_w$1; mkdir -p $KRL $Q/fails
  while :; do
    local did=0 left=0
    for ID in ${TASKS}; do
      [ -d $Q/claims/$ID ] && continue
      [ $(cat $Q/fails/$ID 2>/dev/null || echo 0) -ge 3 ] && continue; left=1
      EXP=$(printf $EXPFMT $ID); NN=$A/$EXP/nn
      ck=$(ls $NN/last_Assembly_ep_${IT}_*.pth 2>/dev/null | grep -v cpu_resume | head -1); [ -n "$ck" ] || continue
      mkdir $Q/claims/$ID 2>/dev/null || continue
      echo "$WHO $(date +%F_%T)" > $Q/claims/$ID/owner
      O=$SHARED_ROOT/outputs/$OUTD/$ID; mkdir -p $O; sha=$(sha256sum $ck | cut -c1-12)
      python -c "import torch,sys; torch.save(torch.load(sys.argv[1], map_location='cpu', weights_only=False), sys.argv[2])" $ck $O/final.cpu.pth
      ( cd $ISAACLAB && ./isaaclab.sh -p $REPO_ROOT/roboassemblybench/scripts/training/rollout_logger.py --assembly-id $ID --checkpoint $O/final.cpu.pth --out $O/rollout_final.npz \
          --num_envs 50 --episodes 100 --headless --device cuda:0 "--kit_args=--portable-root $KRL" agent.params.config.device=cuda:0 > $O/rollout_final.log 2>&1 )
      rm -f $O/final.cpu.pth
      if ! grep -qa '\[ROLLOUT\]' $O/rollout_final.log; then
        mkdir -p $SHARED_ROOT/outputs/$OUTD/_failed; mv $O $SHARED_ROOT/outputs/$OUTD/_failed/${ID}_$(date +%H%M%S)
        echo $(( $(cat $Q/fails/$ID 2>/dev/null || echo 0) + 1 )) > $Q/fails/$ID; rm -rf $Q/claims/$ID
        echo "[$(date +%F_%T)] $WHO $ID: no result, retry later" >> $Q/status.txt; sleep 60; continue
      fi
      echo "$ID final exp=$EXP ckpt=$(basename $ck) sha256=$sha $(grep -a '\[ROLLOUT\]' $O/rollout_final.log | sed 's/.*episodes=/episodes=/' | sed 's/ -> .*//')" >> $O/summary.txt
      echo "[$(date +%F_%T)] $WHO $ID evaluated" >> $Q/status.txt; did=1; last=$(date +%s)
    done
    [ $left = 0 ] && break
    [ $did = 0 ] && { [ $(( $(date +%s) - last )) -gt $STALL ] && break; sleep 120; }
  done
  echo "[$(date +%F_%T)] $WHO WORKER_DONE" >> $Q/status.txt
}
for w in $(seq 1 $N); do worker $w & sleep 20; done
wait
