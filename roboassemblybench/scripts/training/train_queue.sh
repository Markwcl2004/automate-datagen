#!/bin/bash
# Training queue: one or more workers per GPU (several: set MPS=1) claim tasks from TASKS (atomic mkdir claims) and train
# each with train_task.sh (the stages in task_variants.json, resuming from the newest checkpoint), then claim the next.
# usage: QUEUE=<name> TASKS="00346 00213 ..." [NENV=128 TAG=evalsig DATE=YYYYMMDD MPS=1] train_queue.sh <activate.sh> <GPU>
# env: SHARED_ROOT (outputs/ on the shared disk)
umask 000
ACT=$1; GPU=$2
set +u; source $ACT
: "${SHARED_ROOT:?set SHARED_ROOT (shared disk with outputs/)}"
Q=$SHARED_ROOT/outputs/queues/${QUEUE:?}; mkdir -p $Q/claims
SC=$(dirname $(readlink -f $0))
TAG=${TAG:-evalsig}; export NENV=${NENV:-128} TAG
WHO="$(hostname | cut -c1-20)_gpu$GPU${WORKER:+_w$WORKER}"
for ID in ${TASKS:?}; do
  mkdir $Q/claims/$ID 2>/dev/null || continue
  echo "$WHO $(date +%F_%T)" > $Q/claims/$ID/owner
  echo "[$(date +%F_%T)] $WHO claimed $ID (NENV=$NENV)" >> $Q/status.txt
  mkdir -p $SHARED_ROOT/outputs/launch_logs; $SC/train_task.sh $ACT $ID $GPU >> $SHARED_ROOT/outputs/launch_logs/q_${ID}.log 2>&1
  echo "[$(date +%F_%T)] $WHO $ID finished, exit code $?" >> $Q/status.txt
done
echo "[$(date +%F_%T)] $WHO WORKER_DONE" >> $Q/status.txt
