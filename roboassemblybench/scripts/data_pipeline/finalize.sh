#!/bin/bash
# After every collection worker finished: leftovers -> last overexposure pass (tasks short of TARGET -> a top-up task list and
# FAILED: SHORT; run workers with QTAG=topup TASK_LIST=<that list>, then rerun this) -> trim to exactly TARGET -> audit -> QC.
# Every step must pass. Progress in <DATASET_ROOT>/_queue/finalize/stage.txt; A_DONE or FAILED at the end.
# env: DATASET_ROOT TASK_CONFIG [PROFILE TARGET NENV PROTECT_BEFORE QC_JOBS PYTHON]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${DATASET_ROOT:?set DATASET_ROOT}" "${TASK_CONFIG:?set TASK_CONFIG}"
DS=$DATASET_ROOT; P=${PROFILE:-mixed}; TARGET=${TARGET:-1200}; PY=${PYTHON:-python3}; SCRATCH=${TMPDIR:-/tmp}
F=$DS/_queue/finalize; [ -d $F ] && mv $F $F.prev_$(date +%m%d_%H%M%S); mkdir -p $F/qc; umask 000
say() { echo "[$(date +%F_%T)] $*" >> $F/stage.txt; }
fail() { say "FAILED: $*"; touch $F/FAILED; exit 1; }
say start
# 1. partial episode dirs left by killed runs (parallel mode leaves them) -> quarantine (moved, not deleted)
n=0
for d in $DS/[0-9]*/$P/episode_*.incomplete; do
  [ -e "$d" ] || continue; t=$(basename $(dirname $(dirname $d)))
  mkdir -p $DS/_quarantine/incomplete/$t/$P && mv "$d" $DS/_quarantine/incomplete/$t/$P/ && n=$((n+1))
done
say "incomplete dirs moved to _quarantine/incomplete: $n"
# 2. last overexposure pass; every kept episode must have a clean scan record
$PY $HERE/overexposure_filter.py --dataset-root $DS --profile $P >> $F/overexposure.txt 2>&1 || fail "overexposure filter crashed"
say "overexposure: $(tail -1 $F/overexposure.txt)"
$PY - $DS $P > $F/scan_check.txt 2>&1 <<'PY' || fail "scan check: $(tail -1 $F/scan_check.txt)"
import glob, json, os, sys
DS, P = sys.argv[1], sys.argv[2]
scan = {}
for l in open(f"{DS}/_queue/overexposure_scan.jsonl"):
    r = json.loads(l); scan[r["ep"]] = r["overexposed"]
kept = [os.path.relpath(e, DS) for e in glob.glob(f"{DS}/[0-9]*/{P}/episode_*_cartesian_raw") if os.path.exists(e + "/metadata.json")]
bad = [e for e in kept if e not in scan or scan[e]]
assert not bad, f"{len(bad)} kept episodes unscanned or flagged, e.g. {bad[:3]}"
print("scan check ok:", len(kept))
PY
$PY $HERE/build_task_list.py --dataset-root $DS --tasks $TASK_CONFIG --out $DS/_queue/task_list_topup.txt --target $TARGET \
  --nenv ${NENV:-16} --profile $P > $F/short.txt 2>&1 || fail "short check"
say "$(tail -1 $F/short.txt)"
grep -q '^tasks 0 ' $F/short.txt || { say "FAILED: SHORT (top-up list $DS/_queue/task_list_topup.txt)"; touch $F/FAILED; exit 1; }
# 3. trim to exactly TARGET (surplus deleted, manifest first)
$PY $HERE/trim_to_target.py --dataset-root $DS --target $TARGET --profile $P ${PROTECT_BEFORE:+--protect-before $PROTECT_BEFORE} --apply > $F/trim.txt 2>&1 \
  || fail "trim: $(tail -2 $F/trim.txt | tr '\n' ' ')"
say "trim: $(grep -E '^tasks|^manifest|^after' $F/trim.txt | tr '\n' ' ')"
grep -q "^after: all $TARGET" $F/trim.txt || fail "not every task at $TARGET"
# 4. duplicates / success records / leftovers
$PY $HERE/audit.py $DS --profile $P > $F/audit.txt 2>&1 || fail "audit crashed"
say "audit: $(tr '\n' ' ' < $F/audit.txt | cut -c1-400)"
grep -q 'duplicate groups 0' $F/audit.txt && grep -q 'kept != success records: 0' $F/audit.txt && grep -q 'incomplete dirs total 0 ' $F/audit.txt || fail "audit"
# 5. QC per task, QC_JOBS at a time
QS=$SCRATCH/automate_qc_$$; rm -rf $QS
ls $DS | grep -E '^[0-9]{5}$' | xargs -P ${QC_JOBS:-24} -I{} bash -c "mkdir -p $QS/{}/{}; ln -s $DS/{}/$P $QS/{}/{}/$P; \
  $PY $HERE/qc.py $QS/{} --profile $P > $F/qc/{}.txt 2>&1 < /dev/null"
rm -rf $QS
ntask=$(ls $DS | grep -cE '^[0-9]{5}$'); ok=$(grep -l '^problems: none' $F/qc/*.txt | wc -l); checked=$(grep -h '^episodes checked' $F/qc/*.txt | awk '{s+=$3} END {print s}')
say "qc: $ok / $ntask tasks problems: none, episodes checked $checked"
[ "$ok" -eq "$ntask" ] && [ "$checked" -eq $((ntask * TARGET)) ] || fail "qc: $(grep -L '^problems: none' $F/qc/*.txt | head -5 | tr '\n' ' ')"
say A_DONE; touch $F/A_DONE
