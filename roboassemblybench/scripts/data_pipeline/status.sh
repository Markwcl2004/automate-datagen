#!/bin/bash
# Status of the pipeline: status.sh finalize | topup <qtag> [workers] | exports [workers] | summary
# prints OK / BAD <reason> / running. env: DATASET_ROOT EXPORT_ROOT [PROFILE ROBOT_TYPE TARGET]
P=${PROFILE:-mixed}; F=$DATASET_ROOT/_queue/finalize; Q=$EXPORT_ROOT/_queue; X=$EXPORT_ROOT/${ROBOT_TYPE:-franka_panda_single_arm}
case $1 in
  finalize) if [ -e $F/A_DONE ]; then echo OK; elif [ -e $F/FAILED ]; then echo "BAD $(tail -1 $F/stage.txt)"; else echo "running $(tail -1 $F/stage.txt 2>/dev/null)"; fi;;
  topup)
    s=$DATASET_ROOT/_queue/${P}_$2/status.txt; [ -e $s ] || { echo running; exit 0; }
    tb=$(grep -l Traceback $(dirname $s)/logs/*.log 2>/dev/null | wc -l); died=$(grep -c "died\|asset load" $s)
    if [ $tb -gt 0 ] || [ $died -gt 0 ]; then echo "BAD traceback_logs=$tb died=$died"
    elif [ $(grep -c WORKER_DONE $s) -ge ${3:-20} ]; then echo OK; else echo running; fi;;
  exports)
    s=$Q/status.txt; [ -e $s ] || { echo running; exit 0; }
    bad=$(awk '!/WORKER_DONE/ && !(/"structural_check": "pass"/ && (/FEATURE_COMPATIBILITY_PASS/ || /validate=skipped/))' $s | head -2)
    if [ -n "$bad" ]; then echo "BAD $bad"; elif [ $(grep -c WORKER_DONE $s) -ge ${2:-1} ]; then echo OK; else echo "running $(grep -vc WORKER_DONE $s) exported"; fi;;
  summary)
    n=0; for t in $(ls $X); do [ "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" $X/$t/$P/meta/info.json)" = "${TARGET:-1200}" ] && n=$((n+1)); done
    echo "datasets $(ls $X | wc -l) | total_episodes=${TARGET:-1200}: $n | validator PASS $(grep -c FEATURE_COMPATIBILITY_PASS $Q/status.txt) | structural pass $(grep -c '"structural_check": "pass"' $Q/status.txt) | size $(du -sh $X | cut -f1)";;
  *) echo "usage: status.sh finalize | topup <qtag> [workers] | exports [workers] | summary"; exit 2;;
esac
