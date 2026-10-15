"""Task config for the data pipeline from task_variants.json: one line per task `<task> <experiment> <iteration>` from each
task's `policy` entry (the checkpoint with the best formal evaluation), best evaluation first. Append the optional columns
`<grasp raise m | -> <deeper_ok 0|1>` by hand for tasks that need them (see ../data_pipeline/build_task_list.py).
usage: collect_lists.py <task_variants.json> > tasks.txt"""
import json
import sys

tasks = json.load(open(sys.argv[1]))["tasks"]
for t, v in sorted(tasks.items(), key=lambda kv: -kv[1]["policy"]["eval_strict_final"]):
    print(t, v["policy"]["experiment"], v["policy"]["iteration"])
