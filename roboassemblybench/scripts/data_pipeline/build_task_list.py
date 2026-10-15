"""Order the task list of a collection pass and size its runs.

Input: a task config, one line per task `<task> <experiment> <iteration> [<grasp raise m | ->] [<deeper_ok 0|1>]`.
For every task: kept episodes now, success rate from its attempt logs (floor 0.2, times --keep-rate for the share of kept
episodes later removed by the overexposure filter), episodes still needed for --target, max rounds per run
ceil(need / (nenv * p) * 1.5) + 2 (cap 300). Tasks already at the target are left out; the longest expected collection
comes first. Output lines are collect_queue.sh's format: `<task> <max rounds> <experiment> <iteration> <raise> <deeper_ok>`.

--preclaim also claims every episode index a task has ever used (kept, quarantined, listed in trim manifests) as
<task>/<profile>/.idx/<idx>, so later runs never reuse the name of an episode that existed before (the collector itself
claims every index it takes; this is for dataset roots written before that).

usage: build_task_list.py --dataset-root D --tasks CONFIG --out LIST [--target 1200 --nenv 16 --profile mixed
                          --keep-rate 1.0 --preclaim]"""
import argparse, glob, json, math, os

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-root", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--target", type=int, default=1200); ap.add_argument("--nenv", type=int, default=16); ap.add_argument("--profile", default="mixed")
ap.add_argument("--keep-rate", type=float, default=1.0); ap.add_argument("--preclaim", action="store_true")
a = ap.parse_args()
DS = a.dataset_root
os.umask(0)  # claims must be writable by every host (hosts may run as different users)
dropped = {}
for mp in glob.glob(f"{DS}/_manifests/trim_*_manifest_*.json"):
    for t, v in json.load(open(mp))["tasks"].items():
        dropped.setdefault(t, []).extend(v["dropped"])
out = []
for line in open(a.tasks):
    f = line.split()
    if not f or f[0].startswith("#"):
        continue
    t, exp, it = f[0], f[1], f[2]
    raise_m = f[3] if len(f) > 3 else "-"; deeper = f[4] if len(f) > 4 else "0"
    tdir = f"{DS}/{t}/{a.profile}"
    kept = [e for e in glob.glob(f"{tdir}/episode_*_cartesian_raw") if os.path.exists(e + "/metadata.json")]
    if a.preclaim:
        names = [os.path.basename(e) for e in kept] + dropped.get(t, [])
        names += [os.path.basename(q) for q in glob.glob(f"{DS}/_quarantine/*/{t}/{a.profile}/episode_*")]
        os.makedirs(f"{tdir}/.idx", exist_ok=True)
        for i in {int(n.split("_")[1]) for n in names}:
            os.makedirs(f"{tdir}/.idx/{i:06d}", exist_ok=True)
    ok = att = 0
    for log in glob.glob(f"{tdir}/attempts*.jsonl"):
        for l in open(log):
            att += 1; ok += '"success": true' in l
    p = max(ok / att if att else 0.5, 0.2) * a.keep_rate
    need = a.target - len(kept)
    if need <= 0:
        continue
    rounds = min(300, math.ceil(need / (a.nenv * p) * 1.5) + 2)
    out.append((need / p, f"{t} {rounds} {exp} {it} {raise_m} {deeper}", t, len(kept)))
out.sort(key=lambda r: -r[0])
open(a.out, "w").write("".join(r[1] + "\n" for r in out))
print(f"tasks {len(out)} | kept {sum(r[3] for r in out)} | still needed {sum(a.target - r[3] for r in out)} | first {[r[2] for r in out[:5]]} | -> {a.out}")
