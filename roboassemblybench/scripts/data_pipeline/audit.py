"""Dataset audit against failure modes seen in an earlier batch (episodes written twice, missing records):
1. the same episode written twice / physical copies: hash of the numeric content (state, action, plug pose, socket pose,
   joint state) of every kept episode; duplicates within a task and across tasks
2. kept episodes vs success records: per task, every kept episode has exactly one success record (attempts*.jsonl) with
   its index and layout seed, and every other success record belongs to an episode deleted by a trim or quarantined
3. leftovers: .incomplete dirs, FAILED_ dirs inside the dataset
Removed = listed in a trim manifest (<root>/_manifests/trim_*_manifest_*.json, plus --manifests) or present under
<root>/_quarantine.
usage: audit.py <dataset root> [--profile mixed] [--manifests "<glob>"]"""
import collections, glob, hashlib, json, os, sys
import argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--profile", default="mixed"); ap.add_argument("--manifests", default="")
a = ap.parse_args(); R = a.root; P = a.profile
hashes = collections.defaultdict(list); per_task = {}
REMOVED = collections.defaultdict(list)  # task -> episode dir names deleted by a trim or quarantined
for mp in glob.glob(f"{R}/_manifests/trim_*_manifest_*.json") + (glob.glob(a.manifests) if a.manifests else []):
    for t, v in json.load(open(mp))["tasks"].items():
        REMOVED[t] += v["dropped"]
for q in glob.glob(f"{R}/_quarantine/*/[0-9]*/{P}/episode_*") + glob.glob(f"{R}/_quarantine/*/[0-9]*/episode_*"):
    REMOVED[[x for x in q.split("/") if x.isdigit() and len(x) == 5][-1]].append(os.path.basename(q))
for tdir in sorted(glob.glob(f"{R}/[0-9]*/{P}")):
    t = tdir.split("/")[-2]
    eps = sorted(glob.glob(f"{tdir}/episode_*_cartesian_raw"))
    for ep in eps:
        z = np.load(f"{ep}/trajectory.npz")
        h = hashlib.sha256()
        for k in ("observation_state", "action", "plug_pose", "socket_pose", "joint_state"):
            h.update(np.ascontiguousarray(z[k]).tobytes())
        hashes[h.hexdigest()].append(f"{t}/{os.path.basename(ep)}")
    # success records matched to kept episodes by (episode index, layout seed): an index can belong to an episode that was
    # deleted / quarantined and then reused by a later run (01092, the tasks restarted after pre_stall_fix_20261004)
    recs = collections.Counter()
    for f in glob.glob(f"{tdir}/attempts*.jsonl"):
        for l in open(f):
            a = json.loads(l)
            if a.get("success"):
                recs[(int(a["episode_idx"]), a.get("layout_seed"))] += 1
    kept = {int(os.path.basename(e).split("_")[1]): json.load(open(f"{e}/metadata.json")).get("layout_seed") for e in eps}
    removed = {int(n.split("_")[1]) for n in REMOVED.get(t, [])}
    matched = {k for k in recs if kept.get(k[0]) == k[1]}
    per_task[t] = dict(kept=len(eps), success_records=sum(recs.values()), matched=len(matched),
                       kept_without_record=sum(1 for i, ls in kept.items() if (i, ls) not in recs),
                       records_without_dir=sum(1 for k in recs if k not in matched and k[0] not in removed),
                       multi_records=sum(1 for v in recs.values() if v > 1),
                       incomplete=len(glob.glob(f"{tdir}/*.incomplete")), failed_dirs=len(glob.glob(f"{tdir}/FAILED_*")))
dups = {h: v for h, v in hashes.items() if len(v) > 1}
print("kept episodes", sum(len(v) for v in hashes.values()), "| distinct content hashes", len(hashes), "| duplicate groups", len(dups))
for h, v in list(dups.items())[:10]:
    print("  DUP", v)
bad = {t: d for t, d in per_task.items() if d["kept"] != d["matched"] or d["kept_without_record"] or d["records_without_dir"] or d["multi_records"]}
print("tasks", len(per_task), "| tasks with kept != success records:", len(bad))
for t, d in list(bad.items())[:20]:
    print("  ", t, d)
print("incomplete dirs total", sum(d["incomplete"] for d in per_task.values()), "| FAILED_ dirs total", sum(d["failed_dirs"] for d in per_task.values()))
