"""Trim every task of a dataset root to exactly --target kept episodes by deleting a seeded random choice of the surplus
(seed = int(task) * 10 + --seed-offset). Protected episodes are never candidates: those whose metadata.json is older than
--protect-before (e.g. a part of the data already delivered) and those listed in --protect-list (lines `<task> <episode dir>`).
The manifest of deleted episodes is written to <dataset root>/_manifests/trim_<target>_manifest_<time>.json first; their
index claims (.idx) stay, so the names are never reused. Dry run unless --apply.
usage: trim_to_target.py --dataset-root D [--target 1200 --profile mixed --protect-before 2026-10-10T15:30 --protect-list F
                         --seed-offset 1 --apply]"""
import argparse, datetime, glob, json, os, random, shutil, time

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-root", required=True); ap.add_argument("--target", type=int, default=1200); ap.add_argument("--profile", default="mixed")
ap.add_argument("--protect-before", help="ISO local time; episodes written earlier are kept"); ap.add_argument("--protect-list")
ap.add_argument("--seed-offset", type=int, default=1); ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
DS = a.dataset_root; TARGET = a.target
t_protect = datetime.datetime.fromisoformat(a.protect_before).timestamp() if a.protect_before else None
protected = set()
if a.protect_list:
    protected = {tuple(l.split()[:2]) for l in open(a.protect_list) if l.strip()}
man = {"deleted_at": time.strftime("%F %T"), "rule": f"seeded random surplus over {TARGET} per task, seed = int(task) * 10 + {a.seed_offset}, "
       f"protected: written before {a.protect_before or '-'}, listed in {a.protect_list or '-'}", "tasks": {}}
short = []
for t in sorted(x for x in os.listdir(DS) if x.isdigit() and len(x) == 5):
    tdir = f"{DS}/{t}/{a.profile}"
    eps = sorted(os.path.basename(e) for e in glob.glob(f"{tdir}/episode_*_cartesian_raw") if os.path.exists(e + "/metadata.json"))
    n = len(eps)
    if n < TARGET:
        short.append((t, n))
    drop = []
    if n > TARGET:
        cand = [e for e in eps if (t, e) not in protected and not (t_protect and os.path.getmtime(f"{tdir}/{e}/metadata.json") < t_protect)]
        assert len(cand) >= n - TARGET, f"{t}: {n} kept, only {len(cand)} unprotected"
        drop = sorted(random.Random(int(t) * 10 + a.seed_offset).sample(cand, n - TARGET))
    man["tasks"][t] = {"before": n, "dropped": drop, "tdir": tdir}
print("tasks", len(man["tasks"]), "| short of target", short, "| tasks trimmed", sum(1 for v in man["tasks"].values() if v["dropped"]),
      "| episodes to delete", sum(len(v["dropped"]) for v in man["tasks"].values()))
if a.apply:
    assert not short, f"tasks below {TARGET}: {short}"
    os.makedirs(f"{DS}/_manifests", exist_ok=True)
    mp = f"{DS}/_manifests/trim_{TARGET}_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(man, open(mp, "w"), indent=1); print("manifest", mp)
    for t, v in man["tasks"].items():
        for e in v["dropped"]:
            shutil.rmtree(f"{v['tdir']}/{e}")
    left = {t: len(glob.glob(f"{v['tdir']}/episode_*_cartesian_raw")) for t, v in man["tasks"].items()}
    print("after:", "all", TARGET if all(x == TARGET for x in left.values()) else {t: x for t, x in left.items() if x != TARGET})
