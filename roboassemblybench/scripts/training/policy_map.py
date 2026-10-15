"""Per-task policy for collection: the checkpoint with the best formal eval across the evalsig tracks (ties: main
queue first). Writes policy_map.json, tilt_unlock_collect.json (only tasks whose chosen policy was trained with the
tilt unlock), excluded.json and the collection lists (task, max rounds, exp, iteration).
usage: policy_map.py <out dir>"""
import json, os, re, sys
from collections import Counter

W = os.environ["SHARED_ROOT"]; O = sys.argv[1]
tracks = [("main", "eval_evalsig", "20260929_%s_evalsig", 1000), ("2k", "eval_evalsig_2k", "20260929_%s_evalsig_2k", 2000),
          ("grasp", "eval_evalsig_grasp", "20260929_%s_evalsig_grasp", 1000), ("tilt", "eval_evalsig_tilt", "20260929_%s_evalsig_tilt", 1000),
          ("sockbase", "eval_evalsig_sockbase", "20260929_%s_evalsig_sockbase", 1000),
          ("tiltgrasp", "eval_evalsig_tiltgrasp", "20260930_%s_evalsig_tiltgrasp", 1000),
          ("thr", "eval_evalsig_thr", "20260930_%s_evalsig_thr", 2500)]  # lower curriculum thresholds: training only
GRASP_TRACKS, TILT_TRACKS = {"grasp", "tiltgrasp"}, {"tilt", "tiltgrasp"}
order = {t[0]: k for k, t in enumerate(tracks)}
ids = open(f"{W}/scripts/all100_list.txt").read().split()
pending = set(os.environ.get("PENDING", "").split())
grasp_over = set(json.load(open(f"{W}/data/automate_assets/AutoMate_grasp_20260929/grasp_overrides.json")))
pm, excl = {}, {}
for t in ids:
    cands = []
    for name, d, fmt, it in tracks:
        f = f"{W}/outputs/{d}/{t}/summary.txt"
        if os.path.exists(f):
            m = re.search(r"strict_final\(<3mm\)=([0-9.]+)", open(f).read())
            if m:
                cands.append((float(m.group(1)), -order[name], name, fmt % t, it))
    if not cands:
        excl[t] = "no eval"; continue
    cands.sort(reverse=True)
    ev, _, name, exp, it = cands[0]
    if t in pending:
        excl[t] = "fix in progress (best so far %.2f %s)" % (ev, name); continue
    if ev < 0.8:
        excl[t] = "best eval %.2f < 0.8" % ev; continue
    if (name in GRASP_TRACKS) != (t in grasp_over):
        excl[t] = "grasp mismatch: chosen %s, override %s" % (name, t in grasp_over); continue
    nn = f"{W}/outputs/isaaclab_logs/rl_games/Assembly/{exp}/nn"
    ck = [x for x in os.listdir(nn) if x.startswith(f"last_Assembly_ep_{it}_") and "cpu_resume" not in x]
    pm[t] = dict(track=name, exp=exp, it=it, eval=ev, tilt=(name in TILT_TRACKS), grasp_override=(t in grasp_over), ckpt=ck[0] if ck else None)
tilt = {t: 10.0 for t, v in pm.items() if v["tilt"]}
json.dump(pm, open(f"{O}/policy_map.json", "w"), indent=1)
json.dump(tilt, open(f"{O}/tilt_unlock_collect.json", "w"), indent=1)
json.dump(excl, open(f"{O}/excluded.json", "w"), indent=1)
print("collect", len(pm), dict(Counter(v["track"] for v in pm.values())), "| tilt", sorted(tilt),
      "| grasp", sorted(t for t, v in pm.items() if v["grasp_override"]))
print("excluded", excl)
print("missing ckpt", [t for t, v in pm.items() if not v["ckpt"]])
for fn, R in (("list_mixed.txt", 45), ("list_single.txt", 10)):
    rows = sorted(pm.items(), key=lambda kv: -kv[1]["eval"])
    open(f"{O}/{fn}", "w").write("".join("%s %d %s %d\n" % (t, R, v["exp"], v["it"]) for t, v in rows))
