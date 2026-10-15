"""Overexposure filter. In the first recorded frame of the front video (arm at home), table region rows 300-480 / cols
120-520 of 640 x 480, luma as decoded by ffmpeg (gray): an episode is overexposed if the median is >= MED_MAX (table close to
white) or at least CLIP_FRAC of the pixels are >= 250 (a clipped light spot). The thresholds were set from a human review of
rendered first frames. Overexposed episodes are MOVED (not deleted) to <dataset root>/_quarantine/overexposed/<task>/<profile>/;
their index claims stay. Every scanned episode is appended to the scan log, so a rerun only scans new episodes.
usage: overexposure_filter.py --dataset-root D [--profile mixed --log FILE --tasks T1,T2 --workers 32 --dry-run]"""
import argparse, concurrent.futures as cf, glob, json, os, subprocess, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-root", required=True); ap.add_argument("--profile", default="mixed")
ap.add_argument("--log", help="scan log (default <dataset root>/_queue/overexposure_scan.jsonl)")
ap.add_argument("--tasks", help="comma-separated subset of tasks"); ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
DS = a.dataset_root; Q = f"{DS}/_quarantine/overexposed"; LOG = a.log or f"{DS}/_queue/overexposure_scan.jsonl"
MED_MAX, CLIP_FRAC = 239, 0.03; DRY = a.dry_run


def metric(e):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{e}/videos/observation_images_front.mp4", "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
    t = np.frombuffer(out, np.uint8).reshape(480, 640)[300:480, 120:520]
    return float(np.median(t)), float((t >= 250).mean())

done = set()
if os.path.exists(LOG):
    done = {json.loads(l)["ep"] for l in open(LOG)}
pattern = [f"{DS}/{t}/{a.profile}/episode_*_cartesian_raw" for t in a.tasks.split(",")] if a.tasks else [f"{DS}/[0-9]*/{a.profile}/episode_*_cartesian_raw"]
eps = [e for p in pattern for e in glob.glob(p) if os.path.exists(e + "/metadata.json") and os.path.relpath(e, DS) not in done]
with cf.ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(lambda e: (e, *metric(e)), eps))
moved = 0
if not DRY:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(os.devnull if DRY else LOG, "a") as log:
    for e, med, clip in res:
        bad = med >= MED_MAX or clip >= CLIP_FRAC
        rel = os.path.relpath(e, DS); t = rel.split("/")[0]
        if bad and not DRY:
            os.makedirs(f"{Q}/{t}/{a.profile}", exist_ok=True); os.rename(e, f"{Q}/{t}/{a.profile}/{os.path.basename(e)}"); moved += 1
        if not DRY:
            log.write(json.dumps({"ep": rel, "table_median": med, "table_clipped": clip, "overexposed": bad, "time": time.strftime("%F %T")}) + "\n")
print(f"{time.strftime('%F %T')} scanned {len(res)} | overexposed {sum(1 for _, m, c in res if m >= MED_MAX or c >= CLIP_FRAC)} | moved {moved}{' (dry run)' if DRY else ''}")
