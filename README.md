# automate-datagen

**Single-arm assembly data pipeline on Isaac Lab AutoMate.**

<p align="center"><img src="docs/media/demo_00768.gif" width="760" alt="one episode, front camera (left) and wrist camera (right)"></p>
<p align="center"><sub>One episode of task 00768: front D435 (left) and wrist D405 (right), played at 2× speed.
Full-resolution videos: <a href="docs/media/demo_00768.mp4">00768</a> · <a href="docs/media/demo_00016.mp4">00016</a> · <a href="docs/media/demo_00293.mp4">00293</a></sub></p>

This repository contains a simulation data pipeline for **full-process robotic assembly with a single Franka Panda arm**, built on NVIDIA Isaac Lab's [AutoMate](https://github.com/isaac-sim/IsaacLab) plug–socket tasks. Every episode starts at a fixed home pose, picks the plug up from the table, carries it to the socket, inserts it with a learned PPO policy, releases it, and returns home. The episode is recorded from two RealSense-like cameras with RGB, metric depth and per-frame metadata, then exported to **LeRobot v3**.

The pipeline produced `automate_fullproc_20261004`:

| | |
|---|---|
| Robot | Franka Panda + parallel gripper (`robot_type = franka_panda_single_arm`) |
| Tasks | 85 AutoMate assemblies, one dataset per task |
| Size | 85 × 1,200 = **102,000 successful episodes**, 77.6 M frames |
| Duration | 30 fps, 25.4 s per episode on average; **719 h per camera** (1,438 h for both cameras) |
| Cameras | front D435 (fixed in the scene) and wrist D405 (on `panda_hand`), each 640×480 RGB + depth |
| Format | LeRobot v3 + an evaluator diversity extension (`ourdata-diversity-1.1`): calibrations, per-episode environments, camera poses |
| Checks | full QC of all 102,000 raw episodes, content-hash deduplication, structural check of every export, external validator pass on 85 / 85 datasets |

The full dataset (5.1 TB) is not hosted here; this repository holds the code that generated it, plus a small sample to inspect ([`sample/`](sample), see below). The code is published together with an honest list of the problems we found and fixed ([`docs/FIXES.md`](docs/FIXES.md)) and of the problems that remain ([`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)).

---

## What an episode looks like

| phase | name | controller | median frames |
|---:|---|---|---:|
| 0 | approach: move above the plug, soft-landing descent to the grasp | script | 259 |
| 1 | grasp | script | 66 |
| 2 | lift | script | 54 |
| 3 | carry: move above the socket and align | script | 188 |
| 4 | insert | **per-task PPO policy** (rl_games) | 53 |
| 5 | release | script | 15 |
| 6 | return home | script | 121 |

The plug and the socket both stand on the table. There is no attachment or suction assist (`attachment_assist = false`), and plug gravity is on. The socket is placed within ±5 cm / 0–10° of its training pose. The plug is placed uniformly in x ∈ [0.40, 0.75] m and y ∈ [−0.32, 0.32] m (env frame), at least 15 cm from the socket, with yaw ±45°.

**Success** (only successful episodes are kept):

- `keypoint`: the target is reached (mean keypoint error < 3 mm), the gripper opens, the arm returns home, and the error is still < 3 mm at the last frame.
- `deeper_ok`: used for assemblies whose AutoMate goal pose has no physical stop, where the plug slides further in after the release. The target must be reached before the release. At the last frame the plug base may be at most 3 mm out of the goal (any depth further in), at most 3 mm off laterally and tilted at most 5°.

The rule applied to each episode is recorded per episode.

## Repository layout

```
roboassemblybench/
  library/automate/            Isaac Lab AutoMate v2.3.2 (vendored, BSD-3) + our environments
    assembly_env.py            insertion env (relative observations, grasp raise, per-task variants)
    pick_env.py                scripted pick / carry on top of the insertion env
    assembly_collect_env.py    collection env: table layout, cameras, contact sensors, visual randomization hooks,
                               stall rule, soft landing, carry hold, per-frame capture
    task_variants.json         per-assembly grasp / tilt / training-stage / policy settings
  datasets/cartesian_episode.py    episode recorder (trajectory npz, H.264 RGB, chunked zstd depth, annotations)
  scripts/
    collect_automate_single_arm.py collector: rounds of N parallel envs, success rules, visual randomizers,
                                   metadata, parallel runs per task
    training/                      per-task PPO training (stage-wise, resumable), rolling evaluation
    data_pipeline/                 collection queue / workers, overexposure filter, trimming, audit, QC,
                                   LeRobot v3 exporter, finalize / status, depth reader, config.example.env
examples/tasks_automate_fullproc_20261004.txt   the task config used for the released dataset (85 tasks)
sample/                            5 tasks x 5 episodes in the exported format (RGB only, see below)
docs/FIXES.md                      problems found and fixed, with the evidence
docs/KNOWN_ISSUES.md               problems that remain
```

## Pipeline

```
train one PPO insertion policy per assembly        scripts/automate_pipeline (train_queue.sh, eval_rolling.sh)
        │
collect: run_worker.sh → collect_queue.sh → collect_automate_single_arm.py
        │   16 envs per process, 4 processes per 46 GB GPU, up to 3 processes on the same task
        │   global seed ledger: no seed or layout is ever reused; episode indices are claimed permanently
        │
overexposure filter (every 3 min) → washed-out episodes moved to quarantine → collectors top up
        │
finalize.sh: leftovers to quarantine → last overexposure pass → top up short tasks → trim to exactly N per task
        │   → duplicate / success-record audit → full QC
        │
export_queue.sh: export_lerobot_v3.py per task → structural self-check → external validator (optional)
```

Nothing is deleted silently. Rejected episodes are **moved** to `_quarantine/<reason>/` under the dataset root. The only deletion is the trim to a whole number per task, which writes its manifest first.

### Visual randomization (`mixed` profile)

| part | what |
|---|---|
| lighting | dome light 2000 × U(0.35, 1.8), tint 0.75 ± 0.15; one sphere key light per env, U(0, 4e5), radius U(0.05, 0.25) m, position x ∈ [0.2, 1.0], y ∈ [−0.6, 0.6], z ∈ [0.6, 1.5] m |
| distractors | 0–4 visual-only boxes / cylinders on the table |
| surfaces | table: 50 % palette colour (6) / 50 % texture (4); floor texture |
| scene props | 3–6 Isaac Simple_Warehouse props 2–6 m in front of the front camera, never visible from another env |

Each part draws from its own random stream (`[run seed, part, round, env]`). Every applied parameter is written into the episode metadata. The dataset contains two versions of the distractor / prop sampling (`v1_20261004`, `v2_20261010`), and each episode records its version. Details are in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

### Exported format

`<task>/mixed/` is a complete LeRobot v3 dataset:

- **`observation.state`**: end-effector pose (xyz + quaternion wxyz) plus gripper opening.
- **`action`**: the next frame's absolute pose plus the gripper target.
- **`policy_action`**: the raw insertion-policy output.
- **`phase_index`**: the episode phase.
- **Joint state / velocity / effort**.
- **`observation.contact_force`**: finger and plug contact forces.
- **`observation.plug_pose` / `observation.socket_pose`** and **`observation.keypoint_error`**.
- **`observation.images.front` / `observation.images.wrist`**: two H.264 videos, made by concatenating the raw per-episode streams, with no re-encoding.

Depth is a sidecar (`sensors/depth/`): uint16 mm, chunked zstd with 16-bit bit-plane shuffling; `scripts/data_pipeline/read_depth.py` is a standalone reader. `meta/diversity/` holds the schema, the camera calibrations (mount extrinsics), the per-episode environments, the per-episode generation records and the per-frame camera poses.

## Sample

[`sample/franka_panda_single_arm/<task>/mixed/`](sample/franka_panda_single_arm) holds 5 episodes each of 5 tasks (25 episodes, 28 MB). Each task is a complete LeRobot v3 dataset with `meta/diversity/`, made by the same exporter as the release. To keep the repository small, the sample is exported with `--no-depth`: **it has no depth sidecar** (`schema.json` says `depth_present: false`), while the full dataset carries depth for every frame. The 5 tasks cover the default success rule (00016, 00293, 00768), the `deeper_ok` rule (00007) and a raised grasp (00186), and both randomization versions. All 5 pass the structural check and the external validator.

## Running it

Requirements: Isaac Sim 5.1, Isaac Lab 2.3.2 (Python 3.11), rl_games; pyarrow ≥ 17 for the exporter; ffmpeg. The AutoMate assets are downloaded with `roboassemblybench/library/automate/download_assets.py`. Visual assets (Fabrica textures and Isaac Simple_Warehouse props, each listed in a `manifest.json`) are read from `AUTOMATE_VISUAL_ASSETS`.

Nothing is hard-coded to a machine. The scripts read their settings from the environment; see [`roboassemblybench/scripts/data_pipeline/config.example.env`](roboassemblybench/scripts/data_pipeline/config.example.env). Every shell script runs the checkout it lives in. Scratch files go to `$TMPDIR`.

```bash
cp roboassemblybench/scripts/data_pipeline/config.example.env config.env   # edit the paths
set -a; source config.env; set +a
P=roboassemblybench/scripts/data_pipeline

# 1. task list for this pass (longest expected collection first, runs sized from success rates)
python3 $P/build_task_list.py --dataset-root $DATASET_ROOT --tasks $TASK_CONFIG --out $DATASET_ROOT/_queue/task_list_$QTAG.txt --preclaim
# 2. one or more collection workers per GPU (<activate.sh> sets up Isaac Lab and defines ISAACLAB)
bash $P/run_worker.sh <activate.sh> <GPU> [<kit portable-root seed dir>]
bash $P/overexposure_filter_loop.sh &          # optional: filter while collecting
# 3. finalize (top-up list if a task is short), then export on any number of hosts
bash $P/finalize.sh && bash $P/export_queue.sh <worker name>
bash $P/status.sh finalize; bash $P/status.sh exports; bash $P/status.sh summary
```

A single-task smoke run without the queue:

```bash
cd $ISAACLAB && ./isaaclab.sh -p $REPO_ROOT/roboassemblybench/scripts/collect_automate_single_arm.py \
  --assembly-id 00016 --checkpoint <policy.pth> --out-dir /path/to/smoke --target-successes 8 --max-rounds 1 \
  --num_envs 8 --seed 7300016 --production-id smoke --visual-profile mixed --headless --enable_cameras --device cuda:0
```

The released dataset used exactly this code; the scripts were afterwards made machine-independent (paths moved to settings, ad-hoc analysis scripts dropped), and the cleaned versions were re-run against the dataset (task list, trim dry run, audit, QC, overexposure filter, exporter + validator) and in a collection smoke test.

## Credits

- AutoMate environments and assets: NVIDIA Isaac Lab (BSD-3-Clause), vendored at v2.3.2.
- The pipeline grew out of the RoboAssemblyBench project; the single-arm full-process pipeline, its fixes and this dataset were built by Markwcl2004.
