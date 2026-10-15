# AutoMate single-arm pipeline: train, evaluate, collect

Scripts used to train the per-assembly AutoMate insertion policies in the table scene and to evaluate them. Settings
come from the environment: `SHARED_ROOT` (shared disk: `outputs/` holds the Isaac Lab logs, queues and evaluations; every
host's `$ISAACLAB/logs` must link to `$SHARED_ROOT/outputs/isaaclab_logs`), `TRAJ_ROOT` (disassembly reference
trajectories), and an `activate.sh` per host that sets up Isaac Lab (`ISAACLAB`). `REPO_ROOT` defaults to this checkout.

## Per-task settings

`roboassemblybench/library/automate/task_variants.json` holds everything that differs between assemblies:
`grasp` (replaces plug_grasps.json), `max_tilt_deg` (gripper roll / pitch unlock), `training.stages` (iterations and
hydra overrides per stage) and `policy` (the checkpoint with the best formal eval, used for collection). The env reads
the table itself and logs `[TASK_VARIANT] <task> ...`; tasks without an entry are the plain baseline.

| variant | tasks | what |
|---|---|---|
| baseline | 77 | one stage of 1000 iterations |
| continued | 10 | 1000, then resumed to 2000 (the curriculum restarts at level 0 on resume) |
| grasp | 6 | reselected grasp (finger-socket / finger-table collision, in-hand slip or spin) |
| tilt | 4 | gripper roll / pitch unlocked (plug held > 1 deg tilted) |
| grasp + tilt | 1 | 00755 |
| thresholds | 1 | 00210: 1000, 1500, then 2500 with curriculum thresholds 0.6 / 0.4 |
| socket base | 1 | 00471 (the fix is part of the baseline env) |

## Training

    QUEUE=<name> TASKS="00004 00007 ..." [TAG=evalsig DATE=YYYYMMDD NENV=128 MPS=1] train_queue.sh <activate.sh> <GPU>

`train_queue.sh` claims tasks (atomic mkdir under `outputs/queues/<name>/claims`) and runs `train_task.sh`, which runs
the task's stages through `train_resume.sh` (resumes from the newest checkpoint after a crash; `EXTRA_HYDRA` appends
hydra overrides). With `MPS=1` several workers share a GPU through NVIDIA MPS (5 per L20 at 128 envs).

## Evaluation

    TASKS="..." EXPFMT=<date>_%s_<tag> OUTD=<dir> [QUEUE=... N=1 IT=1000] eval_rolling.sh <activate.sh> <GPU>

Evaluates each task as soon as its final checkpoint exists (100 episodes, 50 envs, curriculum off, deterministic,
`rollout_logger.py`), retries starts that fail (e.g. out of GPU memory). Result: `outputs/<dir>/<task>/summary.txt`
with `strict_final(<3mm)`.

## Collection

The task config of the data pipeline (`<task> <experiment> <iteration> [<raise>] [<deeper_ok>]`) comes from
`collect_lists.py task_variants.json > tasks.txt`; see `../data_pipeline/config.example.env` and the top-level README.
