# Problems found and fixed

Every item below was observed in real runs or real data and verified after the fix. Fixes that change what gets recorded are also written into each episode's metadata, so the rule that produced an episode can be read back from the episode itself.

## Physics and control

| problem | how it showed | fix |
|---|---|---|
| **Zero action is not a hold.** The AutoMate controller re-anchors its target to the current fingertip every step, so a zero action lets the arm sink. | After insertion, the end effector sank a median of 1.7 mm (up to 5.8 mm) within 1 s; 29 % of inserted attempts were lost during the "hold". | Lock the target pose when the hold starts. The hold only counts as complete after 30 consecutive frames with < 0.1 mm motion. |
| **Lock-step scripted phases.** All envs had to reach a waypoint before any env moved on. | One env that could not reach its waypoint (arm near full extension) kept the other 15 waiting for up to 20 s per waypoint, and the waiting was recorded on camera. This happened in 22 % of rounds. | Per-env stall rule: an env whose normalized error stops decreasing is released (far from the waypoint: progress relative to the max step; near it: 0.5 mm / 0.005 rad per second). An env that stalls > 1 cm / 0.1 rad away fails as `servo_stalled`. QC rejects over-long scripted phases. |
| **Descent overshoot pushed the fingers into the table.** | The descent jumped the target to the goal 18 mm out while the arm was still moving at ~90 mm/s, overshooting 2.6–5 mm; grasps less than ~5 mm above the table hit it at 20–140 N. | Soft landing: on the last stretch each control step covers at most 30 % of the remaining error. Overshoot dropped to 0.5–0.9 mm. A contact gate also rejects episodes whose fingers press up on anything by > 3 N net during approach / grasp (`finger_pressed_table`). |
| **Failed envs dragged the carry.** | An env that had already lost its plug computed a high carry target, which lifted every env with it (carry 400–660 frames). | Failed envs hold still during the carry; the cruise height uses healthy envs only. |
| **Grasp at table level.** | For some assemblies the nominal AutoMate grasp sits 0.2–0.5 mm above the table, so the fingers cannot close without touching it. | Raise the grasp 4–7 mm along the fingertip approach axis, per task, for both training and collection; the amount is recorded as `grasp_raise_m`. |
| **No physical stop at the goal.** | For three assemblies the plug kept sliding in after the release and failed the keypoint rule, although it was correctly seated. | `deeper_ok` success rule for those tasks only (see README), recorded per episode as `success_rule`. |
| **Changing `render_interval` changes the physics.** | `sim.step(render=True)` advances `dt × render_interval`; with an interval of 4, the reset ran 4× too fast and the policy failed (0 / 32 vs 20 / 32). | Keep `render_interval = 1` and call `sim.render()` explicitly only on recorded frames. |

## Rendering and scene

| problem | how it showed | fix |
|---|---|---|
| **Neighbouring envs in view.** | With 2 m env spacing, the front camera saw the next env's table and objects in the background. | 12 m spacing, beyond the cameras' 10 m far plane. Scene props are placed only where no other env's camera can see them. |
| **One key light for the whole stage.** | The only sphere light sat near env 0; other envs were barely lit, yet their metadata recorded a light position. | One key light per env, positioned in env-local coordinates. |
| **The key light never moved.** | `XformCommonAPI.SetTranslate` failed silently on the spawned light (incompatible xform ops), so every light stayed at its spawn position. | Set the `xformOp:translate` attribute directly and read it back; a mismatch raises. |
| **Lighting not settled before the first frame.** | After a light change, the RTX image keeps brightening for ~275 renders. | Re-render until the mean image is stable; rounds that never settle are rejected (`visuals_not_settled`). |
| **Category imbalance in the randomization.** | Distractor type followed slot parity (box : cylinder = 60 : 40, a single distractor was always a box), and scene props were uniform over assets, so cardboard boxes appeared 4× as often as single-asset categories. | Independent 50 / 50 type per distractor; props pick the category first, then an asset. Both versions are documented, and every episode records its version. |
| **Overexposed frames.** | With no auto exposure, a strong dome light plus a close key light washed pale tables out (10.7 % of episodes). | First-frame table-region filter (median ≥ 239 or ≥ 3 % clipped pixels → quarantine), calibrated on a human review of rendered samples; the collectors top up the removed episodes. |

## Data integrity

| problem | how it showed | fix |
|---|---|---|
| **Seed collisions across runs and profiles.** | An early batch reused seeds across visual profiles and had to be regenerated. | Global seed ledger: each run claims its seed with an atomic `mkdir`. Layout seeds derive from it, so no two episodes share a layout. Duplicate checks run in QC and in the audit. |
| **Episode names reused after deletion.** | The next index was "highest existing + 1", so trimming the highest indices would let a later run reuse a deleted name for different content. | Every index ever used keeps a permanent claim (`<task>/mixed/.idx/<idx>`). |
| **Duplicate and missing frames (lessons from a previous batch).** | An earlier LeRobot batch contained episodes written twice after a disk-full resume, and frames silently dropped by a streaming encoder. | Content-hash audit of every episode; each kept episode must match exactly one success record (index + layout seed). Videos are concatenated with `-c copy` (no streaming encoder); the exporter checks frame counts against trajectory lengths and compares each episode's first frame pixel by pixel with the raw video. |
| **Asset server failures.** | The S3 content server intermittently failed to load a prop or the ground plane. | Detect asset-load failures in the run log, quarantine that run's episodes, retry with the next seed. |
| **Kit hangs on NFS.** | Isaac Kit holds POSIX locks in its portable root; on NFS v3 the unlock never returns and the process hangs forever at exit. | Portable root on local disk / `/dev/shm`; forced exit after 120 s. |
| **Silent physics failure on a shared GPU.** | When PhysX could not allocate GPU memory, one task produced 0 / 96 successes and still "finished". | Never share a GPU with training jobs; collectors are started only on GPUs with enough free memory; every run must end with a summary line, or it is retried. |
| **Exporter metadata gaps** (found in an independent review). | `batch_id` was a directory name; the `deeper_ok` rule was not exported, so some episodes contradicted the exported success definition; the depth index path and filter name were wrong; camera poses were not fixed-size lists; some extra features lacked names and units. | All fixed; the external validator passes on every dataset. |
| **Wrist mount calibration.** | The wrist camera mount differs by 1–2 µm between episodes (simulation noise), which broke a "one calibration" assertion. | One calibration ID per distinct mount content. |

## Operations

| problem | fix |
|---|---|
| `ffmpeg` inside `while read` consumed the loop's stdin, so each worker processed one task and stopped. | All long-running loops iterate with `for` or redirect `< /dev/null`. |
| Directories created by one host were not writable by another host running as a different uid. | Create shared directories with `umask 0` / mode 777. |
| The exporter's structural check decoded start frames one episode at a time (quadratic in dataset size). | One decoding pass per video file. |
