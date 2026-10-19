# Known issues

These problems are not solved in the released data or code. Where a workaround exists, it is given.

## In the data

- **Single arm in a dual-arm schema.** The evaluator schema has left and right slots, and the one arm is written into both (documented in `schema.json` → `state_indices`). Any quantity summed over left + right (for example a total end-effector path length) is **2×** the true single-arm value.
- **Two randomization versions.** 15,210 episodes are `v1_20261004`: distractor type by slot parity, scene props uniform over assets. 86,790 episodes are `v2_20261010`: independent distractor type, category-first props. The lighting range is identical. Filter on `episodes.jsonl` → `visual_randomization_version` if you need a single distribution.
- **Residual prop-category imbalance in v2.** Within an env, assets are drawn without replacement. Categories with several assets (cardboard box, pallet, barrel) therefore stay in the draw after the single-asset categories are used, and appear about 1.3× as often.
- **White table texture.** One of the four table textures is white painted metal (12.5 % of episodes). The washed-out ones are filtered; the rest are still bright under strong light. Darkening the texture had little visible effect at the bright end of the lighting range, so it was left unchanged.
- **The overexposure filter measures table brightness, not light intensity.** A dark table under a very strong key light passes the filter, although it is lit far more than a human reviewer's preferred range. A full-resolution review of such cases found them acceptable, so the data keeps the original lighting range.
- **The renderer has no auto exposure.** About 11 % of collected episodes were rejected as overexposed, a pure compute loss. Sampling the lighting jointly with the table look (or an exposure model) would avoid it.
- **The policy was trained without plug gravity.** Upstream AutoMate disables gravity on the held plug during training, and the insertion policies inherit that. Collection runs with gravity on, so only successful episodes are kept, but the policy's success rate is lower than it could be.
- **15 of 100 assemblies are excluded.** 8 have plugs that tip over when standing on the table; 4 cannot be grasped from the table even with a raised grasp (one was retrained without success); 1 has a cavity under the goal, so the plug cannot stay seated; 2 stayed below 60 % policy success at the training limit.
- **Instruction diversity is capped by the task family.** Every task puts a plug into or onto a socket. Labels per task (2 actions, 55 action + plug + socket combinations, 5.36 bit) are in `examples/instruction_labels_*.json`; the released export still uses the single instruction "Insert the plug into the socket." until it is re-exported with these labels.
- **Joint commands are not recorded.** The controller outputs joint torques, so only measured joint states are available.
- **An earlier delivered 10 × 10 sample contains 18 overexposed episodes.** It was made before the filter existed, and none of those episodes are in the final dataset.

## In the code

- **Built for a shared-disk cluster.** Paths are settings now, but the design assumes one NFS root shared by every host (atomic `mkdir` claims, seed ledger, queue state), Isaac Kit portable roots on local disk, and GPUs not shared with other jobs.
- **Not packaged.** There is no `pyproject.toml` or install step for this subset; set `PYTHONPATH` manually. Only the vendored AutoMate assets have a download script.
- **No daemon chains the stages.** Collection, `finalize.sh` and the export queue are started separately (by hand or by a watcher). In our run the watching session ended, and a finished collection waited ~10 h before the export started.
- **Lock-step rounds waste time.** A round of 16 envs lasts as long as its slowest episode (median 797 frames for the longest vs 757 for a typical episode), plus ~100 settle renders after each lighting change.
- **The external validator does not decode full videos.** Video integrity relies on our exporter's structural check (frame counts, first frame of every episode).
- **One host's export workers never started** in the final export (probably a permission issue on the shared queue directory, not investigated); the other two hosts exported all 85 datasets.
- **Custom depth container.** Depth is stored as a sidecar with its own chunked format rather than as a standard LeRobot feature, so readers need `read_depth.py` or an equivalent.
