The machine rebooted unexpectedly during the previous MOSAIC-Rec run.

Before doing any new scientific execution, perform a crash-recovery audit.

Read completely:
- AGENTS.md
- docs/MOSAIC_REC_SPEC.md
- MASTER_PROMPT_MOSAIC_REC.md
- PLANS.md if present
- reports/PROGRESS.md
- reports/BLOCKERS.md
- reports/NEEDS_USER_ACTION.md
- reports/DECISIONS.md
- evidence/stage_status.json
- the newest reports/RECOVERY_PREFLIGHT_*.txt

Then inspect:
- git status and diffs;
- existing stage receipts;
- experiment manifests;
- protocol identities;
- checkpoints;
- hashes;
- lock files;
- partial outputs;
- WSL scratch;
- GPU state;
- logs from the most recent incomplete stage.

Recovery rules:

1. Treat any stage that was RUNNING/IN_PROGRESS at reboot as INTERRUPTED,
   not PASS.

2. Do not delete stale lock files merely because they exist.
   First verify no owning process remains and inspect what the lock protected.

3. Do not overwrite partially written experiment evidence.
   Preserve it, classify it, and either resume safely or start a new run identity.

4. If the interruption occurred during DEVELOPMENT/HPO, resume only after
   validating checkpoint/config/data identity.

5. If the interruption occurred during QUALIFICATION/FROZEN/LOCKED work,
   be substantially more conservative.

6. If protected outcomes may have been exposed before interruption, do not
   simply rerun the same locked experiment. Determine whether the protocol
   permits exact mechanical recovery. Otherwise preserve the interrupted
   protocol as terminal evidence and create the scientifically correct new
   identity according to the project rules.

7. Verify all completed outputs by hash/schema before trusting stage_status.

8. Re-run CUDA/project-environment qualification before long neural work.

9. Verify dataset downloads and partially written archives before resuming them.

10. Update PLANS.md, reports/PROGRESS.md, reports/BLOCKERS.md,
    reports/DECISIONS.md, and evidence/stage_status.json with the recovery
    determination.

Determine the last scientifically complete stage and the first safe incomplete
stage. Resume from there, not from the beginning.

After recovery, continue MOSAIC-Rec end to end until reports/FINAL_STATUS.md
exists, preserving all negative/interrupted evidence and all claim boundaries.

Use GPT-5.6 Sol Medium and the local RTX GPU for qualified neural workloads.
Do not silently fall back to long CPU-only neural training.