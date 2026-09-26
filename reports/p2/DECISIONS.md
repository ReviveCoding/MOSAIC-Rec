# p2 decisions

## 2026-09-23 — new independent protocol

- Created `mosaic-rec-2026-09-23-p2`; p1 remains closed and immutable.
- Classified all p1 post-hoc patterns as hypothesis provenance, never p2 evidence.
- Chose PixelRec50K as primary multimodal study and MovieLens 32M as behavioral transfer only.
- Reused qualified raw source bytes and publisher feature bytes under new p2 identities; p1 receipts are not copied into p2 canonical counts.
- Kept raw-cover extraction and transformations on hold because the current PixelRec dataset license grants no modification or redistribution rights.
- Fixed PixelRec temporal boundaries from the source distribution before p2 model outcomes.
- Defined warm as FIT support greater than 20 and controlled zero support as a new deterministic p2 cohort.
- Fixed three equal-budget DEVELOPMENT trials per neural family and three new independent seeds.
- Fixed a three-seed mean-score ensemble; best-seed selection is prohibited.
- Fixed exact full-catalog protected evaluation and mandatory top-20/top-100 persistence.
- Fixed H1 superiority and H2 noninferiority as conjunctive primary gates.
- Adopted adaptive-routing terminology; reliability language is prohibited unless separately supported.

## Live-source identities

Checked 2026-09-23:

- PixelRec repository HEAD: `a4e4a01b01aba96f746ae66de9eb35a88484a681`.
- PixelRec dataset LICENSE response SHA-256: `cd3a4254bc080aee5f93468d822c9b3535adcdaaf2d6b99233f586feb4fc876a`.
- PixelRec README response SHA-256: `37f173bbe9e4d3705e2c25c54db94a0d34b6d14030f3e05507a01019845333a0`.
- MovieLens 32M README response SHA-256: `e6dcbfef34db80c937babd537b3fab2f33f4d62c904a57b7b1811f6b9e4f7353`.
- RecBole repository HEAD: `7b02be5ec80a88310f2d04a27a82adfcbb5dc211`.
- MMRec repository HEAD: `c68ee7c9b0de1ca7c5150258322edb4994be29d9`.
- Existing WSL environment passed a real CUDA operation with torch `2.14.0+cu130`, CUDA runtime `13.0`, BF16 support, and the RTX 4090 Laptop GPU.

## 2026-09-23 — PixelRec p2 data boundary

- P2-A02 attempt 1 failed before outputs because DuckDB rejected an unquoted alias; its failure receipt is preserved.
- Attempt 2 produced only the deterministic 3,013-item controlled-zero list before a second alias parse failure. The partial file and directory are preserved with their hash.
- Attempt 3 materialized all fresh p2 roles. Mechanical temporal and zero-support leakage checks passed.
- The fresh roles contain 659,790 FIT-visible, 32,995 FIT-withheld controlled, 98,175 validation, 99,827 qualification, and 98,707 locked rows.
- The p2 controlled-zero cohort contains 3,013 items. Protected model-outcome access remains zero.

## 2026-09-23 — text source amendment 001

- The official publisher text-vector file advertised 9,290,203,646 bytes but Google Drive returned a quota-exceeded response. The response is preserved and hashed.
- Selected the existing authorized local MiniLM representation derived solely from official `title`, `tag`, and `description` attributes. Engagement aggregates remain excluded.
- This resolves an explicitly open source identity before any model execution and does not change the preregistered text modality, hypothesis, role, metric, or gate semantics.

## 2026-09-23 — common candidate contract

- Candidate availability requires first-observed time at or before the query.
- All models mask the user's FIT-visible history except the target item, so repeat targets remain rankable.
- User tail preference uses the bottom 20% among positive-support FIT items; target popularity uses the complete catalog with zero support reported separately.
- Earliest eligible per-user query manifests contain 27,918 VALIDATION, 26,900 QUALIFICATION, and 25,367 LOCKED_FINAL users. These are protocol inputs, not accessed model outcomes.
- Candidate-time violations are zero in all three manifests.

## 2026-09-23 — first expert pilots

- `CSEQ_BERT4REC_PILOT` trained for three CUDA epochs and scored all 27,918 validation users against the exact time-available catalog. NDCG@20 was 0.001803 overall, 0.013161 warm, and zero on controlled and natural zero-support targets.
- `CCOLD_TWO_TOWER_PILOT` trained for three CUDA epochs on authorized image/text/tag inputs. NDCG@20 was 0.001292 overall, 0.000845 controlled zero, 0.001374 natural zero, and 0.001637 warm.
- These are DEVELOPMENT pilots, not frozen comparator results. The unfavorable low scores and expert specialization are preserved.
- The old shared p1 lock named absent PID 79333 and was treated as stale without deletion. p2 uses `/home/bjw-0/mosaic-rec-work/p2/locks/heavy_gpu.lock`; both pilots held it exclusively.

### Candidate-mask audit correction

The first C_seq and C_cold scoring attempts are invalid for model selection. The shared loader retained only the last 50 FIT events and the evaluator therefore failed to mask earlier FIT-history items for 905 validation users; maximum history was 340. The artifacts and receipts remain preserved. The loader now retains full FIT histories for masking while neural tensors still use the declared last-50 window. Both pilots require distinct reruns before their metrics are used.

## 2026-09-23 — classic validation baselines

- The first exact CPU popularity attempt was interrupted after 264 seconds before output because its zero-score ranking loop scaled poorly. The failed attempt is preserved.
- An exact offline timestamp sweep with a Fenwick prefix counter reduced attempt 2 to 12.7 seconds for global popularity and 14.3 seconds for recent popularity.
- Global popularity NDCG@20 was 0.000067 overall and 0.000491 warm. Recent 30-day popularity was 0.000452 overall and 0.003316 warm. Both scored zero on controlled and natural zero-support targets.

## 2026-09-23 — canonical invalidation semantics

- The p2 warehouse retains invalidated receipts for audit history and marks them `valid_for_selection=false` with the invalidating receipt identity.
- Metric rows from invalidated receipts are excluded from the canonical metrics table. This prevents the defective first expert scores from entering development selection while preserving their full raw receipt payloads.

## 2026-09-23 — corrected expert identities

- Corrected C_seq and C_cold reruns use distinct `*_PILOT_A2` model identities, checkpoints, outputs, and receipts. They do not overwrite the invalid first attempts.
- A PowerShell-to-Bash quoting failure was preserved as `FAILED_INFRASTRUCTURE_PRESTART`; it acquired no lock and created no model artifact.
- The router ladder now points only to the independently rehashed attempt-2 checkpoints.

## 2026-09-23 — equal-budget HPO manifest

- Froze three trials and ten scheduled epochs for each neural family under selection seed 20260930; failed scientific trials consume their budget and are not replaced.
- Expert and baseline families select by validation overall NDCG@20 with predeclared warm, cold, and identity tie breaks.
- Router families use a predeclared worst-gate development score relative to the corrected content and sequence experts, so overall quality cannot hide failure on either side of the cold/warm tradeoff.

## 2026-09-23 — reboot recovery checkpoint

- Recovery anchor was Git `49ee02aa3da271fc1a0dec3350ea8f526d142785`; no committed scientific identity changed during the interruption.
- Preserved `src/mosaic_rec/p2/slices.py` and `tests/p2/test_slices.py`. Review confirmed that they only implement the support, popularity, age, user, cross-slice, and minimum-sample rules frozen before protected access.
- R3 attempt 1 remains `BLOCKED_EXTERNAL_PRESTART`: it acquired no p2 lock and created no checkpoint, log, scored output, or metric receipt. No R3 attempt-2 artifact survived the reboot.
- The lock file's released R2 PID is historical text. A nonblocking `flock` probe passed; the GPU had no CUDA process and requalified successfully.
- Exact scientific resume point is P2-D03 R3 attempt 2 with the committed router ladder and queue. QUALIFICATION and LOCKED_FINAL counters remain zero.

## 2026-09-23 — R3 recovery result

- R3 attempt 2 remains a prestart infrastructure event. R3 attempt 3 waited without lock ownership while ACT-Agent completed training and evaluation, then passed both sustained idle checks and ran under the exclusive p2 lock.
- R3 attempt 3 used the unchanged preregistered configuration and preserved an unfavorable controlled-zero NDCG@20 of zero. Its overall and warm scores exactly match R1 at the reported precision; this is development evidence and is not a promotion result.
- The checkpoint SHA-256 is `d644f4d746ff86a99bc25c98a847f974e31700a0c620c19135fa466b5363eca6`, scored output SHA-256 is `445aa2cd8f30778d14b2f37a9c6bce74d75a3bc4b3db1056d9ac335542515b66`, and log SHA-256 is `dfb314371bd8d8172e83fbb8ba3d2dda0cb6a7e46ddf243b2e75a0787ef9f4e4`.

## 2026-09-23 — score-alignment implementation clarification 001

- Before running the preregistered comparative alignment candidates, fixed the evaluation model as the R1 hard zero-support router. This isolates the expert cross-scale transformation from learned-gate fitting.
- Made the already preregistered unit scaling executable: controlled-zero performance is divided by C_cold controlled-zero VALIDATION NDCG@20, while warm performance is expressed relative to C_seq with one unit equal to the default 10% warm margin. Their arithmetic mean is the frozen alignment selection score.
- Materialized all nine candidates already listed in `configs/p2/alignment_development.json` under distinct hashed config identities. No candidate, temperature, role, metric, population, or tie rule was added after outcome access; QUALIFICATION and LOCKED_FINAL remain unopened.

## 2026-09-23 — R4 development result

- Preserved R4 attempts 1 and 2 as external prestart contention events. Attempt 3 used the unchanged ladder and completed only after the final unrelated evaluation exited.
- The R4 warm-preservation objective produced warm NDCG@20 `0.013536823102781708` and controlled-zero NDCG@20 `0`. This did not improve the standardized-alignment cold failure seen for R1 and R3; the result remains eligible only as DEVELOPMENT evidence.

## 2026-09-23 — VBPR pilot attempt-1 OOM recovery

- Preserve the attempt-1 checkpoint and log. The failure occurred after training and before any exact-scoring output or scientific metric; it is classified as a resource failure caused by evaluator tensor materialization.
- The scorer had expanded publisher image features to batch x catalog x image dimension before projection, requesting 40.46 GiB. The validated replacement computes the same VBPR expression by projecting the catalog once and applying two matrix products. This changes evaluator memory use without changing data, model parameters, candidate semantics, metric, or score formula.
- Any retry uses attempt-2 config and separate log/checkpoint/output/receipt paths. The frozen HPO search space remains unchanged, and both protected-access counters remain zero.
- Attempt 2 passed with peak allocated VRAM 810,937,344 bytes and exact overall NDCG@20 `0.0004777958439066055`. The outcome remains DEVELOPMENT evidence and does not erase the attempt-1 OOM.
- Legacy M1 produced controlled-zero NDCG@20 `0.005334111164672065` but warm NDCG@20 `0.0008058694871134669`. Preserve this cold/warm tradeoff and proceed to the preregistered alignment study without model reselection.

## 2026-09-23 — alignment queue external contention

- Preserve `ALIGN_STD` and `ALIGN_RANK` as completed DEVELOPMENT outcomes. `ALIGN_TEMP_0.5_0.5` launch attempt 1 is only a prestart external-contention event: it acquired no p2 lock, loaded no model, and created no score or metric.
- Resume the same frozen alignment identity under a distinct launch-attempt-2 log after a sustained idle window. No alignment method, temperature, objective, tie rule, or protected role changes.
- Launch attempt 2 was also blocked during the inner idle check, this time by ACT-Agent evaluation PID 53602. Source review confirms the flock is acquired before `sustained_idle`; it was automatically released on the exception, before CUDA qualification and before the owner record or model loading. Preserve that sequence and resume only under launch-attempt-3 logging.
- Launch attempt 3 completed `ALIGN_TEMP_0.5_0.5`, followed by terminal `ALIGN_TEMP_0.5_1.0`. A new unrelated turn-14 evaluation then blocked `ALIGN_TEMP_1.0_0.5` before qualification. Preserve its attempt-1 log and retry that unchanged candidate only under attempt-2 logging.
- Strengthen orchestration without changing scientific semantics: wait for sustained GPU idleness before every incomplete alignment candidate, retain the existing inner flock guard, and treat transient `nvidia-smi` probe failures as non-idle instead of terminating the waiter.

## 2026-09-24 — fifth alignment candidate recovery

- Accept the untracked `ALIGN_TEMP_1.0_0.5` terminal receipt without rewriting it: status, protocol, model identity, frozen config hash, VALIDATION role, output path, and output hash all validate. The successful attempt-2 log is preserved in scratch at SHA-256 `734913e44164c28e4c849dcb89a044a6ea90e496bfd44aea5bcf29d919c75563`.
- Derive the next candidate from the frozen manifest order: `ALIGN_TEMP_1.0_1.0`. Do not infer order from filenames and do not rerun the five completed candidates.

## 2026-09-25 — ALIGN_TEMP_1.0_1.0 reboot recovery

- Classify launch attempt 1 as `BLOCKED_EXTERNAL_PRESTART` from its preserved traceback, not from analogy with earlier failures. PID 68032 was detected inside `sustained_idle`; the transient flock released on exception before CUDA qualification or scientific execution.
- Preserve the original log and use only `align_temp_1.0_1.0_attempt2.log` for retry. Alignment method, temperatures, score semantics, objective, comparator, and frozen HPO search space remain unchanged.
- Accept launch attempt 2 as terminal DEVELOPMENT evidence after receipt/config/output rehash and warehouse reconciliation. Preserve the successful log at SHA-256 `b362c56a2ea97e158e0455ca9544c63c029f0b05c2c48ce6ca4c19952596383b`; do not rerun this candidate.
- Resume strictly from frozen manifest-next `ALIGN_TEMP_1.0_2.0`. No HPO trial is considered complete because the audited HPO receipt count remains zero.

## 2026-09-25 — dGPU-only idle-guard recovery

- Preserve the frozen 256 MiB/5% aggregate threshold for configurations where the RTX GPU is not directly driving a display.
- In Lenovo dGPU-only mode, aggregate `nvidia-smi` memory and utilization include Windows display work and cannot identify CUDA contention. Require both `--query-compute-apps` and `nvidia-smi pmon` to show no unrelated compute PID throughout the sustained window, then repeat the same probes under the p2 flock before CUDA qualification.
- Treat any failed NVIDIA probe as non-idle. This is an operational compatibility correction caused by the host GPU-mode change; it does not amend any scientific identity or frozen model/HPO configuration.

## 2026-09-25 — score-alignment development selection

- All nine frozen alignment candidates terminated PASS on VALIDATION with matching config and scored-output hashes.
- Apply the frozen unit-scaled cold/warm objective and tie rule without amendment. `ALIGN_RANK` has the highest selection score (`0.6470106506911057`) and is selected for subsequent router development/HPO configuration.
- This selection opens no protected role and does not change the frozen P2-E00 HPO search space.

## 2026-09-25 — P2-E00 queue validation repair

- `CSEQ_T1` completed scientific execution and wrote a terminal PASS receipt before orchestration failed while looking up a nonexistent `config_sha256` manifest key. The manifest has always stored the materialized file hash as `sha256`; the receipt uses that same hash.
- Rehash and canonicalize the existing checkpoint/output rather than rerunning the trial. Repair only the post-run validator key lookup, preserve all scientific configuration, and resume from frozen next trial `CSEQ_T2`.

## 2026-09-26 — LIGHTGCN_T3 interrupted attempt 1

- Classify attempt 1 as `INTERRUPTED_UNATTRIBUTED`. Preserve its original log and hash; do not infer cause from the sparse invariant warning in that log or the dxgkrnl warning recorded during the later recovered boot.
- The absence of checkpoint, output, terminal receipt, and scientific metric permits an infrastructure retry of the unchanged frozen trial under a distinct attempt-2 log identity. No scientific budget or configuration changes.
- Keep the current WSL/kernel environment during active p2 execution. Timestamp-monitor attempt 2 for dxgkrnl warnings, CUDA errors, GPU activity, and process liveness. A second interruption/hang with a same-time dxgkrnl warning stops the queue pending an explicit environment amendment.

### Attempt-2 disposition

- Attempt 2 passed without the predeclared same-time failure signature. Preserve attempt 1 as `INTERRUPTED_UNATTRIBUTED`, accept attempt 2 as the terminal scientific result, and continue the frozen queue without a WSL/kernel amendment.
- The completed 21-trial pre-router grid uses the frozen family selection rule. Its selected identities remain DEVELOPMENT inputs to router HPO and do not open QUALIFICATION.

## 2026-09-26 — router HPO closure

- Apply the preregistered two-sided router objective and tie rule unchanged. Select `R2_T3`, `R3_T3`, and `R4_T1` within their families for independent-seed stability; no family is promoted at this stage.
- Preserve the negative router selection scores and every trial result. The scores show that no router simultaneously matched both development comparators under the frozen objective.
- Preserve all nine original receipts with their inherited pilot stage labels. Record the mismatch as a metadata defect, retain the P2-E00 config hashes as the scientific identity, and correct only future receipt generation.

## 2026-09-26 — independent-seed stability closure

- Accept all 30 preregistered P2-E01 runs as DEVELOPMENT evidence after the queue revalidated every terminal receipt, checkpoint hash, and VALIDATION-output hash without rerunning any model.
- Apply the frozen no-best-seed rule. Subsequent protected scoring must use the preregistered arithmetic mean of aligned scores from all three independent-seed checkpoints.
- Retain `ALIGN_RANK` and the already frozen metric, slice, robustness, ablation, and statistics definitions. The seed summary does not select a seed and does not open either protected role.
- Treat the observed stability differences as development evidence only. In particular, preserve R3's high controlled-zero variability and every unfavorable family result; do not use them to alter the HPO space or seed policy.
