# p2 progress

Protocol: `mosaic-rec-2026-09-23-p2`

Scientific state: `QUALIFICATION`

## Current phase

Development and QUALIFICATION are complete. C_seq, C_cold, R2, R3, and R4 were exact-scored on the same 26,900 QUALIFICATION queries with immutable top-k outputs. The frozen rule selected R4 as the finalist; its negative selection score preserves the cold/warm tradeoff and is not promotion. LOCKED_FINAL remains sealed.

## Completed

- Verified p1 Git and artifact hashes without modifying p1.
- Rechecked current official PixelRec and MovieLens terms and source locations.
- Rechecked current RecBole/MMRec repository identities and primary-environment CUDA operation.
- Preregistered roles, cohorts, model families, primary gates, metrics, top-k retention, statistics, robustness, ablations, seed policy, and protected-access gates.
- Committed the preregistration boundary at Git `6f00e42` before p2 model execution.
- Materialized and hashed fresh PixelRec roles under `/home/bjw-0/mosaic-rec-work/p2`.
- Created a distinct deterministic controlled-zero cohort with 3,013 items.
- Passed temporal-boundary, controlled-zero, natural-zero, count, null, and duplicate audits.
- Preserved two failed preparation attempts rather than overwriting them.
- Created p2-owned feature paths for 82,865 items and verified aligned indices, expected image/text dimensions, and finite values.
- Excluded all seven engagement aggregates from the permitted metadata inputs.
- Built FIT-only item and user context tables and hashed earliest eligible role query manifests.
- Confirmed zero candidate-time violations across 27,918 validation, 26,900 qualification, and 25,367 locked query rows.
- Preserved the first BERT4Rec and Two-Tower attempts but invalidated their validation metrics after detecting incomplete FIT-history masking for 905 users.
- Corrected the shared loader to retain complete FIT histories for candidate masking while keeping last-50 neural inputs.
- Completed exact CPU global and 30-day recent popularity baselines; the optimized scorer preserved exact ranks and top-100 lists.
- R1 attempt 1 was blocked before model loading by an unrelated ACT-Agent CUDA trainer; no p2 router artifact exists yet.
- Added CUDA pilot implementations for BPR-MF, LightGCN, SASRec, and VBPR; corrected the shared zero-based item indexing contract before executing them.
- Rebuilt the p2-only canonical warehouse with invalidation-aware selection semantics: 19 p2 receipts, 8 valid metric rows, zero p1 canonical receipts, and zero protected access. The two invalid expert receipts remain stored but their 8 metric rows are excluded.
- Completed and independently rehashed corrected attempt-2 expert runs after a sustained idle GPU window. C_seq validation NDCG@20 is 0.001864 overall and 0.013602 warm; C_cold is 0.001312 overall, 0.000940 controlled zero, 0.001411 natural zero, and 0.001570 warm.
- R1 hard routing preserved warm quality at 0.013539 but scored zero on controlled zero under per-query standardized alignment. This unfavorable development result is retained and motivates the preregistered alignment comparison.
- R2 monotonic routing reached 0.014892 warm NDCG@20 and 0.002306 overall but also scored zero on controlled zero. R3 attempt 1 then stopped before model loading when ACT-Agent PID 137901 began a new CUDA SFT job.
- Recovered after the reboot from pre-reboot HEAD `49ee02aa3da271fc1a0dec3350ea8f526d142785`. Preserved and validated the untracked preregistered slice implementation and tests; Ruff, targeted tests, and the full 71-test suite pass.
- Reverified all corrected C_seq, C_cold, R1, and R2 hashes and Parquet schemas. The canonical warehouse passes with 25 p2 receipts, 24 metric rows, zero p1 receipts, and zero protected access.
- The reboot audit found no R3, R4, HPO, temporary, or queue artifact. The R3 attempt-1 receipt remains a prestart contention event with no scientific output. The stale lock text names completed R2 PID 137697, while an actual `flock` probe was available.
- After reboot the RTX 4090 Laptop GPU showed 0 MiB, 0% utilization, no CUDA process, and passed the real torch CUDA/BF16 tensor qualification.
- R3 attempt 2 completed the queue idle window, but ACT-Agent launched CUDA PID 2404 during the runner's independent idle guard. The p2 guard failed closed before lock acquisition/model loading; only the 760-byte prestart log exists and is hashed. Any retry will use attempt 3 paths.
- R3 attempt 3 waited through ACT-Agent's follow-on evaluation, acquired the p2 lock only after six idle checks, and completed on the RTX 4090 Laptop GPU. Exact VALIDATION NDCG@20 was 0.001939 overall, 0 controlled zero, 0.000131 natural zero, and 0.013539 warm. The unfavorable zero-support result is preserved; checkpoint and output hashes independently match the receipt.
- Rebuilt the p2-only warehouse after R3: 27 receipts, 28 valid metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS.
- R4 attempt 1 passed its outer idle window, then failed closed before lock acquisition when ACT-Agent CUDA PID 14599 launched. Its 761-byte log is preserved at SHA-256 `a0721489efe9fb32606950cbe61c6e920f554053cdae464cf176d02d2eb4e66c`; no checkpoint, scored output, terminal model receipt, or scientific metric exists.
- R4 attempt 2 likewise reached six outer idle checks after ACT-Agent training step 120, but the final evaluation PID 26411 appeared during the nested guard. Its 761-byte log is preserved at SHA-256 `a1fd0dd39768b62c96ee8e40b7d2e91e0efa39600e0827f72fdfea0bc3656225`; it created no scientific artifact.
- R4 attempt 3 completed under the p2 lock after ACT-Agent's final evaluation exited. Exact VALIDATION NDCG@20 was 0.001938 overall, 0 controlled zero, 0.000131 natural zero, and 0.013537 warm. The warm-preservation term did not recover zero-support quality under the standardized alignment pilot; this negative result is preserved.
- Independently rehashed R4 checkpoint `5474235bbd2b05db6849f905eca77e7f003b9955e9a371c13394394b946869a1`, output `23a6f1b15d8fdfb5f148a7f3cada78ecc94048b32d2f16df8075ee718eed457c`, and log `806a79a93c46561136235a4faf35435943a8b83d3a7e1aea59793d4f7e04135e`.
- Rebuilt the p2-only warehouse after R4: 30 receipts, 32 valid metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS.
- Completed and independently rehashed BPR-MF, LightGCN, and SASRec CUDA pilots. Overall/warm NDCG@20 were respectively 0.000707/0.004266, 0.000068/0.000139, and 0.001932/0.013587; all three scored zero on controlled zero.
- VBPR attempt 1 completed training and wrote checkpoint SHA-256 `def78b03de2ba34aad4b136cd5b777273a1ef0d3569c11b8700d1eaf74122062`, then OOMed before any scored output or metric while the evaluator unnecessarily materialized a 128 x catalog x 1024 image tensor. The log is preserved at SHA-256 `fefd39820f381f1c1ce9bb974fe7cb7d5391b67a528f635a2977e247ed835560`.
- Validated an algebraically equivalent bounded-memory VBPR scorer that projects the image catalog once, then uses matrix multiplication. Attempt 2 has a distinct config, log, checkpoint, output, and receipt identity; attempt-1 artifacts remain untouched.
- Rebuilt the p2-only warehouse at this recovery boundary: 34 receipts, 44 valid metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS.
- VBPR attempt 2 completed on CUDA with exact validation NDCG@20 0.000478 overall, 0 controlled zero, 0.000559 natural zero, and 0.000395 warm. Its distinct checkpoint and output independently match SHA-256 `abc7595d0ba0b06bf86a7f03b32dac938789f4ecaac1e706ee989a6c07c61b7b` and `6049421b5ffb3592f29ba00b2b3c137a2b10d94ad3dfb32daf436122c4009bb1`.
- Legacy M1 completed on CUDA with exact validation NDCG@20 0.005991 overall, 0.005334 controlled zero, 0.009025 natural zero, and 0.000806 warm. This preserves the intended development tradeoff: stronger cold behavior with severe warm loss.
- Rebuilt the p2-only warehouse after the remaining pilots: 36 receipts, 52 valid metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS.
- Alignment candidates `ALIGN_STD` and `ALIGN_RANK` completed. `ALIGN_RANK` recovered controlled-zero NDCG@20 to 0.000845 with warm NDCG@20 0.012780; this remains DEVELOPMENT selection evidence.
- `ALIGN_TEMP_0.5_0.5` launch attempt 1 failed closed before lock acquisition or model loading when unrelated ACT-Agent CUDA PID 38158 appeared. Its 775-byte log is preserved at SHA-256 `cd3ede7462f713cef893b40357f3018d0f676a39c2166b26ac698a92f1e1f6e3`; no alignment output or metric exists.
- The committed wait wrapper later reached launch attempt 2, which failed in the inner idle guard on unrelated ACT-Agent evaluation PID 53602. The OS flock was acquired transiently and released on the prestart exception before CUDA qualification or owner-record writing. No model, scoring output, or metric was created; the 775-byte log is preserved at SHA-256 `43c75c9aeac1968bf0ca5696987637472bbf9fa6863973d79e8867deb669a601`.
- After PID 53602 exited, launch attempt 3 completed `ALIGN_TEMP_0.5_0.5`; `ALIGN_TEMP_0.5_1.0` also completed. Both produced overall/warm NDCG@20 0.001864/0.013602 and controlled-zero NDCG@20 0.
- The next candidate, `ALIGN_TEMP_1.0_0.5`, was blocked before CUDA qualification by new ACT-Agent turn-14 evaluation PID 58174. Its attempt-1 log is preserved at SHA-256 `30acb827575ab5df3a9c59f9f1f01ff86270961703df5d800a0206a2112c7894`; no output or metric exists.
- PID 58174 later exited, but ACT-Agent launched Qwen3-4B control turn-14 evaluation PID 61711 before the six-check idle window completed. The hardened waiter reset without launching attempt 2, acquiring the p2 lock, or creating an artifact. Per-candidate no-lock idle checks and transient `nvidia-smi` retry handling are committed.
- `ALIGN_TEMP_1.0_0.5` launch attempt 2 subsequently completed on CUDA. Its frozen config hash and 29,446,255-byte VALIDATION output independently match the terminal receipt; exact NDCG@20 was 0.001864 overall, 0 controlled zero, and 0.013602 warm. The p2-only warehouse now passes with 44 receipts and 72 metric rows.
- The receipt schema identifies this candidate through `model_id=R1_ALIGN_TEMP_1.0_0.5` and its frozen config hash; it does not include separate `alignment_id` or `protected_roles_opened` fields. Global protocol counters remain authoritative at zero, and p1 immutability hashes pass.

## 2026-09-26 — independent-seed stability

- Materialized and committed 30 frozen P2-E01 identities before execution: ten selected families across independent seeds `20261001`, `20261002`, and `20261003`. HPO seed `20260930` was excluded.
- All 30 CUDA runs terminated `PASS`; the crash-safe queue then rehashed every checkpoint and VALIDATION output and skipped all terminal identities without retraining.
- The deterministic summary uses all three seeds and explicitly prohibits best-seed selection. `ALIGN_RANK`, frozen metrics, frozen slices, and zero protected access were preserved.
- Development means show the preregistered tradeoff remains material: `R2` controlled-zero/warm NDCG@20 means are `0.0090449251`/`0.0041864526`; `R3` is substantially less stable on controlled zero (CV `1.2112`); `SASREC` has the strongest listed warm mean at `0.0171005726`. These are DEVELOPMENT stability results, not promotion or protected evidence.
- All three independent-seed LIGHTGCN runs executed after the historical HPO interruption completed without the defined recurring WSL GPU failure signature. No kernel/environment amendment was made.
- The first clean pre-QUALIFICATION guard was invalidated before protected access after audit found that it hashed robustness/ablation definitions but not their executable functions. The receipt is preserved; v2 adds deterministic implementation identities and tests while both protected counters remain zero.

## Active next action

P2-E04 is frozen without opening LOCKED_FINAL. Build and commit the complete E05 manifest, then run the clean frozen protocol guard. Robustness and ablation outcomes remain scheduled for Gxx/Hxx after F00 as preregistered.

## 2026-09-26 — E04 analysis-family freeze

- Revalidated all five immutable QUALIFICATION outputs and receipts, p1 immutability, and the p2-only warehouse at 127 receipts and 328 metric rows with protected counts `1/0`.
- Retained R4 as an unpromoted finalist with selection score `-0.6350770566` and froze the numeric warm margin `0.001320914873267875` unchanged.
- Materialized the complete preregistered 14-cell robustness family and A0-A7 ablation family, including retraining decisions and seeds. These definitions are executable identities for later Gxx/Hxx; no LOCKED_FINAL result was accessed during E04.
- Added the separate exact LOCKED_FINAL scorer entrypoint for E05 hashing. It has not been executed.

## 2026-09-25 reboot recovery

- Audited the 775-byte `ALIGN_TEMP_1.0_1.0` log at SHA-256 `4a26981f69518090633cc56c5a1511677687971f06ccf85f14511dc8b69ee6be`. The inner idle guard observed unrelated CUDA PID 68032 after transient flock acquisition.
- The exception occurred before CUDA qualification, lock owner-record writing, expert checkpoint loading, exact scoring, validation-row output, or metric computation. No output, receipt, temporary, or partial candidate artifact exists.
- Post-reboot GPU inspection showed 0 MiB, 0% utilization, no CUDA PID, no alignment/HPO process, and an available actual flock. Live preflight is repeated immediately before attempt 2.
- Recovered and independently verified the successful launch-attempt-2 receipt without rewriting it. Its frozen config and 29,512,674-byte VALIDATION output hashes match; exact NDCG@20 was 0.001864 overall, 0 controlled zero, and 0.013602 warm.
- Reconciled the interrupted warehouse state deterministically: 46 p2-only receipts, 76 metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS. Manifest-next is `ALIGN_TEMP_1.0_2.0`; HPO and E01 receipt counts remain zero.
- The immediate post-commit live preflight found unrelated ACT-Agent DPO PID 1346 using about 10.8 GiB at 100% GPU utilization. The sustained-idle gate failed closed before CUDA qualification for the next candidate; no `ALIGN_TEMP_1.0_2.0` artifact exists, and the hardened waiter holds no flock while waiting.

## 2026-09-25 dGPU-only recovery

- Recovered from Git `7e9641ffdce487a1976057adef5d88c4175aaae3` with a clean worktree. Revalidated all six terminal alignment receipts against the nine-candidate frozen manifest; every config and 27,918-row VALIDATION output hash matches.
- Found no candidate-specific artifact for `ALIGN_TEMP_1.0_2.0`, `ALIGN_TEMP_2.0_1.0`, or `ALIGN_TEMP_2.0_2.0`, and no HPO/E01 receipt or scratch artifact. The exact resume point remains frozen manifest candidate 7, `ALIGN_TEMP_1.0_2.0`.
- The p2 warehouse remains consistent at 46 p2-only receipts and 76 metric rows, validation PASS, with zero p1 receipts and zero protected access. The p1 immutability hashes pass.
- The prior ACT-Agent PID 1346 and all other CUDA workers are absent. The actual p2 flock is available. Under the new dGPU-only machine mode, aggregate RTX memory/utilization includes the Windows display (about 1.8--2.0 GiB and variable graphics utilization) even with no compute PID; `nvidia-smi pmon` and the compute-app query both report no compute process.
- The operational idle guard now requires both independent NVIDIA compute-process probes to remain empty when `display_active=Enabled`; the original aggregate 256 MiB/5% fail-closed threshold is retained when the GPU is not directly driving the display. Targeted tests pass. This changes no model, role, metric, alignment, HPO, or protected-access rule.
- Completed `ALIGN_TEMP_1.0_2.0`, `ALIGN_TEMP_2.0_1.0`, and `ALIGN_TEMP_2.0_2.0` serially on CUDA after both idle guards passed. All nine candidate receipts, frozen configs, and 27,918-row outputs revalidate.
- The deterministic frozen selection rule chose `ALIGN_RANK`: selection score `0.6470106506911057`, controlled-zero NDCG@20 `0.0008446567843548943`, and warm NDCG@20 `0.012779977211517015`. This remains DEVELOPMENT evidence.
- Rebuilt the p2-only warehouse at the alignment-complete boundary: 50 receipts, 88 metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS.
- `CSEQ_T1` completed all 10 scheduled CUDA epochs and exact VALIDATION scoring before the queue hit an implementation-only terminal-validation key error. Its receipt, 21,505,487-byte checkpoint, and 11,200,760-byte output independently match their recorded hashes; it was not rerun.
- Corrected the queue to read the materialized config byte hash from the manifest's existing `sha256` field. Targeted tests pass, the warehouse canonicalizes `CSEQ_T1` at 51 receipts/92 metric rows, and protected counters remain zero. Frozen next trial is `CSEQ_T2`.
- The frozen queue subsequently completed `CSEQ_T2`, `CSEQ_T3`, all three C_cold, all three legacy M1, all three BPR-MF, and `LIGHTGCN_T1/T2` trials. Their terminal receipts and artifacts are preserved for canonicalization.
- `LIGHTGCN_T3` attempt 1 left only a 535-byte log at SHA-256 `0ffc2ab81388751eebf282b3e0ec73f645fa9b67b188f63072fd65bd424d210b`; no checkpoint, output, terminal receipt, or metric exists. It is `INTERRUPTED_UNATTRIBUTED`. Neither its PyTorch sparse warning nor a dxgkrnl warning from the later recovered boot is assigned as cause.
- Recovery environment: WSL `2.7.14.0`, kernel `6.18.33.2-microsoft-standard-WSL2` / package `6.18.33.2-2`, NVIDIA driver `616.92`, RTX 4090 Laptop GPU. The actual flock is free, no unrelated CUDA worker exists, and a real CUDA/BF16 tensor qualification passes.
- `LIGHTGCN_T3` attempt 2 completed all 10 scheduled epochs and exact scoring. Checkpoint/output hashes are `8d815882c5db85863c7653d6b29f9859333558d48ec07c4bbe6d95ed96415a6e` and `c0f08ce27a447e6c1700337c238b765501272c46b65529f9f67ebed5e4dcf65a`; the distinct attempt-2 log hash is `f79454e93aac6db96ab336fd18d2191acd30da76e4b4c2b9ae2a5bb05c886aed`.
- Timestamped monitoring found no new field-spanning/FORTIFY warning, CUDA error, device-not-ready failure, or loss of process liveness during attempt 2. The monitor log is preserved in scratch at SHA-256 `ef9c47be521686d1ad96841291150b3e696e33898f2f00d41ac04b5b9029233b`. Routine `dxgkio_escape -22` lines were observed and are not reclassified as the specified failure signature.
- All 21 pre-router HPO receipts and artifact hashes validate. The deterministic summary selected `CSEQ_T1`, `CCOLD_T2`, `LEGACY_T3`, `BPR_T3`, `LIGHTGCN_T3`, `SASREC_T1`, and `VBPR_T3` within their families. This is DEVELOPMENT selection only.
- All nine router receipts are terminal PASS. Frozen config, checkpoint, and VALIDATION output hashes match; the selection seed is `20260930`; early-stopped runs completed 6 or 9 of the scheduled maximum 10 epochs; protected counters remain zero.
- The frozen router rule selected `R2_T3`, `R3_T3`, and `R4_T1` within their families. All router selection scores are negative, preserving the unfavorable two-sided tradeoff rather than presenting family selection as gate success.
- The router receipts inherited pilot `stage_id` labels despite P2-E00 config identities. Existing receipts remain immutable; `p2-e00-router-stage-label-audit.json` records the metadata-only defect, and future gate receipts use the frozen top-level stage label.

## Protected access counters

- QUALIFICATION model outcomes: 0
- LOCKED_FINAL model outcomes: 0

## Current blockers

- Raw-cover extraction and modification remain on rights hold.
- Official publisher text-feature download is quota-blocked. Amendment 001 selects the already authorized local MiniLM embedding derived from official title/tag/description fields; its p2 copy and alignment qualification passed.
- The external ACT-Agent training/evaluation cycle is complete at the R4 boundary. Every subsequent CUDA job still repeats the live process, sustained-idle, and lock checks.
