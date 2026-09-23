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
