# MOSAIC-Rec

[![CPU scientific contracts](https://github.com/ReviveCoding/MOSAIC-Rec/actions/workflows/ci.yml/badge.svg)](https://github.com/ReviveCoding/MOSAIC-Rec/actions/workflows/ci.yml)

Auditable multimodal sequential recommendation research with exact full-catalog evaluation, frozen model-selection gates, uncertainty-aware diagnostics, and reproducible evidence.

MOSAIC-Rec studies whether a support-aware router can preserve a multimodal content expert's zero-support behavior while retaining the quality of a sequence expert on warm items. Protocol `mosaic-rec-2026-09-23-p2` is **CLOSED** with the permanent decision **RETAIN_BASELINE**.

## What the project studies

The p2 study uses PixelRec50K under chronological `FIT_VISIBLE`, `VALIDATION`, `QUALIFICATION`, and `LOCKED_FINAL` roles. It compares:

- **C_seq:** a frozen three-seed BERT4Rec-style sequence ensemble;
- **C_cold:** a frozen three-seed multimodal Two-Tower content ensemble;
- **R4:** a frozen three-seed context-aware router with a warm-preservation objective.

R4 combines separately trained sequence and content experts using pre-query support, item-age, user-history, recency, preference, and content-availability covariates. Router weights are model control values, not calibrated reliability probabilities.

The protocol uses exact full-catalog ranking, identical candidate semantics, immutable top-k artifacts, one-time protected-role access, deterministic paired bootstrap inference, and a canonical Parquet/DuckDB evidence warehouse.

## Locked primary results

The one-time `LOCKED_FINAL` evaluation contained **25,367 queries**. Overall NDCG@20 is secondary and cannot override either primary gate.

| Result | Estimate | Frozen inference | Decision |
| --- | ---: | --- | --- |
| R4 overall NDCG@20 | 0.001846 | Secondary metric | — |
| H1: R4 − C_cold, controlled zero support | +0.0008903 | Paired 95% interval [0, 0.0026710] | **FAIL** |
| H2: R4 − C_seq, warm | −0.0013377 | Paired 95% interval [−0.0022765, −0.0005345]; margin 0.0013209 | **FAIL** |

Both gates were required for promotion. The final scientific decision is therefore **RETAIN_BASELINE**. Eleven of fourteen preregistered robustness cells also violated the declared routing-direction expectation.

See the [p2 final status](reports/p2/FINAL_STATUS.md), [technical report](reports/p2/MOSAIC_REC_P2_TECHNICAL_REPORT.md), [claim ledger](reports/p2/CLAIM_LEDGER.md), and [adversarial audit](reports/p2/ADVERSARIAL_AUDIT.md).

## Posthoc industry diagnostics

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Read-only diagnostics reused immutable locked outputs and did not rerun inference or alter p2:

- 25,367 locked queries and 11,214 target items were analyzed.
- In the torso-popularity slice, R4 − C_seq was +0.003296 NDCG@20 over 8,817 queries.
- In the support `>50` slice, R4 − C_seq was −0.004394 over 520 queries.
- Only 1 of 295 controlled-zero queries had a positive R4 − C_cold NDCG difference.
- The experts tied on 99.68% of per-query NDCG outcomes.
- Routing-side agreement was 86.25% among the 80 informative expert non-ties.

These values support future hypotheses and diagnostics only. They do not revise the locked decision or establish model superiority. See the [industry diagnostic report](reports/posthoc/INDUSTRY_DIAGNOSTIC_REPORT.md), [dashboard](reports/posthoc/INDUSTRY_DASHBOARD.html), and [artifact manifest](analysis/posthoc/posthoc_manifest.json).

## Reproducibility

Public CI validates formatting, linting, CPU-safe contracts, synthetic data tests, and pure posthoc helpers. It does not train models or access protected roles.

```bash
python -m pip install --no-deps -e .
ruff check src tests scripts
ruff format --check src tests scripts
PYTHONPATH=src:. pytest -q --import-mode=importlib tests
PYTHONPATH=src pytest -q scripts/posthoc/test_diagnostics.py
```

The full scientific replay requires the authorized public datasets and the external scratch artifacts described in the [p2 reproduction runbook](docs/p2/REPRODUCTION_RUNBOOK.md). Large datasets, feature arrays, checkpoints, and protected full top-k outputs are intentionally excluded from Git. Checked-in compact evidence, aggregate tables, manifests, and reports retain hashes for auditability.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/mosaic_rec/` | Installable package, models, metrics, protocol guards, statistics, and evidence tools |
| `configs/p2/` | Preregistered and frozen p2 configuration identities |
| `evidence/p2/` | Compact canonical receipts, manifests, aggregate evidence, and warehouse |
| `analysis/posthoc/` | Read-only exploratory diagnostic tables and hash manifests |
| `reports/p2/` | Canonical closed-study reports and claim ledger |
| `reports/posthoc/` | Exploratory slicing, routing, exposure, robustness, and dashboard reports |
| `scripts/` | Crash-safe execution, validation, and analysis entry points |
| `tests/` | CPU-safe unit, schema, leakage, protocol, metric, and replay contracts |

## Hardware and systems disclosure

Scientific neural workloads used one NVIDIA RTX 4090 Laptop GPU under Ubuntu 22.04 in WSL. Frozen R4 exact full-catalog scoring measured **667.6 users/s**. This is a local, single-GPU systems result; no multi-GPU or production-readiness claim is supported.

## Claim boundaries

This repository does **not** claim:

- production or online impact;
- causal lift;
- statistical cold-start superiority;
- warm retention or noninferiority;
- reliability-aware routing;
- external replication;
- valid canonical ablation conclusions;
- multi-GPU scaling.

Negative and invalid results remain part of the public scientific record. The canonical p2 decision is **RETAIN_BASELINE**.

## Data and licensing

Raw datasets and derived publisher assets are not redistributed. Dataset provenance, terms, citation requirements, derived-feature restrictions, and redistribution limits are documented in the repository's data and rights audits. TMDB content was excluded from model training and feature generation.
