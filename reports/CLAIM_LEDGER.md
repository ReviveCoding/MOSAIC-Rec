# Claim ledger

Canonical source: `evidence/claims.json` and the `claims` table in `evidence/warehouse.duckdb`.

| Claim | Status | Evidence | Constraint |
|---|---|---|---|
| p1 terminal decision is RETAIN_BASELINE | SUPPORTED | `f00_locked_statistics_p1` | Specific to frozen p1 |
| M1_T2 cold point estimate exceeded B4 by 0.008940 | SUPPORTED | `f00_locked_statistics_p1` | 95% interval crossed zero; superiority unsupported |
| M1_T2 failed warm noninferiority | SUPPORTED | `f00_locked_statistics_p1` | Offline MicroLens p1 only |
| M1_T2 overall secondary metric exceeded B7 | SUPPORTED | `f00_locked_statistics_p1` | Cannot replace failed co-primary gates |
| PixelRec local feature alignment passed | SUPPORTED | `pixelrec_content_alignment_p1` | QA, not replication or redistribution permission |
| Local API measured 4.49 ms median latency | SUPPORTED | `j00_j04_systems_p1` | Single-process development demonstration |
| CORE_PROMOTE | UNSUPPORTED | `f00_locked_statistics_p1` | Both co-primary gates failed |
| Reliability-aware gating | UNSUPPORTED | `gxx_hxx_definition_audit_p1` | Confirmatory intervention protocol invalid |
| External replication | UNSUPPORTED | — | No frozen external transfer study |
| Production impact/readiness | UNSUPPORTED | — | No online or production evidence |
| Multi-GPU scaling | UNSUPPORTED | — | One physical GPU |

Every supported numerical claim maps to a canonical receipt. Stronger prohibited formulations are stored in `evidence/claims.json`.
