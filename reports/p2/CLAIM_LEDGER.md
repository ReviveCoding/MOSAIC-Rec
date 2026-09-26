# MOSAIC-Rec p2 claim ledger

Protocol: `mosaic-rec-2026-09-23-p2`

## Supported locked claims

| Statement | Evidence | Limitations | Status |
|---|---|---|---|
| R4 achieved locked overall NDCG@20 `0.0018460` on 25,367 PixelRec queries. | `p2-f00-f02-locked-summary.json` | Offline exact ranking only. | SUPPORTED |
| R4's controlled-zero point estimate exceeded C_cold by `0.0008903`, but H1 failed because the paired-user 95% lower bound was `0`. | Same receipt, 10,000 draws | One hit made uncertainty material; no superiority claim. | SUPPORTED |
| R4 failed warm noninferiority: mean delta versus C_seq `-0.0013377`, 95% interval `[-0.0022765, -0.0005345]`, frozen margin `0.0013209`. | Same receipt | PixelRec offline warm population only. | SUPPORTED |
| The conjunctive promotion gate failed; final decision is `RETAIN_BASELINE`. | Frozen H1/H2 rule and locked receipt | R4 remains the evaluated finalist. | SUPPORTED |
| Eleven of fourteen preregistered robustness cells violated the positive-support routing direction. | `p2-g00-robustness-summary.json` | Diagnostic offline interventions. | SUPPORTED |
| Exact locked scoring processed 667.6 users/s for R4, 1,425.0 for C_seq, and 1,480.1 for C_cold on one RTX 4090 Laptop GPU. | `p2-j00-j05-systems-summary.json` | Local single-run workload measurements. | SUPPORTED |

## Qualified secondary claims

- The locked top-k analysis reports coverage, popularity, tail exposure, exposure concentration, entropy, novelty, and intra-list diversity under frozen FIT-only definitions.
- Personalization is unavailable because the frozen pair-selection algorithm was underspecified.
- The canonical ablation family is INVALID because no executable retraining and locked-scoring runner was bound before protected access.
- MovieLens behavioral transfer is HOLD; p2 provides no external replication claim.

## Prohibited claims

- R4 superiority, promotion, or warm retention.
- Reliability-aware behavior.
- Production, engagement, causal, online, or business impact.
- External replication or multimodal MovieLens replication.
- Valid confirmatory ablation conclusions.
- Multi-GPU scaling, production readiness, or generic `torch.compile` speedup.
