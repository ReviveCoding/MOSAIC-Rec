# MOSAIC-Rec p2 technical report

## Question and protocol

P2 tested whether a support-aware router could retain a content expert's controlled-zero advantage and remain noninferior to a sequence expert for FIT support above 20. PixelRec50K was evaluated with fixed temporal roles, exact full-catalog ranking, three-seed ensembles, retained top-k lists, and two conjunctive primary gates.

## Frozen identities

- C_seq: `CSEQ_BERT4REC`
- C_cold: `CCOLD_TWO_TOWER`
- Finalist: R4, selected on QUALIFICATION with score `-0.6350770566`
- Alignment: `ALIGN_RANK`
- Warm margin: `0.001320914873267875`
- E05 manifest: `15233342edf761c957f061e5ff72323feda8cdcfae9119e8a464fc046c509de7`

## Locked results

| Model | Overall NDCG@20 | Controlled zero | Warm >20 |
|---|---:|---:|---:|
| C_seq | 0.0001107 | 0 | 0.0013377 |
| C_cold | 0.0010083 | 0 | 0.0016131 |
| R4 | 0.0018460 | 0.0008903 | 0 |

H1 failed despite a positive cold point estimate because its paired-user 95% lower bound was exactly zero. H2 failed because the lower bound `-0.0022765` was below the frozen threshold `-0.0013209`. The conjunction therefore failed.

## Robustness and ablations

All 14 frozen robustness cells executed. Eleven missingness cells moved positive-support routing in the prohibited direction, so G00 failed and reliability language is unsupported. The ablation family is INVALID: its overrides and seeds were frozen, but an executable retraining and locked-scoring runner was not bound before protected access.

## Secondary and systems evidence

The retained lists enabled 213 model-by-slice rows plus coverage, exposure, novelty, and diversity metrics. Personalization was not computed because pair construction was underspecified. R4 exact scoring achieved 667.6 users/s and produced a 28.8 MB compressed top-k artifact on one RTX 4090 Laptop GPU.

## Decision

`RETAIN_BASELINE`. R4 remains the evaluated finalist and is not replaced post hoc. Future work should preregister an executable ablation pipeline and test a structurally hard zero-support router with a warm sequence fallback.
