# MOSAIC-Rec Final Scientific and Engineering Specification

## Identity

MOSAIC-Rec

Reliability-Conditioned Multimodal Sequential Recommendation under
Behavioral Sparsity, Inductive Cold Start, and Temporal Shift.

## Research objective

Test whether text/image/metadata signals improve sequential recommendation
when collaborative item evidence is sparse or absent, and whether adaptive
fusion can exploit these signals without materially degrading warm-item
recommendation.

Do not assume multimodal is always better.

## Evidence tracks

### A. MicroLens-100K
Primary multimodal + sequential study.

### B. MovieLens 32M
Large-scale behavioral / temporal replication.
Do not fabricate visual-modality claims.

### C. PixelRec50K
External multimodal replication.

A fourth dataset is optional only if a specific scientific limitation
requires it and its license/resource gate passes.

TMDB content is excluded from ML feature generation.

## End-to-end lifecycle

Desktop study
→ license/provenance audit
→ downloads/hashes
→ data QA
→ EDA
→ temporal splits
→ leakage audit
→ frozen content features
→ classical baselines
→ graph/retrieval baselines
→ sequential baselines
→ multimodal baselines
→ MOSAIC variants
→ equal-budget HPO
→ seed qualification
→ comparator freeze
→ MOSAIC freeze
→ scientific freeze
→ locked full-catalog evaluation
→ cold/warm analysis
→ sparsity / long-tail / popularity analysis
→ robustness
→ ablation
→ external replication
→ GPU/system benchmark
→ canonical evidence
→ statistical inference
→ claim ledger
→ final report.

## Data roles

Use chronological:

FIT
VALIDATION
QUALIFICATION
LOCKED_FINAL

Initial target split:
70 / 10 / 10 / 10 percent by time.

Exact cutoff timestamps are frozen before locked access.

## Controlled inductive cold start

Select a deterministic item cohort before final evaluation.

For selected cold items:
- preserve content/metadata
- remove FIT collaborative interactions from model-visible training data
- make learned collaborative ID evidence unavailable

Report as:
controlled inductive item cold start.

Do not claim first-observed time equals real platform launch time.

Initial support bins:
Cold = 0
Emerging = 1-5
Sparse = 6-20
Warm > 20

Final thresholds are frozen during qualification.

## Candidate availability

Never recommend obviously future items in historical evaluation.

Use defensible first-observed/release approximation and document its limits.

## Core modalities

collaborative
text
image
metadata

MicroLens video features are a secondary extension.

Primary content encoders remain frozen.

## Baselines

B0 Global Popularity
B1 Recent Popularity
B2 BPR-MF
B3 LightGCN
B4 Two-Tower
B5 GRU4Rec
B6 SASRec
B7 BERT4Rec
B8 VBPR
B9 BM3
B10 FREEDOM or equivalently strong maintained multimodal challenger

Use RecBole/MMRec/original maintained implementations for reference
qualification where appropriate.

Do not intentionally undertrain baselines.

## Fair HPO

Comparable neural models use:
- same data roles
- common evaluation code
- common candidate semantics
- same seed policy
- common early stopping semantics
- declared trial budget
- declared GPU-time budget

Pilot: 1 seed
Shortlist: 3 seeds
Qualification: 3 seeds
Locked: frozen checkpoints only

No best-seed cherry picking.

## MOSAIC representations

z_c collaborative
z_t text
z_v image
z_m metadata

Time-safe gate context may include:

log(1 + support before t)
item age
text missing
image missing
metadata coverage
text-image agreement
image-quality proxy

No future support information.

## Fusion

a = softmax(g(context))

z_mosaic =
a_c P_c z_c +
a_t P_t z_t +
a_v P_v z_v +
a_m P_m z_m

Mask unavailable branches and renormalize.

Gate weights are not calibrated reliability probabilities.

## Behavioral dropout

During FIT, deliberately hide collaborative information for selected
training examples so the model learns inductive content fallback.

## Modality dropout

During FIT, create controlled text/image/metadata missingness.

Keep locked robustness corruption distinct from development augmentation.

## Sequence backbone

Primary proposed backbone:
SASRec-style Transformer.

Candidate score uses dynamic user state and MOSAIC item representation.

## MOSAIC ladder

M0 naive sequence + content
M1 fixed fusion
M2 global learned fusion
M3 support-conditioned gate
M4 + behavioral dropout
M5 + modality dropout
M6 + consistency signal
M7 monotonic-support gate challenger
M8 final MOSAIC finalist

Do not assume M8 wins.

## Two frozen comparators

C_seq:
strongest qualified sequential baseline.

C_cold:
strongest qualified inductive-capable cold baseline.

Do not make an ID-only cold-incapable baseline the sole cold comparator.

## Metrics

Primary:
NDCG@20

Secondary:
Recall@20
MRR@20
HitRate@20

Sensitivity:
NDCG@10
Recall@50

Ecosystem:
coverage
tail percentage
average popularity
Gini
entropy

Do not collapse these into one weighted score.

## Evaluation

Development may use sampled ranking for speed.

Qualification and locked final use exact full-catalog ranking where tractable.

ANN belongs to the systems benchmark, not the primary scientific headline.

## Co-primary gates

P1 cold superiority:

Delta_cold =
NDCG@20(MOSAIC cold) - NDCG@20(C_cold cold)

Require predeclared paired 95% lower bound > 0.

P2 warm noninferiority:

Delta_warm =
NDCG@20(MOSAIC warm) - NDCG@20(C_seq warm)

Require lower bound > -delta.

Both required for CORE_PROMOTE.

Delta must be defined and frozen during QUALIFICATION.

## Reliability gates

Reliability language requires separate evidence.

Test:
- missing image
- missing text
- blur
- crop/occlusion
- incorrect image pairing
- text truncation
- metadata loss
- collaborative support reduction
- temporal shift
- long-tail slices

Validate directional gate response.

If those gates fail, use:
adaptive multimodal fusion

rather than stronger reliability claims.

## Ablations

no sequence
no collaborative
no text
no image
no metadata
no adaptive gate
no behavioral dropout
no modality dropout
no consistency signal
unconstrained vs monotonic gate
no auxiliary alignment
retrieval-only vs reranking

Pre-freeze confirmatory ablation family.

## Statistics

Primary repeated unit:
user.

Primary:
paired user-cluster bootstrap.

Cold sensitivity:
item-cluster bootstrap.

Target:
10,000 final bootstrap draws unless preregistered compute/degeneracy evidence
justifies a change.

Secondary comparison family:
Holm.

Exploratory slices:
descriptive or explicitly FDR-controlled.

## Replication

MovieLens 32M:
behavioral / sequential replication.

PixelRec50K:
multimodal / cold-start replication.

Do not require equal effect magnitude across domains.

## GPU

Long neural training must use CUDA when CUDA is qualified.

Use:
BF16/AMP
pinned host memory
multiple DataLoader workers
prefetch
non-blocking H2D
cached multimodal embeddings
batched CUDA scoring
exact GPU full-catalog scoring

One physical GPU:
one heavy training process at a time.

CPU data/statistics/report tasks may overlap safely.

No silent neural CPU fallback.

## torch.compile

Benchmark eager vs compile.

Measure:
cold compilation cost
steady throughput
VRAM
numerical parity
break-even

Adopt only if measured useful.

## Systems experiments

embedding items/s
training sequences/s
GPU memory
exact ranking users/s
ANN Recall@200
retrieval latency
reranking p50/p95
FastAPI p50/p95
throughput
errors
startup time

Keep system performance separate from scientific model identity.

## Canonical evidence

Parquet + DuckDB tables:

runs
datasets
splits
trials
checkpoints
metrics
slices
robustness
ablations
gate_weights
latency
gpu
bootstrap
claims

Ad-hoc notebook numbers cannot become resume/public claims.

## Stage DAG

A00 desktop study
A01 license audit
A02 acquisition/provenance
A03 schema QA
A04 EDA
A05 split freeze
A06 leakage audit

B00 content feature extraction
B01 feature qualification

C00 classic baselines
C01 graph baseline
C02 retrieval baseline
C03 sequential baselines
C04 multimodal baselines
C05 reference parity

D00-D08 MOSAIC development

E00 HPO closure
E01 seed qualification
E02 comparator freeze
E03 MOSAIC freeze
E04 margin/gate freeze
E05 final manifest

F00 locked overall
F01 cold
F02 warm
F03 support bins
F04 long-tail/popularity
F05 user-history slices

Gxx robustness
Hxx ablations

I00 MovieLens replication
I01 PixelRec replication
I02 replication synthesis

J00 GPU profiling
J01 exact scoring
J02 ANN
J03 reranking
J04 serving
J05 Pareto analysis

K00 canonical statistics
K01 failure analysis
K02 claim ledger
K03 technical report
K04 reproducibility audit
K05 FINAL_STATUS

## State machine

PILOT
→ DEVELOPMENT
→ QUALIFICATION
→ FROZEN
→ LOCKED_EXECUTION
→ ANALYSIS
→ CLOSED

Post-lock scientific changes require a new protocol identity.

## Terminal decisions

CORE_PROMOTE
RELIABILITY_PROMOTE
LOCAL_VALIDITY_ONLY
COLD_START_TRADEOFF
RETAIN_BASELINE
INVALID_SCIENTIFIC_RESULT
BLOCKED_EXTERNAL
BLOCKED_GPU

## Final deliverables

installable Python package
dependency lock
data/license registry
desktop study
EDA
protocol
stage receipts
frozen configs
canonical evidence warehouse
figures
robustness/ablation reports
FastAPI service
final technical report
claim ledger
resume-safe bullets
interview study guide
reproduction runbook
reports/FINAL_STATUS.md

No project completion claim before all stages have terminal status.