# MOSAIC-Rec Master Execution Prompt

Read completely before acting:

1. AGENTS.md
2. docs/MOSAIC_REC_SPEC.md
3. configs/bootstrap.json
4. configs/gpu_policy.json
5. reports/BOOTSTRAP_PREFLIGHT.txt

This is an execution request, not a planning-only request.

Implement and execute MOSAIC-Rec from the bootstrap repository through every
feasible terminal stage.

Start by verifying:

- current WSL state
- GPU/CUDA
- disk/RAM/swap
- current official public-dataset sources
- current dataset terms/licenses
- current package/reference compatibility

Use live web research for changing external facts.

All substantive Python/CUDA/PyTorch execution must run in Ubuntu-22.04.

Use the local RTX GPU for neural recommendation workloads, feature extraction,
and full-catalog neural scoring.

Do not silently perform long neural training on CPU.

On one physical GPU:
- one heavy model training process at a time
- use batched CUDA kernels for actual parallel computation
- use BF16/AMP when qualified
- overlap CPU preprocessing/DataLoader/statistics with GPU computation
- cache content embeddings
- maximize GPU utilization rather than spawning competing model trainers

Create and qualify an isolated Linux project environment before installing the
scientific stack.

Prefer WSL-native scratch/cache paths from configs/bootstrap.json for large
data and high-I/O intermediate artifacts.

Keep source code, configs, compact evidence, reports, and final aggregate
results in the Windows repository.

Do not stop after scaffolding, data download, or first model success.

Proceed through:

desktop study
license/provenance
data acquisition
data QA
EDA
temporal split
leakage checks
feature extraction
baselines
MOSAIC variants
HPO
qualification
freeze
locked evaluation
cold/warm analysis
robustness
ablation
replication
systems benchmarking
statistics
evidence canonicalization
claim ledger
final report
reproducibility audit

If one external branch is blocked:
- record the blocker exactly
- continue every independent branch
- never invent substitute evidence

If a scientific gate fails:
- preserve the failure
- do not weaken the gate
- continue only scientifically authorized work

Maintain reports/PROGRESS.md throughout.

Before completion perform an adversarial audit covering:

future leakage
split contamination
cold-item collaborative leakage
baseline undertraining
unequal HPO budgets
sampled-metric headline leakage
best-seed cherry picking
license violations
CPU fallback
unsupported reliability claims
unsupported production claims
unmeasured multi-GPU claims

Finish only when reports/FINAL_STATUS.md exists and every planned stage has
a terminal status.