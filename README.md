# MOSAIC-Rec

MOSAIC-Rec is an auditable local research study of multimodal sequential recommendation. The primary track uses MicroLens-100K; MovieLens 32M and PixelRec50K are external validity tracks. The scientific design and outcome gates are in [the specification](docs/MOSAIC_REC_SPEC.md). [PLANS.md](PLANS.md) records execution state, and [reports/PROGRESS.md](reports/PROGRESS.md) records measured work.

**Current state:** development pilots. No baseline, MOSAIC finalist, promotion result, or production effect is established. The controlled cold and FIT-warm slices have shown conflicting pilot behavior; preserve both. For exact stage status and limitations, see [evidence/stage_status.json](evidence/stage_status.json) and [reports/BLOCKERS.md](reports/BLOCKERS.md).

## Runtime and data

Windows PowerShell is the control plane. All scientific Python, PyTorch and CUDA execution runs in Ubuntu-22.04 WSL. The isolated environment and large raw data, features, caches and checkpoints live in `/home/bjw-0/mosaic-rec-work`; the Windows repository holds code, compact evidence and reports. `configs/bootstrap.json` and `configs/gpu_policy.json` define these paths and the single-GPU rule.

Datasets are obtained from their official sources, checked against terms, hashed, and schema-qualified before use. Raw data, media and extracted vectors must stay outside this repository. See [the license audit](reports/DATASET_LICENSE_AUDIT.md) for source URLs and restrictions. TMDB content is excluded.

## Reproduction entry points

Run these from PowerShell at the repository root. The `wsl` commands cross into Ubuntu-22.04 for scientific execution.

```powershell
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python scripts/gpu_probe.py
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m unittest discover -s tests -v
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.warehouse evidence evidence/warehouse.duckdb evidence/tables
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.validate_evidence evidence evidence/warehouse.duckdb
```

The WSL environment has the package installed in editable mode; `requirements.lock` captures its resolved packages. The current pilot does **not** reproduce a locked scientific conclusion. The complete acquisition identities, split cutoffs, and results are in `configs/data_registry.json`, `configs/split_p1.json`, and `evidence/`.
