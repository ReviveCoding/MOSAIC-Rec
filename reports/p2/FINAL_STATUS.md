# MOSAIC-Rec p2 final status

- Protocol: `mosaic-rec-2026-09-23-p2`
- State: `CLOSED`
- Decision: `RETAIN_BASELINE`
- Frozen finalist: R4, not promoted
- Protected access counts: QUALIFICATION `1`, LOCKED_FINAL `1`
- Dataset: PixelRec50K primary; MovieLens 32M transfer HOLD
- Hardware: one NVIDIA GeForce RTX 4090 Laptop GPU
- Canonical warehouse: PASS, 154 p2 receipts, 328 metric rows, zero p1 receipts
- Locked artifact source boundary: Git `738ed6212baaf3f595359327498b6781c3209a73`

## Stage disposition

| Stage | Status | Result |
|---|---|---|
| A00-A06 | PASS / PASS_WITH_HOLD | Protocol, rights, provenance, roles, leakage |
| B00-B03 | PASS | Features, contexts, alignment |
| C00-C04 | PASS_DEVELOPMENT | Comparator and baseline development |
| D00-D04 | PASS_DEVELOPMENT | Legacy and router ladder |
| E00-E05 | PASS | HPO, seeds, qualification, freeze, guard |
| F00 | PASS | Three exact locked outputs, 25,367 queries |
| F01 H1 | FAIL | Lower 95% bound `0`, strict superiority not met |
| F02 H2 | FAIL | Lower bound `-0.0022765 < -0.0013209` |
| F03-F05 | PASS_DESCRIPTIVE | Frozen slices and top-k secondary metrics |
| G00 | FAIL | 11/14 routing-direction failures |
| H00 | INVALID | Executable ablation runner absent from freeze |
| I00-I02 | HOLD | No executable transfer protocol frozen |
| J00-J05 | PARTIAL_PASS / HOLD | Training, VRAM, scoring and storage measured; API/compile held |
| K00-K05 | PASS | Ledger, audit, reports, canonical closure |

## Primary conclusion

R4 improved the controlled-zero point estimate over C_cold, but uncertainty included zero. It collapsed on the frozen warm population and failed noninferiority. Both conjunctive gates failed, so the evidence supports retaining the baseline rather than promoting MOSAIC-Router.

## Reproduction

```bash
cd /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
/home/bjw-0/mosaic-rec-work/.venv/bin/python scripts/p2/build_warehouse.py \
  --repo . --warehouse evidence/p2/warehouse.duckdb \
  --table-dir evidence/p2/tables --manifest evidence/p2/warehouse_manifest.json
/home/bjw-0/mosaic-rec-work/.venv/bin/python -m pytest -q tests/p2
```

Protected scoring must not be rerun: the protocol is closed and both protected roles have been opened exactly once.

## Recommended next research step

Start a new protocol with the hard support router as the primary candidate, fully bind the ablation training/scoring implementation before protected access, and retain the same conjunctive cold-superiority and warm-noninferiority discipline.
