# Desktop study (2026-09-20)

Research question: whether frozen text/image/metadata features improve temporal sequential recommendation for behaviorally sparse or controlled cold items without materially harming warm-item ranking. The design deliberately permits negative findings. Primary outcome is exact full-catalog NDCG@20 with user-cluster paired inference and separate cold/warm comparators, per `docs/MOSAIC_REC_SPEC.md`.

Official source review:

- [MicroLens project](https://github.com/westlake-repl/MicroLens) publishes 100K interactions and content links. Its repository still describes original Python 3.8/PyTorch 1.8 code; this is a reference, not a safe primary dependency lock. The [current portal](https://recsys.westlake.edu.cn/) lists the 100K pairs, titles, covers, extracted features and a revised March 2026 directory.
- [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) is a stable behavioral dataset. It has ratings, tags and metadata, but no native image modality; use it for behavioral/sequential replication only.
- [PixelRec](https://github.com/westlake-repl/PixelRec) provides PixelRec50K via Google Drive and notes 989,494 interactions, 50,000 users, 82,865 items. Source identity and access remain to be qualified.
- [PyTorch Linux installation](https://pytorch.org/get-started/locally/) supports Ubuntu 22.04 and Python 3.10, with a CUDA wheel selected for the machine. Actual CUDA tensor qualification is required after install.

Theoretical/engineering risks: observational interactions are not randomized exposure; first observed time is not item launch; missing modalities can correlate with popularity; fixed content encoders can import external pretraining information; candidate availability must be time safe; original baseline environments are outdated. Do not infer production impact from public offline ranking.
