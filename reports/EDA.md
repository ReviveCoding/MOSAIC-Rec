# Pre-freeze EDA (measured 2026-09-20)

Source receipts: [MicroLens QA](../evidence/microlens_qa.json), [MovieLens QA](../evidence/ml32m_qa.json), [MicroLens EDA](../evidence/microlens_eda.json), [MovieLens EDA](../evidence/ml32m_eda.json). Exact cutoffs below are exploratory and **not frozen**.

| Dataset | Events | Users | Items | FIT/validation/qualification/final candidate events | FIT users overlapping validation/final |
|---|---:|---:|---:|---|---|
| MicroLens 100K | 719,405 | 100,000 | 19,738 | 503,584 / 71,940 / 71,941 / 71,940 | 41,851 / 39,063 |
| MovieLens 32M | 32,000,204 | 200,948 | 84,432 rated | 22,400,143 / 3,200,021 / 3,200,021 / 3,200,019 | 6,465 / 3,239 |
| PixelRec50K | 989,494 | 50,000 | 82,865 | 692,646 / 98,950 / 98,949 / 98,949 | 31,979 / 29,502 |

The MovieLens global time split produces mostly **new users** in later roles: 19,013 of 22,252 final-role users have no FIT interaction. A sequential replication restricted to users with FIT history would be a small, selected subset. This protocol issue must be resolved during qualification and disclosed; do not present all final users as ordinary warm-user sequential prediction.

MicroLens FIT item support is also sparse: 533 items have one event, 2,357 have 2–5, 5,576 have 6–20, and 7,579 have >20. FIT has 16,045 observed items; 3,693 additional items are absent from FIT. First observed time is only a visibility proxy, not a launch time. No locked outcome metric has been read.

The MicroLens item-to-title mapping covers all 19,738 interaction items, with 72 empty titles. The PixelRec item metadata covers all 82,865 interaction items, with 192 empty titles. PixelRec's seven aggregate engagement-count columns are excluded from features because their as-of times are unspecified and could leak future behavior. Neither dataset provides title/description revision timestamps; historical content leakage remains a limitation to qualify before confirmatory claims.
