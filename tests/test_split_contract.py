"""Known-answer temporal and controlled-cold leakage contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mosaic_rec.leakage_audit import audit
from mosaic_rec.split import materialize


class SplitContractTest(unittest.TestCase):
    def test_all_fit_items_hidden_and_corruption_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pairs.csv"
            source.write_text(
                "user,item,timestamp\n"
                "1,1,1\n2,2,2\n1,3,3\n1,2,4\n1,3,5\n1,4,6\n"
            )
            config = root / "split.json"
            config.write_text(json.dumps({
                "protocol_id": "test", "seed": 1,
                "cold_cohort_fraction": 1.0,
                "datasets": {"microlens-100k": {
                    "fit_end": 3, "validation_end": 4,
                    "qualification_end": 5, "controlled_cold": True,
                }},
            }))
            split_dir = root / "splits"
            result = materialize("microlens-100k", source, config, split_dir)
            self.assertEqual(result["source_rows"], 6)
            self.assertEqual(result["outputs"]["FIT_VISIBLE"]["rows"], 0)
            self.assertEqual(result["outputs"]["FIT_WITHHELD_COLD"]["rows"], 3)
            self.assertEqual(audit("microlens-100k", split_dir, config)["status"], "PASS")

            fit_path = split_dir / "FIT_VISIBLE.parquet"
            injected = pa.table({
                "user_id": ["1"], "item_id": ["1"], "timestamp": [1],
            })
            pq.write_table(injected, fit_path)
            failure = audit("microlens-100k", split_dir, config)
            self.assertEqual(failure["status"], "FAIL")
            self.assertEqual(failure["controlled_cold_items_with_visible_fit_interactions"], 1)


if __name__ == "__main__":
    unittest.main()
