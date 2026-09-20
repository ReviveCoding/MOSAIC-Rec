import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mosaic_rec.publisher_feature_qa import qualify


class PublisherFeatureMappingTests(unittest.TestCase):
    def test_complete_vectors_follow_item_ids_not_json_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "item_info.csv"
            with metadata.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["item_id"])
                writer.writeheader()
                writer.writerows([{"item_id": "i2"}, {"item_id": "i1"}])
            source = root / "official.json"
            source.write_text(json.dumps({"i2": [2.0] * 1024, "i1": [1.0] * 1024}))
            output = root / "features.npy"
            index = root / "ids.txt"
            result = qualify(source, metadata, output, index)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["items_mapped"], 2)
            self.assertEqual(index.read_text().splitlines(), ["i1", "i2"])
            vectors = np.load(output)
            self.assertTrue(np.all(vectors[0] == 1.0))
            self.assertTrue(np.all(vectors[1] == 2.0))

    def test_missing_item_fails_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "item_info.csv").write_text("item_id\ni1\ni2\n")
            (root / "official.json").write_text(json.dumps({"i1": [1.0] * 1024}))
            output = root / "features.npy"
            result = qualify(root / "official.json", root / "item_info.csv",
                             output, root / "ids.txt")
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["items_missing"], 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
