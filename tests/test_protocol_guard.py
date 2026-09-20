"""Protected outcome guard checks before any qualification or locked reading."""

import json
import tempfile
import unittest
from pathlib import Path

from mosaic_rec.protocol import (ScientificState, assert_role_access,
                                 assert_transition, validate_freeze_manifest)


class ProtocolGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = {"protocol_id": "mosaic-rec-2026-09-20-p1",
                       "scientific_state": "DEVELOPMENT"}

    def test_protected_roles_sealed_in_development(self) -> None:
        assert_role_access(self.status, "VALIDATION")
        with self.assertRaises(PermissionError):
            assert_role_access(self.status, "QUALIFICATION")
        with self.assertRaises(PermissionError):
            assert_role_access(self.status, "LOCKED_FINAL")

    def test_freeze_requires_complete_manifest(self) -> None:
        self.status["scientific_state"] = "QUALIFICATION"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            path.write_text(json.dumps({"protocol_id": self.status["protocol_id"]}))
            with self.assertRaises(ValueError):
                validate_freeze_manifest(path, self.status["protocol_id"])
            with self.assertRaises(ValueError):
                assert_transition(self.status, ScientificState.FROZEN,
                                  frozen_manifest_path=path)

    def test_no_skipping_states(self) -> None:
        with self.assertRaises(ValueError):
            assert_transition(self.status, ScientificState.FROZEN)


if __name__ == "__main__":
    unittest.main()
