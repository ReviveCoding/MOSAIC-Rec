"""CPU-safe API contract checks; no development checkpoint is loaded."""

import unittest
from importlib.util import find_spec

if find_spec("fastapi") and find_spec("torch"):
    from fastapi.testclient import TestClient
    from mosaic_rec.serving import create_app
else:
    TestClient = None
    create_app = None


@unittest.skipIf(create_app is None, "Optional serving stack is not installed")
class ServingContractTests(unittest.TestCase):
    def test_unconfigured_service_is_explicit(self) -> None:
        with TestClient(create_app()) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "unconfigured")
            response = client.post("/recommend", json={"user_id": 1, "k": 20})
            self.assertEqual(response.status_code, 503)
            response = client.post("/diagnostics", json={"user_id": 1, "k": 20})
            self.assertEqual(response.status_code, 503)

    def test_invalid_request_rejected(self) -> None:
        with TestClient(create_app()) as client:
            self.assertEqual(client.post("/recommend", json={"user_id": -1}).status_code, 422)
            self.assertEqual(client.post("/recommend", json={"user_id": 1, "k": 101}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
