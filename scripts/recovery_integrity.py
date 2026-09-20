"""Read-only post-crash artifact audit; never opens protected outcome rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/home/bjw-0/mosaic-rec-work")
EVIDENCE = Path("evidence")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(label: str, path: Path, expected: str, rows: int | None = None) -> None:
        record: dict[str, object] = {"label": label, "path": str(path)}
        if not path.is_file():
            record["status"] = "MISSING"
        else:
            actual = digest(path)
            record["status"] = "PASS" if actual == expected else "HASH_MISMATCH"
            record["sha256"] = actual
            if rows is not None and record["status"] == "PASS":
                # Footer metadata only. No QUALIFICATION or LOCKED_FINAL outcome rows read.
                actual_rows = pq.read_metadata(path).num_rows
                record["rows"] = actual_rows
                if actual_rows != rows:
                    record["status"] = "ROW_COUNT_MISMATCH"
        checks.append(record)

    registry = json.loads(Path("configs/data_registry.json").read_text())["datasets"]
    micro = ROOT / "data/microlens-100k"
    movie = ROOT / "data/ml-32m"
    pixel = ROOT / "data/pixelrec50k"
    for label, path, expected in (
        ("MicroLens pairs", micro / "pairs.csv", registry["microlens-100k"]["pairs_sha256"]),
        ("MicroLens titles", micro / "title_en.csv", registry["microlens-100k"]["title_en_sha256"]),
        ("MovieLens archive", movie / "ml-32m.zip", registry["ml-32m"]["archive_sha256"]),
        ("MovieLens ratings", movie / "ml-32m/ratings.csv", registry["ml-32m"]["ratings_sha256"]),
        ("PixelRec interactions", pixel / "interaction.csv", registry["pixelrec50k"]["interactions_sha256"]),
        ("PixelRec item info", pixel / "item_info.csv", registry["pixelrec50k"]["item_info_sha256"]),
    ):
        check(label, path, expected)

    for name in ("microlens", "ml32m", "pixelrec"):
        manifest = json.loads((EVIDENCE / f"{name}_split_p1.json").read_text())
        check(f"{name} source", {
            "microlens": micro / "pairs.csv",
            "ml32m": movie / "ml-32m/ratings.csv",
            "pixelrec": pixel / "interaction.csv",
        }[name], manifest["source_sha256"])
        cold_path = Path(next(iter(manifest["outputs"].values()))["path"]).parent / "controlled_cold_item_ids.txt"
        check(f"{name} cold IDs", cold_path, manifest["cold_cohort_sha256"])
        for role, info in manifest["outputs"].items():
            check(f"{name} {role}", Path(info["path"]), info["sha256"], info["rows"])

    for path in sorted(EVIDENCE.glob("*.json")):
        if path.name.startswith("recovery_integrity_"):
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        if data.get("checkpoint_sha256"):
            check(f"checkpoint {path.stem}", Path(data["checkpoint"]), data["checkpoint_sha256"])
        if data.get("features_sha256"):
            check(f"local features {path.stem}", Path(data["features_path"]), data["features_sha256"])

    feature = json.loads((EVIDENCE / "microlens_feature_qa.json").read_text())["features"]
    directory = ROOT / "features/microlens-100k"
    for key, prefix in (
        ("image", "MicroLens-100k_image_features_CLIPRN50"),
        ("text", "MicroLens-100k_title_en_text_features_BgeM3"),
        ("video_secondary", "MicroLens-100k_video_features_VideoMAE"),
    ):
        check(f"publisher {key} NPY", directory / f"{prefix}.npy", feature[key]["npy_sha256"])
        check(f"publisher {key} JSON", directory / f"{prefix}.json", feature[key]["json_sha256"])

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
