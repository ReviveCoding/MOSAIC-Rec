"""Record local PixelRec raw-cover archive integrity without extracting media."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import py7zr

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/home/bjw-0/mosaic-rec-work")
DATA = SCRATCH / "data/pixelrec50k"
ARCHIVE = DATA / "cover.7z"
METADATA = DATA / "item_info.csv"
CRC_LOG = SCRATCH / "logs/pixelrec_cover_7zz_crc_p1.log"
RECEIPT = ROOT / "evidence/pixelrec_cover_archive_integrity_p1.json"
EXPECTED_ARCHIVE_SHA256 = (
    "a27ba156123a30555d563be820011c406b5817621f93b7d5dcedc9b090dfa5ee"
)
SEVEN_ZIP_SHA256 = "dc99eff5008f1ab79bd7084c68513701547a808a89502bf4133683535ab3c695"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    if RECEIPT.exists():
        raise FileExistsError("PixelRec archive receipt exists; preserve it")
    started = datetime.now(timezone.utc).isoformat()
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != 20_988_051_113:
        raise ValueError("Completed official cover archive missing or size changed")
    archive_sha = digest(ARCHIVE)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("PixelRec cover archive SHA-256 changed")
    if not CRC_LOG.is_file() or "Everything is Ok" not in CRC_LOG.read_text():
        raise ValueError("Full streaming 7-Zip CRC result absent")
    tool_archive = SCRATCH / "tools/7zip-26.03/7z2603-linux-x64.tar.xz"
    if digest(tool_archive) != SEVEN_ZIP_SHA256:
        raise ValueError("7-Zip package identity changed")
    with py7zr.SevenZipFile(ARCHIVE, "r") as source:
        names = source.getnames()
    images = {Path(name).stem for name in names if name.lower().endswith(".jpg")}
    with METADATA.open(newline="") as handle:
        metadata = {row["item_id"] for row in csv.DictReader(handle)}
    if len(names) != 82_866 or len(images) != 82_865 or images != metadata:
        raise ValueError("PixelRec archive image-to-metadata mapping mismatch")
    receipt = {
        "stage_id": "A02",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "RAW_ARCHIVE_INTEGRITY_PASS_FEATURE_RIGHTS_HOLD",
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset": "pixelrec50k",
        "source_url": "https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG",
        "source_file_id": "1VlQxEUoFntgQx0WnNcRsgJu_EtYEBzlS",
        "archive_path": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_sha,
        "metadata_sha256": digest(METADATA),
        "archive_entries": len(names),
        "image_items": len(images),
        "metadata_items": len(metadata),
        "missing_images": 0,
        "extra_images": 0,
        "crc_tool": "7-Zip 26.03 Linux x64 console",
        "crc_tool_package_sha256": SEVEN_ZIP_SHA256,
        "crc_command": "7zz t -mmt=2 -bb0 cover.7z",
        "crc_exit_code": 0,
        "crc_log": str(CRC_LOG),
        "crc_log_sha256": digest(CRC_LOG),
        "warnings": [
            "Raw archive integrity only; no images were extracted or used for model features.",
            "Local derived visual feature rights remain under a conservative hold.",
            "Official source did not provide an independent publisher SHA-256 to compare with this local digest.",
        ],
        "evidence_class": "measured_source_integrity",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(RECEIPT)


if __name__ == "__main__":
    main()
