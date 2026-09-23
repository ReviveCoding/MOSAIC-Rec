# p2 blockers

## HOLD — PixelRec raw covers

The current official dataset license says no rights are granted to copy, modify, publish, distribute, or commercialize the dataset. The local `cover.7z` remains unextracted and unused. Blur, crop, occlusion, or new raw-cover derivatives are excluded unless explicit permission or a source-specific grant is obtained.

Independent path: use official publisher-provided extracted features and permitted item attributes locally without redistribution.

## BLOCKED_EXTERNAL — official publisher text file

The official PixelRec README links a 9,290,203,646-byte publisher text-vector file, but Google Drive returned a quota-exceeded response on 2026-09-23. The response is preserved at `/home/bjw-0/mosaic-rec-work/p2/downloads/publisher_text_feature_quota_attempt1.html`, SHA-256 `61be7a39fb35ddee071f7314112254bdfd5ae84da4c3dbde69cc145bd14a52eb`.

Independent path: amendment 001 selects the already authorized local MiniLM embedding derived only from official title, tag, and description attributes. Feature alignment must pass before model execution.

## Resource note

The WSL filesystem has adequate capacity. The Windows volume is 94% used, so large p2 artifacts must remain under `/home/bjw-0/mosaic-rec-work/p2`.

## ACTIVE TRANSIENT — unrelated ACT-Agent CUDA trainer

R1 attempt 1 was blocked before model loading when the 30-second idle guard observed PID 126553 running ACT-Agent SFT with approximately 7.6 GiB VRAM and 53% GPU utilization. No p2 output was created. CPU-only p2 work continues; the next R1 attempt requires a new sustained idle window.

R3 attempt 1 was likewise blocked before lock acquisition by ACT-Agent PID 137901. After the reboot, neither CUDA process exists, the GPU is idle, and the actual p2 `flock` is available. These receipts remain historical external-prestart evidence; there is no current GPU blocker as of the 2026-09-23 recovery audit.

After that audit, R3 attempt 2's outer queue observed an idle window, but ACT-Agent launched PID 2404 during the runner's second idle guard. The runner failed closed before acquiring the lock or loading R3. Its log is preserved and hashed; the next permitted identity is attempt 3.

R3 attempt 3 is queued through the committed crash-safe wrapper. ACT-Agent training PID 2404 released CUDA, but follow-on evaluation PID 9045 acquired CUDA before the required six-check idle window completed. The queue has created no R3 scientific artifact and does not hold the p2 GPU lock while waiting.
