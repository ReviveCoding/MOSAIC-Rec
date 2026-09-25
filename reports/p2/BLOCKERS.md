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

The follow-on evaluation exited at 2026-09-23 15:59 UTC. R3 attempt 3 subsequently completed under the p2 lock, so this transient blocker is resolved and retained here as recovery history.

R4 attempt 1 passed the outer idle checks, but ACT-Agent launched CUDA PID 14599 during the independent inner idle guard. The guard failed closed before lock acquisition, CUDA qualification, model loading, or metrics. The log and failure receipt are preserved; R4 may resume only as attempt 2 on distinct paths.

R4 attempt 2 is now queued on those distinct paths without lock ownership. ACT-Agent step-30 evaluation PID 17045 currently owns CUDA; the queue has produced no p2 artifact and continues to wait.

R4 attempt 2 later reached the outer idle threshold after ACT-Agent completed training step 120, but final evaluation PID 26411 appeared during the nested guard. Attempt 2 failed closed before lock acquisition or model load and is preserved. The next identity is attempt 3.

The final evaluation exited, and R4 attempt 3 completed under the p2 lock. The transient CUDA contention is resolved at this boundary; the historical prestart receipts remain canonical evidence.

VBPR pilot attempt 1 exhausted VRAM during exact scoring after creating a checkpoint because the implementation expanded a batch x catalog x image-feature tensor. It produced no scored output or metric. The checkpoint and log remain preserved. Distinct attempt 2 completed with the bounded-memory algebraically equivalent scorer, so this blocker is resolved.

Unrelated ACT-Agent CUDA PID 38158 started after two alignment candidates completed and blocked `ALIGN_TEMP_0.5_0.5` launch attempt 1 during the inner sustained-idle guard. The p2 lock is free, and no p2 model or output was created by that launch. Resume uses a distinct attempt-2 log only after the unrelated CUDA process exits.

The wait wrapper later launched attempt 2 after its outer idle checks, but unrelated ACT-Agent capacity evaluation PID 53602 appeared by the inner guard. Attempt 2 failed before CUDA qualification, model loading, scoring, output, or metrics. Its transient flock was automatically released, and the lock is currently available. Attempt 3 must wait for all unrelated CUDA processes to exit.

PID 53602 exited and the queue completed two more frozen candidates. ACT-Agent then launched turn-14 evaluation PID 58174, blocking `ALIGN_TEMP_1.0_0.5` attempt 1 during the inner idle guard. No scientific artifact was created by that launch; distinct attempt 2 waits for the GPU to become fully idle.

PID 58174 exited, but ACT-Agent launched Qwen3-4B control turn-14 evaluation PID 61711 before the outer six-check idle requirement completed. The waiter remains active without lock ownership and has not launched attempt 2. This is the current external GPU blocker.

`ALIGN_TEMP_1.0_0.5` attempt 2 later completed and is canonical. The current blocker is unrelated ACT-Agent full-rollout PID 113021, using about 11.6 GiB VRAM. The MOSAIC flock is available but does not establish GPU idleness. Manifest-next `ALIGN_TEMP_1.0_1.0` must wait without lock ownership.

The machine rebooted after `ALIGN_TEMP_1.0_1.0` launch attempt 1 failed its inner idle guard on unrelated CUDA PID 68032. The log is preserved; no model, scored output, or metric exists. After reboot the prior PID is absent and the GPU/flock are idle, so this historical blocker is resolved subject to the normal live attempt-2 preflight.

`ALIGN_TEMP_1.0_1.0` launch attempt 2 completed before the later interruption and is now canonical. The prior PID 68032 blocker is resolved. No current p2 scientific blocker exists at this boundary; live GPU/process/partial-artifact checks remain mandatory before `ALIGN_TEMP_1.0_2.0`.

The live preflight for `ALIGN_TEMP_1.0_2.0` found unrelated ACT-Agent DPO PID 1346 using material VRAM/compute. No p2 artifact or lock ownership exists for that candidate. The crash-safe waiter is active and will require six idle checks plus the independent inner guard before launch.

After the unexpected shutdown and switch to dGPU-only display mode, PID 1346 and every other CUDA compute process are absent and the p2 flock is available. The aggregate display reservation prevents the old headless-style memory threshold from reaching 256 MiB, so the guard was corrected to use two independent NVIDIA compute-process probes while display activity is enabled. There is no current external GPU blocker.
