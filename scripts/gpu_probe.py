from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
    }

    try:
        import torch
    except Exception as exc:
        result.update(
            torch_import=False,
            error=repr(exc),
        )
        print(json.dumps(result, indent=2))
        return 2

    result.update(
        torch_import=True,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_runtime=torch.version.cuda,
        device_count=torch.cuda.device_count(),
    )

    if not torch.cuda.is_available():
        print(json.dumps(result, indent=2))
        return 3

    result["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]

    result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

    # Real CUDA compute qualification.
    x = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    y = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    z = x @ y

    torch.cuda.synchronize()

    result["cuda_matmul_finite"] = bool(torch.isfinite(z).all().item())

    print(json.dumps(result, indent=2))
    return 0 if result["cuda_matmul_finite"] else 4


if __name__ == "__main__":
    raise SystemExit(main())