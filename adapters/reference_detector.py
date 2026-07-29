#!/usr/bin/env python3
"""Protocol placeholder for a future external detector.

The built-in worker currently generates clearly-labelled reference metrics. A real
detector adapter should consume a TaskRequest and write COCO detection JSON without
importing platform code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["describe", "health"])
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = {
        "protocol_version": "1.0",
        "adapter_id": "reference-detector",
        "status": "healthy",
        "warning": "流程样例适配器，不执行真实模型推理。",
    }
    if args.result:
        args.result.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

