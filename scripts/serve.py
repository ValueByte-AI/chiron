#!/usr/bin/env python3
"""Serve Chiron with the experiment's FP32 LoRA overlay and BF16 vLLM engine."""

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
MODULE_DIR = ROOT / "chiron" if (ROOT / "chiron" / "sitecustomize.py").is_file() else ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = ROOT / "models" / "chiron" if MODULE_DIR != ROOT else ROOT
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--base-model", help="Base-model directory or Hugging Face identifier")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    adapter = args.checkpoint.expanduser().resolve()
    if (adapter / "model" / "adapter_config.json").is_file():
        adapter = adapter / "model"
    for file in (adapter / "adapter_config.json", adapter / "adapter_model.safetensors",
                 MODULE_DIR / "sitecustomize.py"):
        if not file.is_file():
            raise FileNotFoundError(file)
    config = json.loads((adapter / "adapter_config.json").read_text())
    local_base = ROOT / "models" / "qwen3.6-35b-a3b"
    base = args.base_model or (
        str(local_base) if (local_base / "config.json").is_file()
        else config["base_model_name_or_path"]
    )
    devices = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if any(not d for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the distinct GPUs allocated to this job")
    env = dict(os.environ, CHIRON_LORA_F32_OVERLAY=str(adapter))
    env["PYTHONPATH"] = str(MODULE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", base,
        "--served-model-name", "chiron",
        "--tensor-parallel-size", str(len(devices)),
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--generation-config", "vllm",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--gpu-memory-utilization", "0.85",
        "--max-num-seqs", "64",
        "--enforce-eager", "--enable-prefix-caching",
        "--override-generation-config", '{"eos_token_id":[248046,248044]}',
    ]
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
