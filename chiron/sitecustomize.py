"""Apply FP32 LoRA deltas while vLLM streams base weights."""

import os

if os.environ.get("CHIRON_LORA_F32_OVERLAY"):
    import json, torch
    from pathlib import Path
    from safetensors import safe_open
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    adapter = Path(os.environ["CHIRON_LORA_F32_OVERLAY"])
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["bias"] == "none"
    assert not cfg["fan_in_fan_out"] and not cfg["use_dora"] and not cfg["use_rslora"]
    assert (
        not cfg["alpha_pattern"]
        and not cfg["rank_pattern"]
        and not cfg["modules_to_save"]
    )
    pairs = {}
    with safe_open(
        adapter / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as f:
        for key in f.keys():
            assert key.startswith("base_model.model.")
            name = key[len("base_model.model.") :]
            if name.endswith(".lora_A.weight"):
                base = name[: -len(".lora_A.weight")] + ".weight"
                which = "A"
            elif name.endswith(".lora_B.weight"):
                base = name[: -len(".lora_B.weight")] + ".weight"
                which = "B"
            else:
                raise RuntimeError("Unsupported adapter tensor " + key)
            pairs.setdefault(base, {})[which] = f.get_tensor(key).float()
    assert len(pairs) == 350 and all(set(p) == {"A", "B"} for p in pairs.values())
    scale = float(cfg["lora_alpha"]) / cfg["r"]
    original = DefaultModelLoader.get_all_weights

    def get_weights(self, model_config, model):
        seen = set()
        for name, weight in original(self, model_config, model):
            if name in pairs:
                assert name not in seen
                p = pairs[name]
                assert tuple(weight.shape) == (p["B"].shape[0], p["A"].shape[1])
                merged = weight.float() + scale * (p["B"] @ p["A"])
                assert torch.isfinite(merged).all()
                seen.add(name)
                yield name, merged
            else:
                yield name, weight
        missing = set(pairs) - seen
        if missing:
            raise RuntimeError("Missing adapted weights " + str(sorted(missing)))
        print(
            "CHIRON_F32_OVERLAY_APPLIED",
            len(seen),
            "unmodified base tensors retained",
            flush=True,
        )

    DefaultModelLoader.get_all_weights = get_weights
