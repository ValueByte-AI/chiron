#!/usr/bin/env python3
"""Train Chiron with joint prediction, measurement, and population supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import shutil
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "chiron-data"

code = DATA_ROOT / "code"
if not (code / "build_joint_training.py").is_file():
    raise FileNotFoundError("Download chiron-group/chiron-data to chiron-data/ before training.")
sys.path.insert(0, str(code))
from build_joint_training import validate_row


def load_vocab(path):
    vocab = json.loads(Path(path).read_text())
    if len(vocab) != 109 or sorted(vocab.values()) != list(range(109)):
        raise ValueError("Expected 109 constructs indexed from 0 through 108")
    return vocab


DATA = DATA_ROOT / "data/sft/joint_train.jsonl"
VOCAB = DATA_ROOT / "data/raw/measurement/construct_vocab.json"
CONFIG = ROOT / "configs/train_joint.yaml"
DATASET_DIR = ROOT / "configs"
BACKEND = ROOT / "LLaMA-Factory/src"
TRAINING_WORLD_SIZE = 4
EARLY_STEPS = frozenset((50, 100, 200, 400))


def load_config(path=CONFIG):
    """Read the recipe without importing a model or a training framework."""
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")
    required = {
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "template": "qwen3_6",
        "enable_thinking": False,
        "train_on_prompt": False,
        "mask_history": False,
        "packing": False,
        "neat_packing": False,
        "use_chiron_measurement_loss": True,
        "chiron_num_constructs": 109,
        "chiron_measurement_num_layers": 1,
        "use_reentrant_gc": False,
        "save_only_model": False,
        "save_total_limit": None,
        "overwrite_output_dir": False,
    }
    for key, value in required.items():
        if (
            key not in config
            or type(config[key]) is not type(value)
            or config[key] != value
        ):
            raise ValueError(f"Joint training requires {key}={value!r}")
    reserved = {
        "dataset",
        "dataset_dir",
        "eval_dataset",
        "model_name_or_path",
        "output_dir",
        "resume_from_checkpoint",
        "adapter_name_or_path",
        "chiron_measurement_head_init",
        "chiron_construct_vocab",
        "use_ray",
        "use_chiron_population_loss",
    }
    if reserved.intersection(config):
        raise ValueError(
            f"Paths and initialization are managed by the entry point: {sorted(reserved.intersection(config))}"
        )
    ds_path = (ROOT / config["deepspeed"]).resolve()
    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    if ds.get("zero_optimization", {}).get("stage") != 2:
        raise ValueError("Joint training requires DeepSpeed ZeRO-2")
    config["deepspeed"] = str(ds_path)
    dataset_registration()
    return config


def validate_dataset(path=DATA):
    """Check every row with the same schema validator used by the joint builder."""
    vocab = load_vocab(VOCAB)
    counts = Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                validate_row(row, vocab)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Joint data line {number}: {exc}") from exc
            counts[row["signal"]] += 1
    if set(counts) != {"prediction", "measurement", "population"}:
        raise ValueError("Joint data must contain all three supervision signals")
    return dict(counts)


def dataset_registration():
    """Read the native LLaMA-Factory dataset configuration."""
    metadata = json.loads(
        (DATASET_DIR / "dataset_info.json").read_text(encoding="utf-8")
    )
    registration = metadata["chiron_joint"]
    if (
        registration.get("formatting") != "chiron_joint"
        or (DATASET_DIR / registration["file_name"]).resolve() != DATA
    ):
        raise ValueError(
            "Dataset registration must point to the joint file using the chiron_joint format"
        )
    return metadata


def validate_resume(checkpoint, output_dir, world_size):
    """Require a full same-run ZeRO-2 checkpoint, not an inference-only export."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.parent != Path(
        output_dir
    ).resolve() or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError(
            "Resume checkpoint must belong to the selected output directory"
        )
    required = (
        "trainer_state.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "chiron_measurement_head.pt",
        "chiron_person_mean_ema.pt",
        "scheduler.pt",
        "construct_vocab.json",
    )
    missing = [
        name
        for name in required
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size
    ]
    for rank in range(world_size):
        name = f"rng_state_{rank}.pth" if world_size > 1 else "rng_state.pth"
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            missing.append(name)
    optimizers = list(checkpoint.glob("**/*optim_states.pt"))
    models = list(checkpoint.glob("**/*model_states.pt"))
    if len(optimizers) != world_size or any(not p.stat().st_size for p in optimizers):
        missing.append(f"{world_size} complete ZeRO optimizer shards")
    if not models or any(not p.stat().st_size for p in models):
        missing.append("DeepSpeed model state")
    if missing:
        raise ValueError(f"Incomplete resume checkpoint: {', '.join(missing)}")
    saved_vocab = json.loads(
        (checkpoint / "construct_vocab.json").read_text(encoding="utf-8")
    )
    if saved_vocab != json.loads(VOCAB.read_text(encoding="utf-8")):
        raise ValueError(
            "Resume checkpoint vocabulary differs from the training vocabulary"
        )
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step = state.get("global_step", 0)
    if type(step) is not int or checkpoint.name != f"checkpoint-{step}":
        raise ValueError(
            "Checkpoint directory step differs from trainer_state.global_step"
        )
    if not 0 < step < state.get("max_steps", 0):
        raise ValueError("Resume checkpoint has no unfinished training steps")
    return str(checkpoint)


def validate_resume_recipe(
    checkpoint, config, row_count, world_size=TRAINING_WORLD_SIZE
):
    """Compare existing Trainer state with the current batch and epoch schedule."""
    state = json.loads(
        (Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8")
    )
    batch = config["per_device_train_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    epochs = config["num_train_epochs"]
    updates_per_epoch = math.ceil(row_count / (batch * world_size * accumulation))
    expected = {
        "max_steps": math.ceil(updates_per_epoch * epochs),
        "num_train_epochs": math.ceil(epochs),
        "train_batch_size": batch,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Resume schedule differs for {key}: checkpoint={state.get(key)!r}, recipe={value!r}"
            )


def validate_output_dir(output_dir, resume=False):
    """Require a new output directory unless a checkpoint was explicitly selected."""
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output path must be a directory")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Output directory is not empty; choose a new directory or explicitly resume"
        )


def make_save_callback(base_callback):
    """Save construct metadata alongside Trainer checkpoints."""

    class JointSaveCallback(base_callback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in EARLY_STEPS or state.global_step == state.max_steps:
                control.should_save = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                shutil.copyfile(VOCAB, Path(args.output_dir) / "construct_vocab.json")
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                shutil.copyfile(VOCAB, checkpoint / "construct_vocab.json")
            return control

    return JointSaveCallback()


def load_backend():
    """Load the bundled Chiron training backend."""
    if not (BACKEND / "llamafactory").is_dir():
        raise RuntimeError("Bundled LLaMA-Factory source is missing")
    sys.path.insert(0, str(BACKEND))
    import llamafactory
    from llamafactory.data.converter import ChironJointDatasetConverter
    from llamafactory.hparams import DataArguments, FinetuningArguments
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    if not Path(llamafactory.__file__).resolve().is_relative_to(BACKEND):
        raise RuntimeError("A different LLaMA-Factory package was imported")
    if (
        "chiron_construct_vocab" not in DataArguments.__dataclass_fields__
        or "chiron_num_constructs" not in FinetuningArguments.__dataclass_fields__
        or ChironJointDatasetConverter is None
    ):
        raise RuntimeError(
            "This entry point requires the bundled Chiron LLaMA-Factory adaptation"
        )
    return run_exp, make_save_callback(TrainerCallback)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "models/qwen3.6-35b-a3b"),
        help="Base model directory",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/joint")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Full checkpoint inside the same output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and all rows; no model loads or writes",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config()
    output_dir = args.output_dir.expanduser().resolve()
    validate_output_dir(output_dir, resume=args.resume_from_checkpoint is not None)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed rank environment")
    if world_size > 1 and "LOCAL_RANK" not in os.environ:
        raise ValueError("Use torchrun to launch distributed training")
    if not args.dry_run and (
        world_size != TRAINING_WORLD_SIZE or "LOCAL_RANK" not in os.environ
    ):
        raise ValueError(
            "This recipe requires torchrun --nproc_per_node=4; use --dry-run for single-process validation"
        )
    if not args.dry_run and not args.model:
        raise ValueError("Specify --model to train; --dry-run does not require a model")
    if os.environ.get("USE_RAY", "").strip().lower() in {"1", "true", "yes"}:
        raise ValueError("Use torchrun for this training entry point")
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = validate_resume(
            args.resume_from_checkpoint, output_dir, TRAINING_WORLD_SIZE
        )
    counts = validate_dataset()
    if checkpoint is not None:
        validate_resume_recipe(checkpoint, config, sum(counts.values()))
    if rank == 0:
        print(
            f"Validated {sum(counts.values()):,} rows: {counts}; 109 constructs",
            flush=True,
        )
    if args.dry_run:
        return 0
    run_exp, callback = load_backend()
    config.update(
        model_name_or_path=args.model,
        output_dir=str(output_dir),
        dataset="chiron_joint",
        dataset_dir=str(DATASET_DIR),
        chiron_construct_vocab=str(VOCAB),
        resume_from_checkpoint=checkpoint,
    )
    run_exp(config, callbacks=[callback])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
