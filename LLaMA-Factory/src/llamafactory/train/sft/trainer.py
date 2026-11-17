# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class ChironMeasurementHead(torch.nn.Module):
    """Mean-centered, per-construct calibrated readout from model hidden space."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_constructs: int,
        device: "torch.device",
        dtype: "torch.dtype",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_constructs != 109:
            raise ValueError("Chiron measurement heads require 109 constructs.")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("Chiron measurement num_layers must be >= 1.")
        self.construct_embeddings = torch.nn.Embedding(num_constructs, hidden_size, device=device, dtype=dtype)
        torch.nn.init.normal_(self.construct_embeddings.weight, mean=0.0, std=hidden_size**-0.5)
        # exp(log_scale) keeps every construct's calibration strictly increasing,
        # which preserves the direction and pairwise ordering of the cosine readout.
        self.construct_log_scales = torch.nn.Parameter(torch.zeros(num_constructs, device=device, dtype=dtype))
        self.construct_biases = torch.nn.Parameter(torch.zeros(num_constructs, device=device, dtype=dtype))
        self.register_buffer("person_mean", torch.zeros(hidden_size, device=device, dtype=torch.float32))
        # A scalar mixture keeps the readout dimension unchanged while allowing
        # measurement gradients to reach several decoder layers. Initializing
        # the final layer at +4 makes a multilayer run start close to the
        # final-layer readout, then learn whether earlier layers are useful.
        layer_mix = torch.zeros(self.num_layers, device=device, dtype=dtype)
        if self.num_layers > 1:
            layer_mix[-1] = 4.0
        self.layer_mix_logits = torch.nn.Parameter(layer_mix)

    @torch.no_grad()
    def set_person_mean(self, person_mean: "torch.Tensor") -> None:
        if person_mean.shape != self.person_mean.shape:
            raise ValueError(
                f"Chiron person mean shape {tuple(person_mean.shape)} != expected {tuple(self.person_mean.shape)}."
            )
        self.person_mean.copy_(person_mean.detach().float().to(self.person_mean.device))

    def forward(
        self,
        hidden_states: "torch.Tensor",
        construct_ids: "torch.Tensor",
        *,
        clamp: bool = False,
    ) -> "torch.Tensor":
        hidden_states = self.fuse_hidden(hidden_states)
        head_dtype = self.construct_embeddings.weight.dtype
        person_vectors = F.normalize(hidden_states.to(dtype=head_dtype).float(), dim=-1)
        person_vectors = F.normalize(person_vectors - self.person_mean.float(), dim=-1)
        construct_vectors = F.normalize(self.construct_embeddings(construct_ids).float(), dim=-1)
        cosine = torch.sum(person_vectors.unsqueeze(1) * construct_vectors, dim=-1)
        scales = self.construct_log_scales[construct_ids].float().exp()
        biases = self.construct_biases[construct_ids].float()
        scores = scales * cosine + biases
        return scores.clamp(-1.0, 1.0) if clamp else scores

    def fuse_hidden(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """Fuse [batch, layer, hidden] or pass through [batch, hidden]."""
        if hidden_states.ndim == 2:
            if self.num_layers != 1:
                raise ValueError(
                    f"Multilayer Chiron head expects [batch,{self.num_layers},hidden], "
                    f"got {tuple(hidden_states.shape)}."
                )
            return hidden_states
        if hidden_states.ndim != 3 or hidden_states.size(1) != self.num_layers:
            raise ValueError(
                f"Chiron hidden states must be [batch,{self.num_layers},hidden], got {tuple(hidden_states.shape)}."
            )
        weights = torch.softmax(self.layer_mix_logits.float(), dim=0).to(dtype=hidden_states.dtype)
        return torch.sum(hidden_states * weights.view(1, -1, 1), dim=1)


def _chiron_measurement_head_path(output_dir: str) -> str:
    return os.path.join(output_dir, "chiron_measurement_head.pt")


def _clone_chiron_measurement_head_state_dict(measurement_head: "ChironMeasurementHead") -> dict[str, "torch.Tensor"]:
    try:
        import torch.distributed as dist
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except Exception:
        return {name: value.detach().cpu().clone() for name, value in measurement_head.state_dict().items()}

    state_dict: dict[str, torch.Tensor] = {}
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    for name, param in measurement_head.named_parameters():
        status = getattr(param, "ds_status", None)
        enabled = status == ZeroParamStatus.NOT_AVAILABLE or param.numel() == 0
        with zero.GatheredParameters([param], modifier_rank=0, enabled=enabled):
            if is_rank0:
                state_dict[name] = param.detach().cpu().clone()

    if is_rank0:
        for name, buffer in measurement_head.named_buffers():
            state_dict[name] = buffer.detach().cpu().clone()

    return state_dict


def _load_chiron_measurement_head_state_dict(
    measurement_head: "ChironMeasurementHead", state_dict: dict[str, "torch.Tensor"]
) -> None:
    current = measurement_head.state_dict()
    for name, value in current.items():
        loaded = state_dict.get(name)
        if loaded is None:
            raise ValueError(f"Chiron measurement head checkpoint is missing parameter {name}.")

        if not isinstance(loaded, torch.Tensor) or not torch.isfinite(loaded).all():
            raise ValueError(f"Chiron measurement head parameter {name} must be a finite tensor.")
        if loaded.shape != value.shape:
            raise ValueError(
                f"Cannot load Chiron measurement head parameter {name}: "
                f"checkpoint shape {tuple(loaded.shape)} != current shape {tuple(value.shape)}."
            )

    measurement_head.load_state_dict(state_dict)


def _get_chiron_measurement_model(model: "torch.nn.Module") -> "torch.nn.Module":
    seen: set[int] = set()
    stack: list[torch.nn.Module] = [model]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue

        seen.add(id(module))
        if hasattr(module, "chiron_measurement_head"):
            return module

        for attr in ("module", "model", "base_model"):
            child = getattr(module, attr, None)
            if isinstance(child, torch.nn.Module):
                stack.append(child)

    raise AttributeError("Could not find Chiron measurement head on model or wrapped model.")


def attach_chiron_measurement_head(
    model: "torch.nn.Module",
    finetuning_args: "FinetuningArguments",
    *,
    chiron_num_constructs: int,
    resume_from_checkpoint: Optional[str] = None,
    measurement_head_init: Optional[str] = None,
    measurement_num_layers: int = 1,
) -> None:
    if not finetuning_args.use_chiron_measurement_loss:
        return

    if chiron_num_constructs != 109:
        raise ValueError("Chiron measurement heads require `chiron_num_constructs: 109`.")
    if hasattr(model, "chiron_measurement_head"):
        return

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Cannot infer model hidden size for Chiron measurement head.")

    first_param = next(model.parameters())
    model.add_module(
        "chiron_measurement_head",
        ChironMeasurementHead(
            hidden_size=hidden_size,
            num_constructs=chiron_num_constructs,
            device=first_param.device,
            dtype=torch.float32,
            num_layers=measurement_num_layers,
        ),
    )

    head_init_path = None
    if resume_from_checkpoint is not None:
        head_init_path = _chiron_measurement_head_path(resume_from_checkpoint)
    elif measurement_head_init is not None:
        head_init_path = (
            measurement_head_init
            if os.path.isfile(measurement_head_init)
            else _chiron_measurement_head_path(measurement_head_init)
        )
    if head_init_path is not None:
        if os.path.exists(head_init_path):
            state_dict = torch.load(head_init_path, map_location="cpu")
            _load_chiron_measurement_head_state_dict(model.chiron_measurement_head, state_dict)
            logger.info_rank0(f"Loaded Chiron measurement head from {head_init_path}.")
        else:
            raise FileNotFoundError(f"Chiron measurement head init path does not exist: {head_init_path}")


def _find_decoder_layers(model: "torch.nn.Module") -> "torch.nn.ModuleList":
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    expected_layers = getattr(text_config, "num_hidden_layers", None) or getattr(config, "num_hidden_layers", None)
    candidates: list[torch.nn.ModuleList] = []

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            if expected_layers is not None and len(module) == expected_layers:
                return module

            candidates.append(module)

    if candidates:
        return max(candidates, key=len)

    raise ValueError("Could not find decoder layers for Chiron measurement hook.")


def _chiron_person_mean_ema_path(output_dir: str) -> str:
    return os.path.join(output_dir, "chiron_person_mean_ema.pt")


class ChironPersonMeanEMA:
    """Running mean of unit-normalized person representations."""

    def __init__(self, hidden_size: int, beta: float, device: "torch.device") -> None:
        self.beta = float(beta)
        if not math.isfinite(self.beta) or not 0 < self.beta < 1:
            raise ValueError("Chiron EMA beta must be finite and in (0, 1).")
        self.count = 0
        self.mu = torch.zeros(hidden_size, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, boundary_h: "torch.Tensor") -> None:
        persons = F.normalize(boundary_h.detach().float().to(self.mu.device), dim=-1)
        for person in persons:
            self.count += 1
            rate = max(self.beta, 1.0 / self.count)
            delta = person - self.mu
            self.mu += rate * delta

    def state_dict(self) -> dict:
        return {
            "mu": self.mu.detach().cpu().clone(),
            "count": self.count,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"mu", "count", "beta"}:
            raise ValueError("Chiron EMA state requires mu, count, and beta.")
        value = state["mu"]
        if not isinstance(value, torch.Tensor) or value.shape != self.mu.shape or not torch.isfinite(value).all():
            raise ValueError("Chiron EMA mu must be finite and match the hidden dimensions.")
        if type(state["count"]) is not int or state["count"] < 0:
            raise ValueError("Chiron EMA count must be a nonnegative integer.")
        if type(state["beta"]) not in (int, float) or not math.isfinite(state["beta"]) or not 0 < state["beta"] < 1:
            raise ValueError("Chiron EMA beta must be finite and in (0, 1).")
        self.mu.copy_(state["mu"].to(self.mu.device))
        self.count = state["count"]
        self.beta = float(state["beta"])


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        chiron_num_constructs: int = 0,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model
        self.chiron_num_constructs = chiron_num_constructs
        self._chiron_last_loss_pred: Optional[float] = None
        self._chiron_last_loss_measure: Optional[float] = None
        self._chiron_last_loss_total: Optional[float] = None
        self._chiron_measurement_layers: Optional[list[torch.nn.Module]] = None
        self._chiron_person_mean_ema: Optional[ChironPersonMeanEMA] = None
        self._chiron_gather_group: Optional["torch.distributed.ProcessGroup"] = None
        if (
            self._chiron_loss_enabled
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            # gloo: the gather runs between forward and backward, where ZeRO-3 still has async
            # NCCL param collectives in flight. A second NCCL communicator there can interleave
            # with them per-rank and corrupt data or deadlock; CPU-side gloo cannot — the
            # payload is a handful of person vectors, so the roundtrip is negligible.
            self._chiron_gather_group = torch.distributed.new_group(backend="gloo")

        if self._chiron_loss_enabled:
            measurement_model = _get_chiron_measurement_model(self.model)
            measurement_head = measurement_model.chiron_measurement_head
            self._chiron_person_mean_ema = ChironPersonMeanEMA(
                measurement_head.hidden_size,
                self.finetuning_args.chiron_person_mean_ema_beta,
                measurement_head.person_mean.device,
            )
            self._chiron_person_mean_ema.mu.copy_(measurement_head.person_mean)
            self._load_chiron_person_mean_ema_if_available()
            measurement_head.set_person_mean(self._chiron_person_mean_ema.mu)

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @property
    def _chiron_loss_enabled(self) -> bool:
        return self.finetuning_args.use_chiron_measurement_loss

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self._chiron_loss_enabled:
            return_outputs = kwargs.get("return_outputs", False)
            labels = inputs.pop("labels")
            measurement_pos = inputs.pop("chiron_measurement_pos", None)
            construct_ids = inputs.pop("chiron_measurement_construct_ids", None)
            scores = inputs.pop("chiron_measurement_scores", None)
            measurement_mask = inputs.pop("chiron_measurement_mask", None)
            shift_labels = labels[..., 1:].contiguous()
            valid_label_count = shift_labels.ne(IGNORE_INDEX).sum()
            measurement_only_batch = valid_label_count == 0
            if measurement_only_batch and "logits_to_keep" not in inputs:
                inputs["logits_to_keep"] = 1

            measurement_model = _get_chiron_measurement_model(model)
            measurement_head = measurement_model.chiron_measurement_head
            if self._chiron_measurement_layers is None:
                decoder_layers = _find_decoder_layers(model)
                self._chiron_measurement_layers = list(decoder_layers[-measurement_head.num_layers :])
                if len(self._chiron_measurement_layers) != measurement_head.num_layers:
                    raise ValueError(
                        f"Requested {measurement_head.num_layers} Chiron layers, "
                        f"but model exposes {len(decoder_layers)} decoder layers."
                    )

            captured_by_layer: list[Optional[torch.Tensor]] = [None] * len(self._chiron_measurement_layers)

            def capture_layer_output(layer_index, _module, _inputs, output):
                captured_by_layer[layer_index] = output[0] if isinstance(output, tuple) else output

            hook_handles = [
                layer.register_forward_hook(partial(capture_layer_output, layer_index))
                for layer_index, layer in enumerate(self._chiron_measurement_layers)
            ]

            try:
                outputs = model(**inputs)
            finally:
                for hook_handle in hook_handles:
                    hook_handle.remove()

            if any(value is None for value in captured_by_layer):
                raise ValueError("Chiron measurement hook did not capture every requested layer.")

            hidden_states = torch.stack([value for value in captured_by_layer if value is not None], dim=2)
            if measurement_only_batch:
                loss_pred = hidden_states.sum() * 0.0
            else:
                # Only materialize logits for supervised target tokens.  Converting the
                # full [batch, sequence, vocab] tensor to float32 costs several GiB for
                # long contexts even though prompt labels are all IGNORE_INDEX.
                shift_logits = outputs.logits[..., :-1, :]
                valid_label_mask = shift_labels.ne(IGNORE_INDEX)
                supervised_logits = shift_logits[valid_label_mask]
                supervised_labels = shift_labels[valid_label_mask]
                loss_pred = F.cross_entropy(
                    supervised_logits.float(),
                    supervised_labels,
                    reduction="sum",
                ) / valid_label_count.clamp_min(1)

            loss_measure = loss_pred.new_zeros(())
            if measurement_mask is not None and self.finetuning_args.use_chiron_measurement_loss:
                row_mask = measurement_mask.sum(dim=1) > 0
            else:
                row_mask = torch.zeros(hidden_states.size(0), device=hidden_states.device, dtype=torch.bool)

            if not bool(row_mask.any()):
                dummy_h = hidden_states[:1, 0, :, :]
                dummy_construct_ids = torch.zeros((1, 1), device=dummy_h.device, dtype=torch.long)
                loss_measure = measurement_model.chiron_measurement_head(dummy_h, dummy_construct_ids).sum() * 0.0
            else:
                row_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)[row_mask]
                if measurement_pos is None:
                    raise ValueError("Chiron measurement positions are required for measurement rows.")

                row_measurement_pos = measurement_pos[row_mask].to(device=hidden_states.device)
                h = hidden_states[row_indices, row_measurement_pos, :, :]
                row_construct_ids = construct_ids[row_mask]
                pred = measurement_model.chiron_measurement_head(h, row_construct_ids)
                row_scores = scores[row_mask].to(device=pred.device, dtype=pred.dtype)
                row_mask_values = measurement_mask[row_mask].to(device=pred.device, dtype=pred.dtype)
                squared_errors = (pred - row_scores) ** 2 * row_mask_values
                loss_measure = squared_errors.sum() / row_mask_values.sum().clamp_min(1.0)

            new_h = (
                measurement_head.fuse_hidden(h).detach().float()
                if bool(row_mask.any())
                else hidden_states.new_zeros((0, hidden_states.size(-1)), dtype=torch.float32)
            )
            new_h = self._chiron_gather_persons(new_h)
            if self._chiron_person_mean_ema is None:
                self._chiron_person_mean_ema = ChironPersonMeanEMA(
                    hidden_states.size(-1), self.finetuning_args.chiron_person_mean_ema_beta, hidden_states.device
                )
                self._load_chiron_person_mean_ema_if_available()

            if new_h.size(0) > 0:
                self._chiron_person_mean_ema.update(new_h)
                measurement_head.set_person_mean(self._chiron_person_mean_ema.mu)

            loss = loss_pred + self.finetuning_args.chiron_measurement_lambda * loss_measure
            self._chiron_last_loss_pred = float(loss_pred.detach().float().cpu())
            self._chiron_last_loss_measure = float(loss_measure.detach().float().cpu())
            self._chiron_last_loss_total = float(loss.detach().float().cpu())
            return (loss, outputs) if return_outputs else loss

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    def _chiron_measurement_head_path(self, output_dir: Optional[str] = None) -> str:
        return _chiron_measurement_head_path(output_dir or self.args.output_dir)

    def _chiron_gather_persons(self, person_h: "torch.Tensor") -> "torch.Tensor":
        """All-gather per-rank boundary vectors (variable count) so every rank's EMA stays in sync."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return person_h

        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return person_h

        # Dedicated gloo communicator (created in __init__) on CPU tensors — see __init__ note.
        if self._chiron_gather_group is None:
            raise RuntimeError("Chiron gather group missing; expected it to be created in __init__.")

        person_h_cpu = person_h.detach().cpu()
        count = torch.tensor([person_h_cpu.size(0)])
        counts = [torch.zeros_like(count) for _ in range(world_size)]
        torch.distributed.all_gather(counts, count, group=self._chiron_gather_group)
        max_count = int(max(c.item() for c in counts))
        if max_count > 1_000_000:
            raise RuntimeError(f"Chiron person gather got a corrupt count ({max_count}); collective desync.")
        if max_count == 0:
            return person_h

        padded = person_h_cpu.new_zeros((max_count, person_h_cpu.size(1)))
        padded[: person_h_cpu.size(0)] = person_h_cpu
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, padded, group=self._chiron_gather_group)
        merged = torch.cat([g[: int(c.item())] for g, c in zip(gathered, counts)], dim=0)
        return merged.to(device=person_h.device)

    def _chiron_person_mean_ema_path(self, output_dir: Optional[str] = None) -> str:
        return _chiron_person_mean_ema_path(output_dir or self.args.output_dir)

    def _load_chiron_person_mean_ema_if_available(self) -> None:
        checkpoint = self.args.resume_from_checkpoint
        if checkpoint is None or self._chiron_person_mean_ema is None:
            return

        path = self._chiron_person_mean_ema_path(checkpoint)
        if os.path.exists(path):
            self._chiron_person_mean_ema.load_state_dict(torch.load(path, map_location="cpu"))
            logger.info_rank0(f"Loaded Chiron person mean EMA state from {path}.")

    def _load_chiron_measurement_head_if_available(self) -> None:
        checkpoint = self.args.resume_from_checkpoint
        if checkpoint is None:
            return

        path = self._chiron_measurement_head_path(checkpoint)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            self._load_chiron_measurement_head_state_dict(state_dict)

    def _load_chiron_measurement_head_state_dict(self, state_dict: dict[str, "torch.Tensor"]) -> None:
        measurement_model = _get_chiron_measurement_model(self.model)
        _load_chiron_measurement_head_state_dict(measurement_model.chiron_measurement_head, state_dict)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._chiron_loss_enabled and self._chiron_last_loss_total is not None:
            logs = {
                **logs,
                "chiron_loss_pred": self._chiron_last_loss_pred,
                "chiron_loss_total": self._chiron_last_loss_total,
            }
            if self._chiron_loss_enabled:
                logs["chiron_loss_measure"] = self._chiron_last_loss_measure
        return super().log(logs, *args, **kwargs)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self._chiron_loss_enabled:
            measurement_model = _get_chiron_measurement_model(self.model)
            state_dict = _clone_chiron_measurement_head_state_dict(measurement_model.chiron_measurement_head)
            if not self.is_world_process_zero():
                return

            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            torch.save(state_dict, self._chiron_measurement_head_path(output_dir))
            if self._chiron_person_mean_ema is not None:
                torch.save(self._chiron_person_mean_ema.state_dict(), self._chiron_person_mean_ema_path(output_dir))

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
