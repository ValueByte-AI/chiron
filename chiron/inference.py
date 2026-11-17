"""Torch measurement and Transformers/vLLM generation with fixed task templates."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit

import torch
from torch import nn
from torch.nn import functional as F


_MEASUREMENT_SYSTEM = (
    "You are a psychometric measurement and prediction model.\n\n"
    "You will receive person-level records, such as statements, behaviors, events, or attributes. "
    "Use these records as evidence to infer latent construct intensities."
)
_PREDICTION_SYSTEM = (
    "You are a psychometric measurement and prediction model.\n\n"
    "You will receive person-level records, such as statements, behaviors, events, or attributes. "
    "Use these records as evidence to answer the prediction questions about that person.\n\n"
    "Output only the final answer(s)."
)
_DEFAULT_CONTEXT = "Below are records describing this person."
_CLOSE = "</person_records>"
_RESERVED = ("<person_records>", _CLOSE, "<|im_start|>", "<|im_end|>", "<|endoftext|>")


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if any(marker in value for marker in _RESERVED):
        raise ValueError(f"{name} must not contain record or chat boundary markers")
    return value.strip()


def _block(value: str | Sequence[str], name: str, *, questions: bool = False) -> str:
    if isinstance(value, str):
        return _text(value, name)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be nonempty text or a nonempty list of strings")
    values = [_text(item, name) for item in value]
    if questions:
        return "\n\n".join(f"Q{i}.\n{item}" for i, item in enumerate(values, 1))
    return "\n".join(f"- {item}" for item in values)


def measurement_messages(records: str | Sequence[str]) -> list[dict[str, str]]:
    """Fill the fixed measurement template with personal records."""
    records = _block(records, "records")
    return [
        {"role": "system", "content": _MEASUREMENT_SYSTEM},
        {"role": "user", "content": f"Below are records describing the same person.\n<person_records>\n{records}\n{_CLOSE}"},
    ]


def prediction_messages(
    observations: str | Sequence[str],
    questions: str | Sequence[str],
    *,
    record_context: str = _DEFAULT_CONTEXT,
) -> list[dict[str, str]]:
    """Fill only the three variable fields of the fixed prediction template.

    Lists receive bullet points or Q1/Q2 labels. A string supplies an already
    formatted field, for example questions sharing one questionnaire response scale.
    """
    records = _block(observations, "observations")
    questions = _block(questions, "questions", questions=True)
    context = _text(record_context, "record_context")
    return [
        {"role": "system", "content": _PREDICTION_SYSTEM},
        {"role": "user", "content": f"{context}\n<person_records>\n{records}\n{_CLOSE}\n\nQuestions:\n{questions}"},
    ]


def _population_fields(questions: Sequence[dict]) -> tuple[list[str], str]:
    if not isinstance(questions, (list, tuple)) or not questions:
        raise ValueError("questions must be a nonempty list of item/option dictionaries")
    blocks, patterns = [], []
    for i, question in enumerate(questions, 1):
        if not isinstance(question, dict) or set(question) != {"text", "options"}:
            raise ValueError("Each population question must contain text and options")
        item = _text(question["text"], "question text")
        options = question["options"]
        if not isinstance(options, (list, tuple)) or len(options) < 2:
            raise ValueError("Each question needs at least two ordered option texts")
        options = [_text(option, "option text") for option in options]
        blocks.append("\n".join([
            "Please predict this participant's response to the following item:",
            item,
            "Choose exactly one option:",
            *(f"Option {j}: {option}" for j, option in enumerate(options, 1)),
        ]))
        patterns.append(f"Q{i}: Option ({'|'.join(str(j) for j in range(1, len(options) + 1))})")
    return blocks, r"\n".join(patterns)


def population_messages(
    *, participant_context: str, experimental_setting: str, questions: Sequence[dict],
) -> list[dict[str, str]]:
    """Format an experimental condition with the population task's fixed template.

    Each question has ``text`` and an ordered list of ``options``. Option IDs
    start at 1, independently of the scale values written in the option text.
    """
    blocks, _ = _population_fields(questions)
    return prediction_messages(
        record_context="Below are records describing one participant in an experimental scenario.",
        observations=[
            f"Participant context: {_text(participant_context, 'participant_context')}",
            f"Experimental setting: {_text(experimental_setting, 'experimental_setting')}",
        ],
        questions=blocks,
    )


class _MeasurementHead(nn.Module):
    """The checkpointed, centred cosine readout used in Chiron training."""

    def __init__(self, hidden_size: int, num_constructs: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.construct_embeddings = nn.Embedding(num_constructs, hidden_size)
        self.construct_log_scales = nn.Parameter(torch.zeros(num_constructs))
        self.construct_biases = nn.Parameter(torch.zeros(num_constructs))
        self.register_buffer("person_mean", torch.zeros(hidden_size))
        self.layer_mix_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_mix_logits.float(), dim=0).to(hidden.dtype)
        hidden = torch.sum(hidden * weights.view(1, -1, 1), dim=1)
        person = F.normalize(hidden.float(), dim=-1)
        person = F.normalize(person - self.person_mean.float(), dim=-1)
        constructs = F.normalize(self.construct_embeddings(ids).float(), dim=-1)
        cosine = torch.sum(person.unsqueeze(1) * constructs, dim=-1)
        return self.construct_log_scales[ids].float().exp() * cosine + self.construct_biases[ids].float()


def _load_head(path: Path, num_constructs: int) -> _MeasurementHead:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Measurement head must be a tensor state dictionary")
    embedding, mixture = state.get("construct_embeddings.weight"), state.get("layer_mix_logits")
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] != num_constructs or embedding.shape[1] == 0:
        raise ValueError("Measurement head and construct vocabulary do not match")
    if not isinstance(mixture, torch.Tensor) or mixture.ndim != 1 or mixture.numel() == 0:
        raise ValueError("Measurement head is missing its layer mixture")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Measurement head parameters must be finite tensors")
    head = _MeasurementHead(embedding.shape[1], num_constructs, mixture.numel())
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


def _decoder_layers(model):
    config = getattr(model.config, "text_config", model.config)
    expected = config.num_hidden_layers
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == expected:
            return module
    raise ValueError("Could not find the language model's decoder layers")


class PopulationResponseError(ValueError):
    """Invalid population response, retaining all received completions."""

    def __init__(self, message: str, completions: list[dict]):
        super().__init__(message)
        self.completions = completions


class Chiron:
    """Use Transformers locally or connect generation to a vLLM server.

    ``from_pretrained`` loads the local model for all three tasks; ``from_server`` connects
    prediction and population sampling to the server launched by ``serve.py``.
    """

    @classmethod
    def from_server(
        cls,
        base_url: str = "http://127.0.0.1:8000/v1",
        *,
        api_key: str | None = None,
        timeout: float = 1800.0,
        max_retries: int = 2,
    ) -> "Chiron":
        """Connect to the Chiron vLLM server without loading local model weights."""
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTP(S) endpoint")
        endpoint = urlsplit(base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("base_url must be an HTTP(S) endpoint")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        from openai import OpenAI

        obj = cls()
        obj._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("VLLM_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )
        return obj

    def close(self) -> None:
        """Close the HTTP connection pool created by ``from_server``."""
        if hasattr(self, "_client"):
            self._client.close()

    def _generation_client(self):
        if not hasattr(self, "_client"):
            raise RuntimeError(
                "This instance has no vLLM connection. "
                "Connect with Chiron.from_server(base_url)."
            )
        return self._client

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "chiron-group/chiron",
        *,
        base_model: str | Path | None = None,
        revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 8192,
    ) -> "Chiron":
        """Load the adapter, saved centring state, construct names, and base tokenizer."""
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16 or float32")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        path = Path(checkpoint).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            files = ["adapter_config.json", "adapter_model.safetensors", "chiron_measurement_head.pt", "construct_vocab.json"]
            path = Path(snapshot_download(
                str(checkpoint), revision=revision,
                allow_patterns=files + [f"model/{name}" for name in files],
            ))
        if (path / "model" / "adapter_config.json").is_file():
            path = path / "model"
        vocab = json.loads((path / "construct_vocab.json").read_text())
        if not isinstance(vocab, dict) or len(vocab) != 109 or any(not isinstance(name, str) or not name.strip() or type(index) is not int for name, index in vocab.items()) or sorted(vocab.values()) != list(range(109)):
            raise ValueError("Expected the checkpoint's 109-construct vocabulary")
        head = _load_head(path / "chiron_measurement_head.pt", len(vocab))
        adapter = json.loads((path / "adapter_config.json").read_text())
        base = str(base_model or adapter["base_model_name_or_path"])

        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        config = AutoConfig.from_pretrained(base)
        text_config = getattr(config, "text_config", config)
        if text_config.hidden_size != head.construct_embeddings.embedding_dim:
            raise ValueError("Measurement head and base model have different hidden sizes")
        model_class = AutoModelForImageTextToText if type(config) in AutoModelForImageTextToText._model_mapping else AutoModelForCausalLM
        model = model_class.from_pretrained(
            base, config=config, torch_dtype=getattr(torch, dtype),
            device_map=device_map, low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
        model.eval().requires_grad_(False)
        layers = _decoder_layers(model)
        if head.num_layers > len(layers):
            raise ValueError("Measurement head requests more layers than the base model has")
        obj = cls()
        obj.model = model
        obj.tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, padding_side="right")
        obj.head = head
        obj.constructs = tuple(sorted(vocab, key=vocab.get))
        obj._vocab = vocab
        obj._layers = list(layers[-head.num_layers:])
        obj.max_length = min(max_length, getattr(text_config, "max_position_embeddings", max_length))
        return obj

    def _tokens(self, messages, *, generation: bool):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False,
        )
        tokens = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"Input exceeds {self.max_length} tokens; shorten the records explicitly")
        device = self.model.get_input_embeddings().weight.device
        if device.type == "meta":
            raise ValueError("Input embeddings must reside on CPU or GPU")
        return {key: tokens[key].to(device) for key in ("input_ids", "attention_mask")}

    @torch.inference_mode()
    def measure(
        self, records: str | Sequence[str], *, constructs: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Return calibrated construct scores, preserving the trained, unclipped readout."""
        if not hasattr(self, "model"):
            raise RuntimeError("Measurement uses Torch. Load it with Chiron.from_pretrained(checkpoint).")
        if constructs is None:
            names = list(self.constructs)
        elif isinstance(constructs, (list, tuple)) and constructs and all(isinstance(name, str) for name in constructs):
            names = list(constructs)
        else:
            raise ValueError("constructs must be a nonempty list of construct names")
        if len(set(names)) != len(names) or any(name not in self._vocab for name in names):
            raise ValueError("Use distinct names from model.constructs")
        tokens = self._tokens(measurement_messages(records), generation=False)
        sequence = tokens["input_ids"][0].tolist()
        marker = self.tokenizer.encode(_CLOSE, add_special_tokens=False)
        starts = [i for i in range(len(sequence) - len(marker) + 1) if sequence[i:i + len(marker)] == marker] if marker else []
        if len(starts) != 1 or starts[0] == 0:
            raise ValueError("Could not locate the unique measurement boundary")
        position = starts[0] - 1
        captured = [None] * len(self._layers)

        def capture(index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden[:, position, :].detach()
            return hook

        handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(self._layers)]
        try:
            self.model(**tokens, use_cache=False, logits_to_keep=1)
            if any(value is None for value in captured):
                raise RuntimeError("Could not capture all measurement states")
            device = captured[-1].device
            hidden = torch.stack([value.to(device) for value in captured], dim=1)
            self.head.to(device=device, dtype=torch.float32)
            ids = torch.tensor([[self._vocab[name] for name in names]], device=device)
            scores = self.head(hidden, ids).squeeze(0).float().cpu()
            if not torch.isfinite(scores).all():
                raise ValueError("Measurement head returned nonfinite scores")
            return dict(zip(names, scores.tolist()))
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def predict(
        self,
        observations: str | Sequence[str],
        questions: str | Sequence[str],
        *,
        record_context: str = _DEFAULT_CONTEXT,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate final answers with the fixed prediction prompt and thinking off."""
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not hasattr(self, "_client"):
            tokens = self._tokens(prediction_messages(observations, questions, record_context=record_context), generation=True)
            length = tokens["input_ids"].shape[1]
            if length + max_new_tokens > self.max_length:
                raise ValueError("Input plus requested answer exceeds max_length")
            output = self.model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            )
            return self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
        messages = prediction_messages(observations, questions, record_context=record_context)
        response = self._generation_client().chat.completions.create(
            model="chiron", messages=messages, n=1, temperature=0,
            max_tokens=max_new_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if len(response.choices) != 1 or response.choices[0].index != 0:
            raise ValueError("vLLM must return one prediction completion")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("vLLM returned no nonempty prediction text")
        return content.strip()

    @torch.inference_mode()
    def simulate_population(
        self,
        *,
        participant_context: str,
        experimental_setting: str,
        questions: Sequence[dict],
        n_respondents: int,
        seed: int = 20260902,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        """Sample one condition with Transformers or one batched vLLM request.

        Sampling uses temperature 1, top-p 1, no top-k cutoff and thinking off.
        ``seed`` is the condition seed; the paper uses 20260902 plus the
        condition's original evaluation index. Each respondent answers all
        questions under the same structured-output constraint.
        """
        if type(n_respondents) is not int or n_respondents <= 0:
            raise ValueError("n_respondents must be a positive integer")
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be a nonnegative integer below 2**63")
        messages = population_messages(
            participant_context=participant_context,
            experimental_setting=experimental_setting,
            questions=questions,
        )
        _, pattern = _population_fields(questions)
        if max_new_tokens is None:
            max_new_tokens = 8 * len(questions) + 16
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not hasattr(self, "_client"):
            tokens = self._tokens(messages, generation=True)
            length = tokens["input_ids"].shape[1]
            if length + max_new_tokens > self.max_length:
                raise ValueError("Input plus requested answer exceeds max_length")

            import xgrammar as xgr
            from xgrammar.contrib.hf import LogitsProcessor
            from transformers import GenerationConfig

            eos = self.model.generation_config.eos_token_id
            if eos is None:
                eos = self.tokenizer.eos_token_id
            stops = [eos] if isinstance(eos, int) else eos
            if not stops:
                raise ValueError("The model must define an end-of-sequence token")
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.model.get_output_embeddings().weight.shape[0],
                stop_token_ids=stops,
            )
            grammar = xgr.GrammarCompiler(tokenizer_info).compile_regex(pattern)
            config = GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=True, num_beams=1,
                num_return_sequences=1, temperature=1.0, top_p=1.0, top_k=0,
                eos_token_id=eos,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else stops[0],
                use_cache=True,
            )
            devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == "cuda"})
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    torch.cuda.default_generators[device].manual_seed(seed)
                responses = []
                for _ in range(n_respondents):
                    output = self.model.generate(
                        **tokens, generation_config=config,
                        logits_processor=[LogitsProcessor(grammar)],
                    )
                    text = self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
                    if re.fullmatch(pattern, text) is None:
                        raise RuntimeError("Incomplete population response; increase max_new_tokens")
                    responses.append({
                        "raw_output": text,
                        "option_ids": [int(value) for value in re.findall(r"Q\d+: Option (\d+)", text)],
                    })
                return responses
        response = self._generation_client().chat.completions.create(
            model="chiron", messages=messages, n=n_respondents,
            temperature=1.0, top_p=1.0, max_tokens=max_new_tokens, seed=seed,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"regex": pattern},
                "top_k": -1,
                "stop_token_ids": [248044],
            },
        )
        completions = [
            {"text": choice.message.content, "finish_reason": choice.finish_reason,
             "index": choice.index}
            for choice in response.choices
        ]
        indices = [choice["index"] for choice in completions]
        if (len(completions) != n_respondents
                or any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(n_respondents))):
            raise PopulationResponseError(
                "vLLM did not return exactly n uniquely indexed trajectories", completions,
            )
        result = []
        for completion in sorted(completions, key=lambda x: x["index"]):
            text = completion["text"]
            if (completion["finish_reason"] != "stop" or not isinstance(text, str)
                    or re.fullmatch(pattern, text.strip()) is None):
                raise PopulationResponseError(
                    "Population response must finish normally with one valid option per question",
                    completions,
                )
            result.append({
                "raw_output": text.strip(),
                "option_ids": [int(value) for value in re.findall(r"Q\d+: Option (\d+)", text)],
            })
        return result
