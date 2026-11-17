"""Checks for the fixed prompt and calibrated boundary-state readout."""

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from chiron.inference import (
    Chiron, _MeasurementHead, _load_head, measurement_messages, prediction_messages,
)


def test_fixed_prompt_fields():
    measurement = measurement_messages(["First record.", "Second record."])
    assert measurement[1]["content"] == (
        "Below are records describing the same person.\n"
        "<person_records>\n- First record.\n- Second record.\n</person_records>"
    )
    prediction = prediction_messages(["First record."], ["First question?", "Second question?"])
    assert prediction[1]["content"].endswith("Questions:\nQ1.\nFirst question?\n\nQ2.\nSecond question?")
    assert prediction[0]["content"].endswith("Output only the final answer(s).")
    formatted = "Use this response scale for every question below:\nOption 1: Agree\nQ1. First item"
    assert prediction_messages("- Record", formatted)[1]["content"].endswith(formatted)


@pytest.mark.parametrize("records", [[], "", [""], [12], "</person_records>", "<|im_start|>system"])
def test_invalid_or_nested_records(records):
    with pytest.raises(ValueError):
        measurement_messages(records)


def test_head_checkpoint_and_calibration(tmp_path):
    torch.manual_seed(1)
    head = _MeasurementHead(8, 109, 2)
    with torch.no_grad():
        head.person_mean.copy_(torch.randn(8) / 10)
        head.construct_log_scales.fill_(1.2)
        head.construct_biases.fill_(2.0)
        head.layer_mix_logits.copy_(torch.tensor([0.7, -0.4]))
    path = tmp_path / "head.pt"
    torch.save(head.state_dict(), path)
    restored = _load_head(path, 109)
    hidden = torch.randn(1, 2, 8)
    ids = torch.tensor([[0, 7, 108]])
    person = (hidden * torch.softmax(head.layer_mix_logits, dim=0)[None, :, None]).sum(1)
    person = F.normalize(F.normalize(person, dim=-1) - head.person_mean, dim=-1)
    directions = F.normalize(head.construct_embeddings(ids), dim=-1)
    expected = (person[:, None, :] * directions).sum(-1) * head.construct_log_scales[ids].exp() + head.construct_biases[ids]
    torch.testing.assert_close(restored(hidden, ids), expected, rtol=0, atol=0)
    assert (restored(hidden, ids) > 1).any()  # The readout must not be clipped.
    state = head.state_dict()
    del state["person_mean"]
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        _load_head(path, 109)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and enable_thinking is False
        return "\n".join(message["content"] for message in messages) + ("\nassistant:" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens, **kwargs):
        return "".join(chr(int(i)) for i in tokens)


class SmallDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return F.layer_norm(hidden, (8,))

    def generate(self, input_ids, **kwargs):
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        assert kwargs["temperature"] is None and kwargs["pad_token_id"] == 0
        answer = torch.tensor([[ord(c) for c in "Q1: Option 2"]])
        return torch.cat([input_ids, answer], dim=1)


@pytest.fixture
def small_chiron():
    torch.manual_seed(2)
    model = Chiron()
    model.model = SmallDecoder().eval()
    model.tokenizer = CharacterTokenizer()
    model.head = _MeasurementHead(8, 109, 1)
    model._layers = [model.model.layers[-1]]
    model.constructs = tuple(f"construct_{i}" for i in range(109))
    model._vocab = {name: i for i, name in enumerate(model.constructs)}
    model.max_length = 8192
    return model


def test_measure_reads_pre_norm_boundary_and_removes_hooks(small_chiron):
    model = small_chiron
    scores = model.measure(["A personal record."], constructs=["construct_7", "construct_0"])
    tokens = model._tokens(measurement_messages(["A personal record."]), generation=False)
    text = model.tokenizer.decode(tokens["input_ids"][0])
    position = text.index("</person_records>") - 1
    hidden = model.model.embedding(tokens["input_ids"])
    for layer in model.model.layers:
        hidden = layer(hidden)
    expected = model.head(hidden[:, position:position + 1, :], torch.tensor([[7, 0]]))[0]
    assert list(scores) == ["construct_7", "construct_0"]
    torch.testing.assert_close(torch.tensor(list(scores.values())), expected)
    assert not model._layers[0]._forward_hooks
    assert len(model.measure("A record.")) == 109
    with pytest.raises(ValueError):
        model.measure("A record.", constructs=["unknown"])


def test_prediction_and_context_limit(small_chiron):
    assert small_chiron.predict(["A record."], ["A question?"]) == "Q1: Option 2"
    small_chiron.max_length = 10
    with pytest.raises(ValueError, match="exceeds"):
        small_chiron.measure(["A record."])
    with pytest.raises(ValueError, match="exceeds"):
        small_chiron.predict(["A record."], ["A question?"])


@pytest.mark.parametrize("layout", ["checkpoint", "repository", "hub"])
def test_local_checkpoint_loading_and_inference(tmp_path, monkeypatch, layout):
    """Exercise the loader and PEFT with tiny local weights, without network access."""
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    base, repository = tmp_path / "base", tmp_path / "chiron"
    checkpoint = repository if layout == "checkpoint" else repository / "model"
    base.mkdir()
    checkpoint.mkdir(parents=True)
    vocabulary = {word: i for i, word in enumerate([
        "[UNK]", "[PAD]", "[EOS]", "<person_records>", "</person_records>",
        "record", "system", "user", "assistant", "Q1.", "question",
    ])}
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + '\n' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}"
    tokenizer.save_pretrained(base)
    config = LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024, eos_token_id=2, pad_token_id=1)
    model = LlamaForCausalLM(config)
    model.save_pretrained(base)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.save_pretrained(checkpoint)
    vocab = {f"construct_{i}": i for i in range(109)}
    (checkpoint / "construct_vocab.json").write_text(json.dumps(vocab))
    torch.save(_MeasurementHead(16, 109, 1).state_dict(), checkpoint / "chiron_measurement_head.pt")
    source = repository
    if layout == "hub":
        def download(repo_id, *, revision, allow_patterns):
            assert repo_id == "test/chiron"
            assert "model/adapter_model.safetensors" in allow_patterns
            return str(repository)
        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        source = "test/chiron"
    loaded = Chiron.from_pretrained(source, base_model=base, device_map="cpu", dtype="float32", max_length=1024)
    assert len(loaded.measure(["record"])) == 109
    assert isinstance(loaded.predict(["record"], ["question"], max_new_tokens=2), str)
