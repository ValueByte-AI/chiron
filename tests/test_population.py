"""Exercise vLLM request serialization, population batching and response validation."""

import json
import re

import torch

import httpx
import pytest

from chiron.inference import Chiron, PopulationResponseError, population_messages


FIELDS = dict(
    participant_context="I am taking part in a study.",
    experimental_setting="Imagine a hierarchical company.",
    questions=[{"text": "First item?", "options": ["No", "Yes"]},
               {"text": "Second item?", "options": [str(i) for i in range(1, 11)]}],
)


@pytest.fixture
def server(monkeypatch):
    import openai

    original = openai.OpenAI
    calls, control = [], {}

    def respond(request):
        assert str(request.url) == "http://127.0.0.1:8000/v1/chat/completions"
        body = json.loads(request.content)
        calls.append(body)
        choices = control.get("choices", [
            {"index": i, "message": {"role": "assistant", "content": "Q1: Option 2\nQ2: Option 10"},
             "finish_reason": "stop"}
            for i in reversed(range(body["n"]))
        ])
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "chiron", "choices": choices,
        })

    def connect(**kwargs):
        return original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "OpenAI", connect)
    # Explicit connection settings take precedence over unrelated provider settings.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    model = Chiron.from_server(max_retries=0)
    assert not hasattr(model, "model")
    yield model, calls, control
    model.close()


def test_population_prompt_fields():
    messages = population_messages(
        participant_context="I am taking part in a study.",
        experimental_setting="Imagine a hierarchical company.",
        questions=[{"text": "How likely?", "options": ["Unlikely", "Likely"]}],
    )
    assert messages[1]["content"] == (
        "Below are records describing one participant in an experimental scenario.\n"
        "<person_records>\n"
        "- Participant context: I am taking part in a study.\n"
        "- Experimental setting: Imagine a hierarchical company.\n"
        "</person_records>\n\nQuestions:\nQ1.\n"
        "Please predict this participant's response to the following item:\n"
        "How likely?\nChoose exactly one option:\nOption 1: Unlikely\nOption 2: Likely"
    )


@pytest.mark.parametrize("questions", [[], [{"text": "Item", "options": []}],
    [{"text": "Item", "options": ["Only choice"]}],
    [{"text": "Item", "options": ["A", "</person_records>"]}],
    [{"text": "Item", "options": ["A", "B"], "prompt": "override"}]])
def test_population_invalid_fields(questions):
    with pytest.raises(ValueError):
        population_messages(participant_context="Participant", experimental_setting="Setting", questions=questions)


def test_prediction_request_matches_final_protocol(server):
    model, calls, _ = server
    result = model.predict(
        record_context="Below are questionnaire responses.",
        observations=["First record.", "Second record."],
        questions=["First question?", "Second question?"],
    )
    assert result == "Q1: Option 2\nQ2: Option 10"
    assert len(calls) == 1
    body = dict(calls[0])
    messages = body.pop("messages")
    assert body == {
        "model": "chiron", "n": 1, "temperature": 0, "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert messages[1]["content"] == (
        "Below are questionnaire responses.\n<person_records>\n"
        "- First record.\n- Second record.\n</person_records>\n\n"
        "Questions:\nQ1.\nFirst question?\n\nQ2.\nSecond question?"
    )


def test_population_is_one_condition_request(server):
    model, calls, _ = server
    result = model.simulate_population(**FIELDS, n_respondents=103, seed=20261154)
    assert len(result) == 103
    assert all(r["option_ids"] == [2, 10] for r in result)
    assert len(calls) == 1
    body = dict(calls[0])
    body.pop("messages")
    assert body == {
        "model": "chiron", "n": 103, "temperature": 1.0, "top_p": 1.0,
        "max_tokens": 32, "seed": 20261154,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"regex": r"Q1: Option (1|2)\nQ2: Option (1|2|3|4|5|6|7|8|9|10)"},
        "top_k": -1, "stop_token_ids": [248044],
    }


def test_population_restores_choice_order(server):
    model, _, control = server
    control["choices"] = [
        {"index": 1, "message": {"role": "assistant", "content": "Q1: Option 2\nQ2: Option 10"}, "finish_reason": "stop"},
        {"index": 0, "message": {"role": "assistant", "content": "Q1: Option 1\nQ2: Option 4"}, "finish_reason": "stop"},
    ]
    result = model.simulate_population(**FIELDS, n_respondents=2)
    assert [r["option_ids"] for r in result] == [[1, 4], [2, 10]]


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2], [0, 1, 2]])
def test_population_rejects_incomplete_or_duplicate_batches(server, indices):
    model, _, control = server
    control["choices"] = [
        {"index": i, "message": {"role": "assistant", "content": "Q1: Option 2\nQ2: Option 10"}, "finish_reason": "stop"}
        for i in indices
    ]
    with pytest.raises(PopulationResponseError) as error:
        model.simulate_population(**FIELDS, n_respondents=2)
    assert len(error.value.completions) == len(indices)


@pytest.mark.parametrize("text,finish", [
    ("Q1: Option 2", "length"), ("Q1: Option 2\nQ2: Option 10", "length"),
    ("Q1: Option 2\nQ2: Option 11", "stop"), ("Q2: Option 2\nQ1: Option 1", "stop"),
    (None, "stop"), ("", "stop"),
])
def test_population_rejects_truncated_or_invalid_answers(server, text, finish):
    model, _, control = server
    control["choices"] = [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]
    with pytest.raises(PopulationResponseError) as error:
        model.simulate_population(**FIELDS, n_respondents=1)
    assert error.value.completions[0]["text"] == text


def test_explicit_token_budget_and_backend_guard(server):
    model, calls, _ = server
    model.simulate_population(**FIELDS, n_respondents=2, max_new_tokens=64)
    assert calls[-1]["max_tokens"] == 64
    with pytest.raises(RuntimeError, match="from_pretrained"):
        model.measure(["A record."])
    for kwargs in [{"n_respondents": 0}, {"n_respondents": True},
                   {"n_respondents": 1, "seed": -1}, {"n_respondents": 1, "max_new_tokens": 0}]:
        with pytest.raises(ValueError):
            model.simulate_population(**FIELDS, **kwargs)


@pytest.fixture
def tiny_population_model():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {text: i for i, text in enumerate(["[PAD]", "[EOS]", *alphabet])}
    raw = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role'] + '\n' + m['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}"
    config = LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=2048, eos_token_id=1, pad_token_id=0)
    obj = Chiron()
    obj.model = LlamaForCausalLM(config).eval()
    obj.tokenizer = tokenizer
    obj.max_length = 2048
    return obj


def test_population_sampling_and_constraints(tiny_population_model, monkeypatch):
    model = tiny_population_model
    original = model.model.generate
    prompts, processors = [], []

    def generate(**kwargs):
        config = kwargs["generation_config"]
        assert config.do_sample and config.temperature == config.top_p == 1
        assert config.top_k == 0 and config.num_beams == config.num_return_sequences == 1
        prompts.append(kwargs["input_ids"].clone())
        processors.append(kwargs["logits_processor"][0])
        return original(**kwargs)

    monkeypatch.setattr(model.model, "generate", generate)
    result = model.simulate_population(
        participant_context="Participant", experimental_setting="Condition",
        questions=[{"text": "First", "options": ["No", "Yes"]},
                   {"text": "Second", "options": [str(i) for i in range(1, 11)]}],
        n_respondents=3, max_new_tokens=64,
    )
    assert len(result) == 3 and len({id(p) for p in processors}) == 3
    assert all(torch.equal(prompts[0], prompt) for prompt in prompts)
    for respondent in result:
        assert re.fullmatch(r"Q1: Option [12]\nQ2: Option ([1-9]|10)", respondent["raw_output"])
        assert respondent["option_ids"] == [int(x) for x in re.findall(r"Option (\d+)", respondent["raw_output"])]


def test_population_rejects_incomplete_response(tiny_population_model):
    with pytest.raises(RuntimeError, match="Incomplete"):
        tiny_population_model.simulate_population(
            participant_context="Participant", experimental_setting="Condition",
            questions=[{"text": "Item", "options": ["No", "Yes"]}],
            n_respondents=1, max_new_tokens=1,
        )


def test_population_token_budget_and_count(tiny_population_model, monkeypatch):
    model = tiny_population_model
    def generate(**kwargs):
        assert kwargs['generation_config'].max_new_tokens == 32
        answer = model.tokenizer.encode("Q1: Option 2\nQ2: Option 1", add_special_tokens=False)
        return torch.cat([kwargs['input_ids'], torch.tensor([answer])], dim=1)
    monkeypatch.setattr(model.model, 'generate', generate)
    fields = dict(participant_context="Participant", experimental_setting="Condition",
                  questions=[{"text": "First", "options": ["A", "B"]},
                             {"text": "Second", "options": ["C", "D"]}])
    assert len(model.simulate_population(**fields, n_respondents=2)) == 2
    with pytest.raises(ValueError):
        model.simulate_population(**fields, n_respondents=0)
    model.max_length = 10
    with pytest.raises(ValueError, match='exceeds'):
        model.simulate_population(**fields, n_respondents=1)


def test_population_seed_preserves_rng(tiny_population_model):
    model = tiny_population_model
    fields = dict(
        participant_context="Participant", experimental_setting="Condition",
        questions=[{"text": "Item", "options": ["No", "Yes"]}],
        n_respondents=2, seed=123, max_new_tokens=32,
    )
    before = torch.random.get_rng_state().clone()
    first = model.simulate_population(**fields)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert model.simulate_population(**fields) == first
    assert torch.equal(torch.random.get_rng_state(), before)
