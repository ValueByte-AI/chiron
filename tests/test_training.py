import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LLaMA-Factory/src"))

from llamafactory.data.converter import ChironJointDatasetConverter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments, FinetuningArguments
from llamafactory.train.sft.trainer import (
    ChironMeasurementHead,
    ChironPersonMeanEMA,
    CustomSeq2SeqTrainer,
)


def test_person_mean_update_and_resume(tmp_path):
    torch.manual_seed(10)
    hidden = torch.randn(250, 8)
    ema = ChironPersonMeanEMA(8, 0.01, torch.device("cpu"))
    ema.update(hidden[:80])
    torch.testing.assert_close(ema.mu, F.normalize(hidden[:80], dim=-1).mean(0))
    path = tmp_path / "chiron_person_mean_ema.pt"
    torch.save(ema.state_dict(), path)
    state = torch.load(path, weights_only=True)
    assert set(state) == {"mu", "count", "beta"}
    restored = ChironPersonMeanEMA(8, 0.1, torch.device("cpu"))
    restored.load_state_dict(state)
    ema.update(hidden[80:])
    restored.update(hidden[80:])
    torch.testing.assert_close(ema.mu, restored.mu, rtol=0, atol=0)
    assert ema.count == restored.count == 250
    assert restored.beta == 0.01
    expected = torch.zeros(8)
    for index, person in enumerate(F.normalize(hidden, dim=-1), 1):
        expected += max(0.01, 1 / index) * (person - expected)
    torch.testing.assert_close(ema.mu, expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0, 1, float("nan")])
def test_invalid_person_mean_rate(beta):
    with pytest.raises(ValueError):
        ChironPersonMeanEMA(8, beta, torch.device("cpu"))
    with pytest.raises(ValueError):
        FinetuningArguments(chiron_person_mean_ema_beta=beta)


def test_joint_converter_keeps_measurement_and_generation_separate(tmp_path):
    vocab = {f"construct_{i}": i for i in range(109)}
    path = tmp_path / "construct_vocab.json"
    path.write_text(json.dumps(vocab))
    attributes = DatasetAttr(load_from="file", dataset_name="joint")
    for name, value in {
        "formatting": "chiron_joint", "messages": "messages", "role_tag": "role",
        "content_tag": "content", "system_tag": "system", "user_tag": "user",
        "assistant_tag": "assistant",
    }.items():
        setattr(attributes, name, value)
    converter = ChironJointDatasetConverter(attributes, DataArguments(chiron_construct_vocab=str(path)))
    for signal in ("measurement", "prediction", "population"):
        row = {
            "sample_id": "synthetic:1", "domain": "synthetic", "signal": signal,
            "messages": [
                {"role": "system", "content": "Use only the supplied records."},
                {"role": "user", "content": "<person_records>\nI plan ahead.\n</person_records>"},
                {"role": "assistant", "content": "" if signal == "measurement" else "Q1: Option 2"},
            ],
            "measurement": {"targets": [
                {"construct_id": 0, "construct": "construct_0", "score": 0.4}
            ] if signal == "measurement" else []},
        }
        converted = converter(row)
        assert converted["_chiron_repr_only"] is (signal == "measurement")
        assert converted["_measurement_construct_ids"] == ([0] if signal == "measurement" else [])
        assert not any("pair" in key for key in converted)


class SmallTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=1)
        self.embedding = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8)])
        self.lm_head = nn.Linear(8, 16)
        self.chiron_measurement_head = ChironMeasurementHead(
            hidden_size=8, num_constructs=109, device=torch.device("cpu"), dtype=torch.float32,
        )

    def forward(self, input_ids, **kwargs):
        hidden = self.layers[0](self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_joint_loss_backward_updates_only_measurement_mean():
    torch.manual_seed(11)
    model = SmallTrainingModel()
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.finetuning_args = FinetuningArguments(use_chiron_measurement_loss=True, chiron_measurement_lambda=8)
    trainer._chiron_measurement_layers = None
    trainer._chiron_person_mean_ema = ChironPersonMeanEMA(8, 0.01, torch.device("cpu"))
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, -100], [-100, -100, 9, 10]])
    hidden = model.layers[0](model.embedding(ids))
    readout = model.chiron_measurement_head(hidden[:1, 2], torch.tensor([[0]]))
    expected_measurement = (readout - 0.4).square().mean()
    shifted = labels[:, 1:]
    expected_prediction = F.cross_entropy(model(ids).logits[:, :-1][shifted != -100], shifted[shifted != -100])
    expected = expected_prediction + 8 * expected_measurement
    inputs = {
        "input_ids": ids, "labels": labels,
        "chiron_measurement_pos": torch.tensor([2, -1]),
        "chiron_measurement_construct_ids": torch.tensor([[0], [0]]),
        "chiron_measurement_scores": torch.tensor([[0.4], [0.0]]),
        "chiron_measurement_mask": torch.tensor([[1.0], [0.0]]),
    }
    loss = trainer.compute_loss(model, inputs)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.chiron_measurement_head.construct_embeddings.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad.abs().sum() > 0
    assert trainer._chiron_person_mean_ema.count == 1
    torch.testing.assert_close(model.chiron_measurement_head.person_mean, F.normalize(hidden[0, 2].detach(), dim=0))
