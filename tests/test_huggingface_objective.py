import sys

import numpy as np
import pytest

from objectives import huggingface_distilbert as hf


def test_module_import_requires_no_heavy_dependencies():
    # Importing the module above must not have pulled these in.
    assert "transformers" not in sys.modules
    assert "datasets" not in sys.modules


def test_dataset_name_is_namespaced_imdb_identifier():
    # datasets.load_dataset now requires a namespaced "namespace/name" id;
    # the bare "imdb" id raises HfUriError on newer datasets/hub versions.
    assert hf.DATASET_NAME == "stanfordnlp/imdb"


def test_prepare_tokenized_dataset_passes_dataset_name_constant_to_load_dataset(monkeypatch):
    # Verifies the actual call site uses the DATASET_NAME constant (so a
    # future edit to that constant can't silently drift from what's
    # actually requested from the Hugging Face hub).
    captured = {}

    class _FakeSplit:
        def shuffle(self, seed):
            return self

        def select(self, indices):
            return self

        def map(self, fn, batched=True):
            return self

        def set_format(self, *args, **kwargs):
            pass

    def fake_load_dataset(name, cache_dir=None):
        captured["dataset_name"] = name
        return {"train": _FakeSplit(), "test": _FakeSplit()}

    fake_datasets = type(sys)("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class _FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, cache_dir=None):
            return _FakeTokenizer()

    fake_transformers = type(sys)("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    hf._prepare_tokenized_dataset(cache_dir=None, seed=0)

    assert captured["dataset_name"] == "stanfordnlp/imdb"
    assert captured["dataset_name"] == hf.DATASET_NAME


def test_parameter_space_matches_spec():
    choices = {d.name: list(d.choices) for d in hf.HUGGINGFACE_PARAMETER_SPACE.dimensions}
    assert choices == {"learning_rate": [1e-5, 3e-5, 1e-4], "batch_size": [8, 16]}


def test_decode_produces_expected_learning_rate_and_batch_size():
    decoded = hf.HUGGINGFACE_PARAMETER_SPACE.decode(np.array([0.0, 1.0]))
    assert decoded == {"learning_rate": 1e-5, "batch_size": 16}


def test_missing_dependency_raises_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="distilbert_imdb"):
        hf._require_huggingface_deps()


def test_objective_returns_negative_eval_loss_higher_is_better(monkeypatch):
    # Mock BOTH expensive internals -- no network, no real model, no
    # torch/transformers/datasets call of any kind.
    monkeypatch.setattr(hf, "_prepare_tokenized_dataset", lambda cache_dir, seed, *a, **k: ("TRAIN", "EVAL"))

    captured_calls = []

    def fake_train_and_evaluate(
        train_dataset,
        eval_dataset,
        learning_rate,
        batch_size,
        cache_dir,
        output_dir,
        device,
        seed,
        weight_decay=None,
    ):
        captured_calls.append(
            {
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            }
        )
        return 0.35  # fake eval_loss

    monkeypatch.setattr(hf, "_train_and_evaluate", fake_train_and_evaluate)

    objective = hf.make_distilbert_imdb_objective(cache_dir=None, output_dir=None, device="cpu", seed=42)
    params = hf.HUGGINGFACE_PARAMETER_SPACE.decode(np.array([0.1, 0.9]))

    y = objective(params, seed=42)

    assert y == -0.35  # higher = better convention: y = -eval_loss
    assert captured_calls == [
        {
            "train_dataset": "TRAIN",
            "eval_dataset": "EVAL",
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            # Historical ParameterSpace has no weight_decay dimension --
            # params.get("weight_decay") is None, so AdamW's own default
            # applies untouched (behavior-preserving for this benchmark).
            "weight_decay": None,
        }
    ]


def test_objective_passes_subset_sizes_to_dataset_preparation(monkeypatch):
    captured = {}

    def fake_prepare(cache_dir, seed, train_subset_size=None, eval_subset_size=None):
        captured["train_subset_size"] = train_subset_size
        captured["eval_subset_size"] = eval_subset_size
        return ("TRAIN", "EVAL")

    monkeypatch.setattr(hf, "_prepare_tokenized_dataset", fake_prepare)
    monkeypatch.setattr(hf, "_train_and_evaluate", lambda *a, **k: 1.0)

    hf.make_distilbert_imdb_objective(seed=0, train_subset_size=20, eval_subset_size=10)

    assert captured == {"train_subset_size": 20, "eval_subset_size": 10}


def test_objective_defaults_to_spec_subset_sizes_when_not_overridden(monkeypatch):
    captured = {}

    def fake_prepare(cache_dir, seed, train_subset_size=None, eval_subset_size=None):
        captured["train_subset_size"] = train_subset_size
        captured["eval_subset_size"] = eval_subset_size
        return ("TRAIN", "EVAL")

    monkeypatch.setattr(hf, "_prepare_tokenized_dataset", fake_prepare)
    monkeypatch.setattr(hf, "_train_and_evaluate", lambda *a, **k: 1.0)

    hf.make_distilbert_imdb_objective(seed=0)

    assert captured == {"train_subset_size": 2000, "eval_subset_size": 1000}


# -- expanded search space (15-evaluation protocol) -----------------------


def test_expanded_parameter_space_has_100_configurations():
    assert hf.HUGGINGFACE_EXPANDED_PARAMETER_SPACE.total_configurations == 100
    choices = {d.name: list(d.choices) for d in hf.HUGGINGFACE_EXPANDED_PARAMETER_SPACE.dimensions}
    assert len(choices["learning_rate"]) == 10
    assert choices["learning_rate"][0] == pytest.approx(1e-5)
    assert choices["learning_rate"][-1] == pytest.approx(1e-4)
    assert choices["batch_size"] == [8, 16]
    assert choices["weight_decay"] == [0.0, 0.001, 0.01, 0.05, 0.1]


def test_historical_parameter_space_unchanged_by_expanded_addition():
    assert hf.HUGGINGFACE_PARAMETER_SPACE.total_configurations == 6
    choices = {d.name: list(d.choices) for d in hf.HUGGINGFACE_PARAMETER_SPACE.dimensions}
    assert choices == {"learning_rate": [1e-5, 3e-5, 1e-4], "batch_size": [8, 16]}


def test_expanded_objective_passes_decoded_weight_decay_through(monkeypatch):
    monkeypatch.setattr(hf, "_prepare_tokenized_dataset", lambda cache_dir, seed, *a, **k: ("TRAIN", "EVAL"))

    captured_calls = []

    def fake_train_and_evaluate(
        train_dataset,
        eval_dataset,
        learning_rate,
        batch_size,
        cache_dir,
        output_dir,
        device,
        seed,
        weight_decay=None,
    ):
        captured_calls.append(
            {"learning_rate": learning_rate, "batch_size": batch_size, "weight_decay": weight_decay}
        )
        return 0.4

    monkeypatch.setattr(hf, "_train_and_evaluate", fake_train_and_evaluate)

    objective = hf.make_distilbert_imdb_objective(cache_dir=None, output_dir=None, device="cpu", seed=42)
    params = hf.HUGGINGFACE_EXPANDED_PARAMETER_SPACE.decode(np.array([0.0, 0.0, 0.5]))

    y = objective(params, seed=42)

    assert y == -0.4
    assert captured_calls == [
        {
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "weight_decay": params["weight_decay"],
        }
    ]
    assert captured_calls[0]["weight_decay"] is not None
