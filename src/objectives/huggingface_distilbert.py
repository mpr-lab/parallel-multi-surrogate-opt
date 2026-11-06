from pathlib import Path
from typing import Callable

import numpy as np

from parameter_space import ChoiceDimension, ParameterSpace

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
N_TRAIN_EXAMPLES = 2000
N_EVAL_EXAMPLES = 1000
MAX_TOKEN_LENGTH = 128
NUM_EPOCHS = 1

# Exact historical grid: learning_rate = {1e-5, 3e-5, 1e-4}, batch_size =
# {8, 16}. Treated as discrete choices (not a continuous range) for
# consistency with every other workload's discrete-choice treatment.
# This is the HISTORICAL profiling benchmark's space -- unchanged.
HUGGINGFACE_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("learning_rate", [1e-5, 3e-5, 1e-4]),
        ChoiceDimension("batch_size", [8, 16]),
    ]
)

# "Expanded" search space for the matched 15-evaluation experiment
# protocol (n_initial=5, n_rounds=5, batch_size=2). A SEPARATE
# ParameterSpace, not a change to the historical grid above. Reuses the
# identical "learning_rate"/"batch_size" decoded names plus one new
# "weight_decay" dimension -- make_distilbert_imdb_objective (unchanged
# below) handles weight_decay's absence from the historical space by
# falling back to AdamW's own default (see _train_and_evaluate).
# 10 log-spaced learning rates x 2 batch sizes x 5 weight decays -> 100.
HUGGINGFACE_EXPANDED_LEARNING_RATE_VALUES = np.logspace(-5, -4, 10).tolist()
HUGGINGFACE_EXPANDED_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("learning_rate", HUGGINGFACE_EXPANDED_LEARNING_RATE_VALUES),
        ChoiceDimension("batch_size", [8, 16]),
        ChoiceDimension("weight_decay", [0.0, 0.001, 0.01, 0.05, 0.1]),
    ]
)


def _require_huggingface_deps() -> None:
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        # The common, specific case: one of the packages isn't installed.
        raise ImportError(
            "The distilbert_imdb benchmark requires the optional "
            "'transformers', 'datasets', and 'torch' packages. Install "
            "them with: pip install -r requirements-huggingface.txt"
        ) from exc
    except Exception as exc:
        # torch in particular can be pip-installed but fail to load (e.g.
        # a wheel built against a different NumPy ABI, as seen on this
        # project's Intel-macOS dev environment), raising something other
        # than ImportError. We can't name that failure mode's exact type
        # without importing the very package that's failing, so this
        # broader catch -- scoped to only the three imports above -- is
        # deliberate, not a catch-all, and chains the original exception
        # so the real cause is never silently relabeled as "not installed".
        raise ImportError(
            "One of the distilbert_imdb benchmark's optional dependencies "
            "('transformers', 'datasets', 'torch') is installed but failed "
            "to load. See the chained exception below for the underlying "
            "error (on this project's Intel-macOS dev setup, a NumPy ABI "
            "mismatch with the torch wheel is the known cause -- see "
            "requirements-huggingface.txt)."
        ) from exc


def _prepare_tokenized_dataset(
    cache_dir: Path | None,
    seed: int | None,
    train_subset_size: int = N_TRAIN_EXAMPLES,
    eval_subset_size: int = N_EVAL_EXAMPLES,
):
    """One-time: load the IMDB subset + tokenizer, tokenize both splits.
    Heavy imports happen only here (lazily), never at module import time,
    so importing this module never requires transformers/datasets/torch.
    """
    _require_huggingface_deps()
    import datasets
    from transformers import AutoTokenizer

    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str)

    raw = datasets.load_dataset(DATASET_NAME, cache_dir=cache_dir_str)
    shuffle_seed = seed if seed is not None else 0
    train_subset = raw["train"].shuffle(seed=shuffle_seed).select(range(train_subset_size))
    eval_subset = raw["test"].shuffle(seed=shuffle_seed).select(range(eval_subset_size))

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_TOKEN_LENGTH
        )

    train_tokenized = train_subset.map(tokenize, batched=True)
    eval_tokenized = eval_subset.map(tokenize, batched=True)
    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return train_tokenized, eval_tokenized


def _train_and_evaluate(
    train_dataset,
    eval_dataset,
    learning_rate: float,
    batch_size: int,
    cache_dir: Path | None,
    output_dir: Path | None,
    device: str,
    seed: int | None,
    weight_decay: float | None = None,
) -> float:
    """The expensive path: a FRESH pretrained model, one epoch of
    training, eval-set loss. Kept as its own module-level function
    specifically so tests can monkeypatch it directly and verify the
    objective's wiring without ever touching transformers/torch/datasets.

    `weight_decay` is optional: the historical distilbert_imdb benchmark's
    decoded params never include it, so it's None there and AdamW's own
    default applies (this function never overrides it in that case) --
    behavior-preserving for that benchmark. The expanded benchmark's
    ParameterSpace does include it, in which case it's passed through
    explicitly.
    """
    _require_huggingface_deps()
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    if seed is not None:
        torch.manual_seed(seed)

    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, cache_dir=cache_dir_str
    )
    model.to(device)
    model.train()

    optimizer_kwargs = {"lr": learning_rate}
    if weight_decay is not None:
        optimizer_kwargs["weight_decay"] = weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for _ in range(NUM_EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()

    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            n = batch["label"].shape[0]
            total_loss += outputs.loss.item() * n
            total_examples += n

    eval_loss = total_loss / total_examples

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)

    return eval_loss


def make_distilbert_imdb_objective(
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    device: str = "cpu",
    seed: int | None = None,
    train_subset_size: int = N_TRAIN_EXAMPLES,
    eval_subset_size: int = N_EVAL_EXAMPLES,
) -> Callable:
    """One-time dataset/tokenizer preparation; the returned closure does
    ONLY the fresh-model train+eval step per candidate, so WorkerPool's
    existing eval_start_time/eval_end_time window never includes dataset
    loading or tokenization.

    The historical executable objective returned eval_loss (despite one
    comment incorrectly mentioning accuracy). This framework's convention
    is higher = better everywhere, so this returns y = -eval_loss.

    cache_dir/output_dir are plain parameters -- never hard-coded paths --
    so callers can point them at e.g. /work/... on an HPC cluster via
    CLI/config without this module knowing anything about that filesystem.
    """
    train_dataset, eval_dataset = _prepare_tokenized_dataset(
        cache_dir, seed, train_subset_size, eval_subset_size
    )

    def objective(params: dict, seed: int | None = None) -> float:
        eval_loss = _train_and_evaluate(
            train_dataset,
            eval_dataset,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            cache_dir=cache_dir,
            output_dir=output_dir,
            device=device,
            seed=seed,
            # Absent from the historical ParameterSpace's decoded dict ->
            # None -> _train_and_evaluate leaves AdamW's own default
            # untouched (behavior-preserving for distilbert_imdb). Present
            # for distilbert_imdb_expanded -> passed through explicitly.
            weight_decay=params.get("weight_decay"),
        )
        return -float(eval_loss)

    return objective
