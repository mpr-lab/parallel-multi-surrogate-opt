from typing import Any, Callable

import numpy as np


class SearchSpaceExhaustedError(ValueError):
    """Raised when every configuration in a finite decoded search space
    has already been evaluated (in DataStore history or this batch), so
    a surrogate's allocation cannot be filled with anything new -- a
    distinct, explicit condition from a plain "not enough candidates"
    error, which can also happen for other reasons (see merge_candidates
    docstring).
    """


def merge_candidates(
    proposals: dict[str, np.ndarray],
    allocation: dict[str, int],
    decode_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
    evaluated_configs: set[tuple] | None = None,
    total_configurations: int | None = None,
) -> list[dict]:
    """Combine ranked per-surrogate proposals into one fixed-size batch.

    Sources are processed in the order `allocation` is given. Each source
    fills its own allocation only from its own ranked proposal list,
    skipping any candidate already placed in the batch (by itself or by an
    earlier-processed source), OR already present in `evaluated_configs`
    (prior DataStore history), and advancing to its next-ranked candidate
    instead. A slot is never filled from a source other than the one it
    was allocated to -- allocation is fixed, not adaptive.

    `decode_fn`, if given, maps a raw numeric candidate to a decoded
    parameter dict (see ParameterSpace.decode). When provided, duplicate
    detection operates on the DECODED configuration, not the raw numeric
    vector -- so two numerically different proposals that decode to the
    same configuration (e.g. two continuous values snapping to the same
    discrete choice) are treated as one real candidate, and each batch
    entry carries an extra "decoded_params" key. When `decode_fn` is None
    (the default), `evaluated_configs`/`total_configurations` are ignored
    entirely and behavior/batch-entry shape are identical to before these
    parameters existed -- every existing continuous benchmark is
    unaffected.

    `evaluated_configs`, if given, is a set of canonical decoded-config
    keys (`tuple(sorted(decoded.items()))`) already evaluated in prior
    rounds/warm-start. These are treated as already "used" from the
    start, so a candidate decoding to one of them is skipped just like an
    in-batch duplicate, and the caller's ranked list must be long enough
    to find something unused -- this is a proposal-list-length concern
    for the caller (e.g. Orchestrator over-requesting from `propose()`),
    not something this function can fix by inventing candidates.

    If a source's ranked list is exhausted before its allocation is
    filled: when `total_configurations` is given and the number of
    distinct configurations already used has reached it, every possible
    configuration has genuinely been evaluated, and this raises
    SearchSpaceExhaustedError (a ValueError subclass) instead of a
    generic exhaustion error -- naming the real, provable cause rather
    than leaving it to look like an ordinary collision. Otherwise, raises
    a plain ValueError (the proposal list just wasn't long/lucky enough,
    even though unevaluated configurations likely still exist elsewhere)
    rather than silently returning a smaller batch.
    """
    if set(proposals.keys()) != set(allocation.keys()):
        raise ValueError(
            "proposals and allocation must cover the same set of sources: "
            f"proposals={sorted(proposals)}, allocation={sorted(allocation)}"
        )

    batch: list[dict] = []
    if decode_fn is not None and evaluated_configs:
        used: set[tuple] = set(evaluated_configs)
    else:
        used = set()

    for source, slots_needed in allocation.items():
        candidates = proposals[source]
        placed = 0
        rank = 0
        while placed < slots_needed and rank < len(candidates):
            candidate = candidates[rank]
            decoded = decode_fn(candidate) if decode_fn is not None else None
            if decoded is not None:
                key = tuple(sorted(decoded.items()))
            else:
                key = tuple(np.asarray(candidate, dtype=float).tolist())
            if key not in used:
                used.add(key)
                entry = {
                    "candidate": candidate,
                    "surrogate_source": source,
                    "candidate_rank_within_source": rank,
                }
                if decoded is not None:
                    entry["decoded_params"] = decoded
                batch.append(entry)
                placed += 1
            rank += 1

        if placed < slots_needed:
            if (
                decode_fn is not None
                and total_configurations is not None
                and len(used) >= total_configurations
            ):
                raise SearchSpaceExhaustedError(
                    f"the entire finite decoded search space "
                    f"({total_configurations} configurations) has been "
                    f"evaluated; no unevaluated configuration remains for "
                    f"source '{source}' (needed {slots_needed}, placed {placed})"
                )
            raise ValueError(
                f"not enough unique candidates for source '{source}': "
                f"needed {slots_needed}, placed {placed} "
                f"(exhausted {len(candidates)}-candidate proposal list)"
            )

    return batch
