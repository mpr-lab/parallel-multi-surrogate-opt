import numpy as np


def merge_candidates(
    proposals: dict[str, np.ndarray],
    allocation: dict[str, int],
) -> list[dict]:
    """Combine ranked per-surrogate proposals into one fixed-size batch.

    Sources are processed in the order `allocation` is given. Each source
    fills its own allocation only from its own ranked proposal list,
    skipping any candidate already placed in the batch (by itself or by an
    earlier-processed source) and advancing to its next-ranked candidate
    instead. A slot is never filled from a source other than the one it
    was allocated to -- allocation is fixed, not adaptive.

    Raises ValueError if a source's proposal list is exhausted before its
    allocation is filled, rather than silently returning a smaller batch.
    """
    if set(proposals.keys()) != set(allocation.keys()):
        raise ValueError(
            "proposals and allocation must cover the same set of sources: "
            f"proposals={sorted(proposals)}, allocation={sorted(allocation)}"
        )

    batch: list[dict] = []
    used: set[tuple] = set()

    for source, slots_needed in allocation.items():
        candidates = proposals[source]
        placed = 0
        rank = 0
        while placed < slots_needed and rank < len(candidates):
            candidate = candidates[rank]
            key = tuple(np.asarray(candidate, dtype=float).tolist())
            if key not in used:
                used.add(key)
                batch.append(
                    {
                        "candidate": candidate,
                        "surrogate_source": source,
                        "candidate_rank_within_source": rank,
                    }
                )
                placed += 1
            rank += 1

        if placed < slots_needed:
            raise ValueError(
                f"not enough unique candidates for source '{source}': "
                f"needed {slots_needed}, placed {placed} "
                f"(exhausted {len(candidates)}-candidate proposal list)"
            )

    return batch
