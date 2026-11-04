#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never requires plt.show(), always saves to disk
import matplotlib.pyplot as plt
import numpy as np

from run_data import load_candidates, load_metadata, load_round_metrics


def plot_best_objective_vs_evaluations(candidates: list[dict], metadata: dict, out_path: Path) -> None:
    if not candidates:
        raise ValueError("no candidates to plot")

    y = np.array([c["y"] for c in candidates])
    best_so_far = np.maximum.accumulate(y)
    evaluations = np.arange(1, len(y) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(evaluations, best_so_far, marker="o", markersize=3)
    ax.set_xlabel("Cumulative objective evaluations")
    ax.set_ylabel("Best objective so far (higher is better)")
    ax.set_title(f"{metadata['run_name']}\nBest objective vs. evaluations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_objective_vs_wall_time(candidates: list[dict], metadata: dict, out_path: Path) -> None:
    if not candidates:
        raise ValueError("no candidates to plot")

    # Sort by actual completion time, not batch/file order: with
    # n_workers > 1, candidates can finish out of logical batch order.
    ordered = sorted(candidates, key=lambda c: c["eval_end_time"])
    t0 = min(c["eval_start_time"] for c in candidates)
    elapsed = np.array([c["eval_end_time"] - t0 for c in ordered])
    y = np.array([c["y"] for c in ordered])
    best_so_far = np.maximum.accumulate(y)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elapsed, best_so_far, marker="o", markersize=3)
    ax.set_xlabel("Elapsed wall-clock time (s)")
    ax.set_ylabel("Best objective so far (higher is better)")
    # Built from absolute eval_end_time timestamps on a continuously
    # running clock, not summed eval_duration -- so the gaps between
    # batches (surrogate fit/propose/merge time) are included, not just
    # time spent inside the objective call.
    ax.set_title(
        f"{metadata['run_name']}\nBest objective vs. elapsed wall-clock time\n"
        "(includes surrogate fit/propose/merge overhead between batches)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timing_breakdown_by_round(round_metrics: dict, metadata: dict, out_path: Path) -> None:
    name_a = metadata["surrogate_a_name"]
    name_b = metadata["surrogate_b_name"]

    fit_a = f"{name_a} fit"
    proposal_a = f"{name_a} proposal"
    fit_b = f"{name_b} fit"
    proposal_b = f"{name_b} proposal"

    # Only optimizer/surrogate overhead components.
    components = [fit_a, proposal_a, fit_b, proposal_b]

    bars = []

    # Skip warm start because no surrogate fitting/proposal happens there.
    for r in round_metrics["rounds"]:
        bars.append(
            {
                "round_id": r["round_id"],
                fit_a: r["surrogate_a_fit_time"],
                proposal_a: r["surrogate_a_proposal_time"],
                fit_b: r["surrogate_b_fit_time"],
                proposal_b: r["surrogate_b_proposal_time"],
            }
        )

    if not bars:
        raise ValueError("no round timing data to plot")

    round_ids = [b["round_id"] for b in bars]

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(bars))

    for component in components:
        values = np.array([b[component] for b in bars])
        ax.bar(round_ids, values, bottom=bottom, label=component)
        bottom += values

    ax.set_xlabel("Round")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(
        f"{metadata['run_name']}\n"
        "Surrogate overhead breakdown by round"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_surrogate_timing_by_round(round_metrics: dict, metadata: dict, out_path: Path) -> None:
    rounds = round_metrics["rounds"]
    if not rounds:
        raise ValueError("no round data to plot surrogate timing (n_rounds=0)")

    name_a = metadata["surrogate_a_name"]
    name_b = metadata["surrogate_b_name"]
    round_ids = [r["round_id"] for r in rounds]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(round_ids, [r["surrogate_a_fit_time"] for r in rounds], marker="o", label=f"{name_a} fit")
    ax.plot(
        round_ids, [r["surrogate_a_proposal_time"] for r in rounds], marker="o", label=f"{name_a} proposal"
    )
    ax.plot(round_ids, [r["surrogate_b_fit_time"] for r in rounds], marker="s", label=f"{name_b} fit")
    ax.plot(
        round_ids, [r["surrogate_b_proposal_time"] for r in rounds], marker="s", label=f"{name_b} proposal"
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"{metadata['run_name']}\nSurrogate fit/proposal time by round")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_surrogate_contribution(candidates: list[dict], metadata: dict, out_path: Path) -> None:
    if not candidates:
        raise ValueError("no candidates to plot")

    sources = sorted({c["surrogate_source"] for c in candidates})
    totals = []
    improved = []
    for source in sources:
        rows = [c for c in candidates if c["surrogate_source"] == source]
        totals.append(len(rows))
        improved.append(sum(1 for r in rows if r["improved_global_best"]))

    x = np.arange(len(sources))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, totals, width, label="evaluated")
    ax.bar(x + width / 2, improved, width, label="improved global best")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right")
    ax.set_ylabel("Number of candidates")
    ax.set_title(f"{metadata['run_name']}\nSurrogate contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_all_plots(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    candidates = load_candidates(run_dir)
    round_metrics = load_round_metrics(run_dir)
    metadata = load_metadata(run_dir)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("best_objective_vs_evaluations.png", plot_best_objective_vs_evaluations, (candidates, metadata)),
        ("best_objective_vs_wall_time.png", plot_best_objective_vs_wall_time, (candidates, metadata)),
        ("timing_breakdown_by_round.png", plot_timing_breakdown_by_round, (round_metrics, metadata)),
        ("surrogate_timing_by_round.png", plot_surrogate_timing_by_round, (round_metrics, metadata)),
        ("surrogate_contribution.png", plot_surrogate_contribution, (candidates, metadata)),
    ]

    outputs = []
    for filename, plot_fn, data_args in plot_specs:
        path = plots_dir / filename
        plot_fn(*data_args, path)
        outputs.append(path)

    return outputs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for one experiment run (post-processing only)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    outputs = generate_all_plots(args.run_dir)
    print(f"wrote {len(outputs)} plots to {outputs[0].parent}")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
