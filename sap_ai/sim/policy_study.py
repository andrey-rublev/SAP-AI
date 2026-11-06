"""Paired offline policy comparisons on shared seeds, with uncertainty estimates.

The ability-free simulator is an approximation. Its measurements do not assert
live performance, OCR accuracy, or ability-aware combat quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from sap_ai.desktop.state import DesktopPolicy
from sap_ai.desktop.strategy import STRATEGIES, make_policy
from sap_ai.sim.arena import play_run


def _factory(name):
    if name == "recorded":
        return DesktopPolicy.recorded_baseline
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name}")
    return lambda: make_policy(name)


def evaluate_pair(baseline_factory, candidate_factory, opponent_factory, *, runs, seed, include_episodes=False):
    """Fresh policies per episode, equal seeds, paired trophy deltas.

    ci95 is a normal approximation for the paired mean, not a guarantee. With
    one observation no interval is returned. Fresh factory instances prevent
    state carried between runs from changing later comparisons.
    """
    if type(runs) is not int or runs < 1 or type(seed) is not int or seed < 0:
        raise ValueError("runs must be positive and seed must be nonnegative integers")
    episodes = []
    for episode_seed in range(seed, seed + runs):
        baseline = play_run(baseline_factory(), opponent_factory(), episode_seed)
        candidate = play_run(candidate_factory(), opponent_factory(), episode_seed)
        episodes.append({"seed": episode_seed, "baseline": baseline, "candidate": candidate,
                         "trophy_delta": candidate["trophies"] - baseline["trophies"]})
    deltas = [row["trophy_delta"] for row in episodes]
    average = mean(deltas)
    radius = 1.96 * stdev(deltas) / sqrt(runs) if runs > 1 else None
    result = {"runs": runs, "seed_start": seed, "seed_end_exclusive": seed + runs,
              "baseline_mean_trophies": mean(row["baseline"]["trophies"] for row in episodes),
              "candidate_mean_trophies": mean(row["candidate"]["trophies"] for row in episodes),
              "baseline_victories": sum(row["baseline"]["victory"] for row in episodes),
              "candidate_victories": sum(row["candidate"]["victory"] for row in episodes),
              "mean_trophy_delta": average,
              "paired_mean_ci95": [average - radius, average + radius] if radius is not None else None,
              "better": sum(delta > 0 for delta in deltas), "equal": sum(delta == 0 for delta in deltas),
              "worse": sum(delta < 0 for delta in deltas)}
    if include_episodes:
        result["episodes"] = episodes
    return result


def study(*, baseline="conservative", candidate="economy", opponents=("recorded", "conservative"),
          runs=300, seed=10000, include_episodes=False):
    if not opponents or len(set(opponents)) != len(opponents):
        raise ValueError("opponents must be a nonempty sequence of distinct names")
    results = {opponent: evaluate_pair(_factory(baseline), _factory(candidate), _factory(opponent),
                                     runs=runs, seed=seed, include_episodes=include_episodes)
               for opponent in opponents}
    from sap_ai.desktop import state, strategy
    from sap_ai.sim import arena
    files = (Path(__file__), Path(state.__file__), Path(strategy.__file__), Path(arena.__file__))
    return {"schema_version": 1, "offline": True,
            "configuration": {"baseline": baseline, "candidate": candidate, "opponents": list(opponents),
                              "runs_per_opponent": runs, "seed_start": seed, "seed_end_exclusive": seed + runs},
            "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2]).as_posix()):
                              hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "results": results,
            "limitations": "Ability-free approximation, not live win rates. CI is a paired normal approximation. "
                            "Do not tune a candidate on the seeds used for its final evaluation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=STRATEGIES, default="conservative")
    parser.add_argument("--candidate", choices=STRATEGIES, default="economy")
    parser.add_argument("--opponents", nargs="+", choices=("recorded", *STRATEGIES), default=["recorded", "conservative"])
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--include-episodes", action="store_true")
    parser.add_argument("--output", type=Path, help="new JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise ValueError("output must be a new file")
        report = study(baseline=args.baseline, candidate=args.candidate, opponents=args.opponents,
                       runs=args.runs, seed=args.seed, include_episodes=args.include_episodes)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as target:
                target.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name, result in report["results"].items():
            interval = result["paired_mean_ci95"]
            ci = "not estimated" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
            print(f"vs {name}: {result['baseline_mean_trophies']:.3f} -> {result['candidate_mean_trophies']:.3f} "
                  f"trophies; paired delta {result['mean_trophy_delta']:+.3f}, 95% CI {ci}; "
                  f"victories {result['baseline_victories']} -> {result['candidate_victories']}")
        return report
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
