from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from hiddenbench.benchmark import BenchmarkTask


def load_results(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values) / (len(values) ** 0.5)


def _task_lookup(benchmark: list[BenchmarkTask]) -> dict[str, BenchmarkTask]:
    lookup: dict[str, BenchmarkTask] = {}
    for task in benchmark:
        lookup[str(task.id)] = task
        lookup[task.name] = task
    return lookup


def _resolve_task(run: dict[str, Any], lookup: dict[str, BenchmarkTask]) -> BenchmarkTask:
    if "task_id" in run and str(run["task_id"]) in lookup:
        return lookup[str(run["task_id"])]
    if "scenario" in run and run["scenario"] in lookup:
        return lookup[run["scenario"]]
    raise KeyError(f"Could not match run to benchmark task: {run.get('scenario')}")


def iter_result_runs(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Return runs from either the new or legacy result schema."""

    if isinstance(results.get("runs"), list):
        return list(results["runs"])

    runs: list[dict[str, Any]] = []
    for condition in results.get("conditions", []):
        if not isinstance(condition, dict):
            continue
        condition_metadata = {
            "condition_id": condition.get("condition_id"),
            "model": condition.get("model"),
            "percentage_special": condition.get("percentage_special"),
            "discussion_mode": condition.get("discussion_mode"),
        }
        for run in condition.get("runs", []):
            if isinstance(run, dict):
                enriched = dict(run)
                enriched.setdefault("_condition", condition_metadata)
                runs.append(enriched)
    return runs


def result_model_name(results: dict[str, Any]) -> str | None:
    if results.get("metadata", {}).get("model"):
        return results["metadata"]["model"]
    models = [
        condition.get("model")
        for condition in results.get("conditions", [])
        if isinstance(condition, dict) and condition.get("model")
    ]
    unique = sorted(set(models))
    if len(unique) == 1:
        return unique[0]
    if unique:
        return ",".join(unique)
    return None


def result_profile_name(results: dict[str, Any]) -> str | None:
    if results.get("metadata", {}).get("profile"):
        return results["metadata"]["profile"]
    profiles: set[str] = set()
    for run in iter_result_runs(results)[:10]:
        if run.get("profile"):
            profiles.add(str(run["profile"]))
        elif run.get("_condition", {}).get("percentage_special") is not None:
            profiles.add("legacy")
    if len(profiles) == 1:
        return next(iter(profiles))
    if profiles:
        return ",".join(sorted(profiles))
    return None


def _individual_accuracy(votes: list[dict[str, Any]], correct_answer: str) -> float:
    if not votes:
        return 0.0
    return sum(1 for vote in votes if vote.get("vote") == correct_answer) / len(votes)


def _majority_accuracy(votes: list[dict[str, Any]], correct_answer: str) -> float:
    if not votes:
        return 0.0
    return 1.0 if sum(1 for vote in votes if vote.get("vote") == correct_answer) > len(votes) / 2 else 0.0


def score_results(
    results: dict[str, Any],
    benchmark: list[BenchmarkTask],
    *,
    full_profile_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute HiddenBench average-rule and majority-rule metrics."""

    lookup = _task_lookup(benchmark)
    run_scores: list[dict[str, Any]] = []
    by_task_values: dict[str, dict[str, list[float]]] = {}

    for run in iter_result_runs(results):
        task = _resolve_task(run, lookup)
        pre_average = _individual_accuracy(run.get("initial_votes", []), task.correct_answer)
        post_average = _individual_accuracy(run.get("final_votes", []), task.correct_answer)
        pre_majority = _majority_accuracy(run.get("initial_votes", []), task.correct_answer)
        post_majority = _majority_accuracy(run.get("final_votes", []), task.correct_answer)
        row = {
            "task_id": task.id,
            "scenario": task.name,
            "pre_average_accuracy": pre_average,
            "post_average_accuracy": post_average,
            "pre_majority_accuracy": pre_majority,
            "post_majority_accuracy": post_majority,
        }
        run_scores.append(row)
        bucket = by_task_values.setdefault(
            task.name,
            {"pre_average": [], "post_average": [], "pre_majority": [], "post_majority": []},
        )
        bucket["pre_average"].append(pre_average)
        bucket["post_average"].append(post_average)
        bucket["pre_majority"].append(pre_majority)
        bucket["post_majority"].append(post_majority)

    pre_average_values = [row["pre_average_accuracy"] for row in run_scores]
    post_average_values = [row["post_average_accuracy"] for row in run_scores]
    pre_majority_values = [row["pre_majority_accuracy"] for row in run_scores]
    post_majority_values = [row["post_majority_accuracy"] for row in run_scores]

    summary = {
        "profile": result_profile_name(results),
        "model": result_model_name(results),
        "num_runs": len(run_scores),
        "num_tasks": len({row["scenario"] for row in run_scores}),
        "average_accuracy": {
            "pre": mean(pre_average_values) if pre_average_values else 0.0,
            "post": mean(post_average_values) if post_average_values else 0.0,
            "pre_sem": _sem(pre_average_values),
            "post_sem": _sem(post_average_values),
        },
        "majority_accuracy": {
            "pre": mean(pre_majority_values) if pre_majority_values else 0.0,
            "post": mean(post_majority_values) if post_majority_values else 0.0,
            "pre_sem": _sem(pre_majority_values),
            "post_sem": _sem(post_majority_values),
        },
        "by_task": [],
    }
    summary["average_accuracy"]["improvement"] = (
        summary["average_accuracy"]["post"] - summary["average_accuracy"]["pre"]
    )
    summary["majority_accuracy"]["improvement"] = (
        summary["majority_accuracy"]["post"] - summary["majority_accuracy"]["pre"]
    )

    for task_name, values in sorted(by_task_values.items()):
        summary["by_task"].append(
            {
                "scenario": task_name,
                "num_runs": len(values["pre_average"]),
                "pre_average_accuracy": mean(values["pre_average"]),
                "post_average_accuracy": mean(values["post_average"]),
                "pre_majority_accuracy": mean(values["pre_majority"]),
                "post_majority_accuracy": mean(values["post_majority"]),
            }
        )

    if full_profile_results is not None:
        full_scores = score_results(full_profile_results, benchmark)
        full_pre = full_scores["average_accuracy"]["pre"]
        summary["full_profile_pre_average_accuracy"] = full_pre
        summary["gap_to_full_profile_pre"] = summary["average_accuracy"]["post"] - full_pre

    return summary


def format_score_summary(scores: dict[str, Any]) -> str:
    avg = scores["average_accuracy"]
    maj = scores["majority_accuracy"]
    lines = [
        f"Model: {scores.get('model') or 'unknown'}",
        f"Profile: {scores.get('profile') or 'unknown'}",
        f"Runs: {scores['num_runs']} across {scores['num_tasks']} tasks",
        f"Average rule: {avg['pre']:.3f} -> {avg['post']:.3f} ({avg['improvement']:+.3f})",
        f"Majority rule: {maj['pre']:.3f} -> {maj['post']:.3f} ({maj['improvement']:+.3f})",
    ]
    if "gap_to_full_profile_pre" in scores:
        lines.append(f"Gap to full-profile pre-discussion average: {scores['gap_to_full_profile_pre']:+.3f}")
    return "\n".join(lines)
