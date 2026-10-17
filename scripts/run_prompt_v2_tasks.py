"""Run/resume HiddenBench Prompt V2 on selected task IDs with GPT-4o-mini."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.models import OpenAICompatibleClient
from hiddenbench.prompts import load_prompts
from hiddenbench.simulator import Profile, run_scenario

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = {
    "v1": ROOT / "runs/prompt_v1_gpt4o_mini_hidden_tasks_1_10",
    "v2": ROOT / "runs/prompt_v2_gpt4o_mini_hidden_tasks_1_10",
}
PREVIOUS_TASK1 = ROOT / "runs/prompt_v2_gpt4o_mini_task1_hidden_3round.json"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if len(set(args.task_ids)) != len(args.task_ids):
        parser.error("task IDs must be unique")

    load_dotenv(args.dotenv)
    model = OpenAICompatibleClient(
        "gpt-4o-mini", provider="openai", temperature=args.temperature,
        max_retries=3, timeout=90,
    )
    prompts = load_prompts(version=args.prompt_version)
    output_dir = args.output_dir or DEFAULT_OUTPUTS[args.prompt_version]
    benchmark = load_benchmark(path=str(ROOT / "data/benchmark.json"))
    config = {
        "model": model.model_name,
        "profile": "hidden",
        "rounds": args.rounds,
        "seed": args.seed,
        "temperature": args.temperature,
        "prompt_version": args.prompt_version,
        "prompt_file": str(prompts.prompt_file),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict] = {}

    for task_id in args.task_ids:
        task = find_task(benchmark, task_id=str(task_id))
        path = output_dir / f"task_{task_id}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if any(saved.get("metadata", {}).get(k) != v for k, v in config.items()):
                raise ValueError(f"Existing result has different config: {path}")
            run = saved["runs"][0]
            status[str(task_id)] = {"status": "reused", "path": str(path), "run": run}
            print(f"Task {task_id}: cached result reused", flush=True)
            continue

        if args.prompt_version == "v2" and task_id == 1 and PREVIOUS_TASK1.exists():
            previous = json.loads(PREVIOUS_TASK1.read_text(encoding="utf-8"))
            if (previous.get("metadata", {}).get("model") == model.model_name
                    and previous.get("metadata", {}).get("profile") == "hidden"
                    and previous.get("metadata", {}).get("rounds") == args.rounds
                    and previous.get("metadata", {}).get("seed") == args.seed
                    and previous.get("metadata", {}).get("temperature") == args.temperature
                    and previous.get("metadata", {}).get("prompt_version") == "v2"
                    and previous.get("runs", [{}])[0].get("task_id") == task.id):
                atomic_json(path, previous)
                run = previous["runs"][0]
                status[str(task_id)] = {"status": "reused", "path": str(path), "run": run}
                print(f"Task {task_id}: prior matching result reused", flush=True)
                continue

        expected_calls = len(task.hidden_information) * (args.rounds + 2)
        print(f"Task {task_id}/{args.task_ids[-1]} started ({task.name}; {expected_calls} expected calls)", flush=True)
        try:
            run = run_scenario(
                task, model, Profile.HIDDEN, prompts, rounds=args.rounds, seed=args.seed,
            )
            run["model_call_diagnostics"] = list(model.json_call_traces)
            result = {
                "metadata": {**config, "task_ids": [task.id],
                             "expected_generation_calls": expected_calls,
                             "created_at": datetime.now(timezone.utc).isoformat()},
                "runs": [run],
            }
            atomic_json(path, result)
            status[str(task_id)] = {"status": "complete", "path": str(path), "run": run}
            print(
                f"Task {task_id} complete: initial correct "
                f"{sum(v['vote'] == task.correct_answer for v in run['initial_votes'])}/"
                f"{len(run['initial_votes'])}; final majority={run['majority_vote']}; "
                f"correct={run['majority_vote'] == task.correct_answer}; saved={path}",
                flush=True,
            )
        except Exception as exc:
            failure = {"task_id": task.id, "config": config,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "model_call_diagnostics": list(model.json_call_traces),
                       "created_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(output_dir / f"task_{task_id}.error.json", failure)
            status[str(task_id)] = {"status": "failed", "path": str(output_dir / f"task_{task_id}.error.json"),
                                    "error": str(exc)}
            print(f"Task {task_id} failed: {type(exc).__name__}: {exc}", flush=True)

    summary_rows = []
    for task_id in args.task_ids:
        item = status[str(task_id)]
        if item["status"] == "failed":
            summary_rows.append({"task_id": task_id, "status": "failed", "error": item["error"]})
            continue
        run = item["run"]
        correct = run["correct_answer"]
        final_votes = run["final_votes"]
        counts = Counter(v["vote"] for v in final_votes)
        summary_rows.append({
            "task_id": task_id, "status": item["status"], "correct_answer": correct,
            "initial_votes": [v["vote"] for v in run["initial_votes"]],
            "final_votes": [v["vote"] for v in final_votes],
            "final_vote_counts": dict(counts), "majority_vote": run["majority_vote"],
            "majority_correct": run["majority_vote"] == correct,
            "correct_final_votes": sum(v["vote"] == correct for v in final_votes),
            "agents": len(final_votes), "rounds": len(run["rounds"]), "path": item["path"],
        })
    summary = {
        "metadata": {**config, "requested_task_ids": args.task_ids,
                     "created_at": datetime.now(timezone.utc).isoformat()},
        "tasks": summary_rows,
        "completed": sum(row["status"] != "failed" for row in summary_rows),
        "failed": sum(row["status"] == "failed" for row in summary_rows),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
