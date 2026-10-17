"""Run V2 leave-one-component-out ablations without modifying V2 itself.

Smoke mode runs a fixed engineering subset for every variant. Full mode resumes
the same task JSON files, then writes per-variant and paired V2 reports.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.models import OpenAICompatibleClient
from hiddenbench.prompts import load_prompts
import hiddenbench.simulator as simulator
from hiddenbench.simulator import Profile, run_scenario

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015/summary.json"
OUT_ROOT = ROOT / "runs/v2_leave_one_component_out_gpt4o_mini_hidden_20261016"
VARIANTS = (
    "V2_MINUS_AWARENESS",
    "V2_MINUS_EVIDENCE_DISCIPLINE",
    "V2_MINUS_DISCLOSURE_REVISION",
)
BASE_PROMPT_PATH = ROOT / "src/hiddenbench/prompts_v2.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_raw_prompts() -> dict[str, Any]:
    return json.loads(BASE_PROMPT_PATH.read_text(encoding="utf-8"))


def joined(value: str | list[str] | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return "\n\n".join(value)


def variant_prompt_data(variant: str) -> dict[str, Any]:
    data = load_raw_prompts()
    # Keep a clean copy of the source text for the diff artifact.
    if variant == "V2_MINUS_AWARENESS":
        data["system_prompt"] = [
            "[Situation]\n%description%",
            "[Task Structure]\nThere are %num_agents% participants.\nThe participants will discuss sequentially and decide among the candidate answers.",
            "[Available Facts]\n%shared_information%%private_information%",
            "[Reasoning Guidance]\n- Your initial vote is a provisional decision based on the information currently available and can be revised after discussion.\n- Evaluate all candidate options using the evidence currently available.\n- Distinguish hard constraints that can eliminate an option from soft benefits that only make an option attractive.\n- Hard feasibility constraints should take priority over soft benefits.\n- Do not assume facts that have not been explicitly provided.\n- Treat other participants' statements as evidence, not as authority.\n- Revise your judgment when new evidence warrants it.",
            "[Communication Goal]\nDuring discussion, prioritize decision-relevant information.\nEspecially communicate evidence that rules out an option, rescues an option, or materially changes the decision.\nAvoid merely repeating an existing consensus.",
            "Keep the public response concise, one or two sentences.\nDo not expose hidden chain-of-thought. State only the relevant evidence and its decision implication.",
            "%extra%",
        ]
        data["first_vote_prompt"] = [
            "This is a provisional pre-discussion decision based on the information currently available.\nBefore choosing, silently compare every candidate using only the information currently available.\nPrioritize explicit hard constraints over soft benefits and do not infer unstated facts.\nYour rationale should briefly explain why the selected option is currently preferable and why the main alternatives are less supported or remain uncertain.",
            data["first_vote_prompt"][1],
        ]
        data["first_user_prompt"] = [
            "Your initial vote was provisional.\nDo not defend it merely because you selected it earlier.\nYou are the first to speak.\nShare the most decision-relevant information you currently have.\nExplain briefly which candidate or candidates this evidence affects.",
            data["first_user_prompt"][1],
        ]
        user = joined(data["user_prompt"])
        user = user.replace("1. your own shared and private information, and", "1. your own information, and")
        user = user.replace("- newly revealed private information,", "- newly revealed information,")
        data["user_prompt"] = user
    elif variant == "V2_MINUS_EVIDENCE_DISCIPLINE":
        system = joined(data["system_prompt"])
        system = system.replace("- Evaluate all candidate options using the evidence currently available.\n", "")
        system = system.replace("- Distinguish hard constraints that can eliminate an option from soft benefits that only make an option attractive.\n", "")
        system = system.replace("- Hard feasibility constraints should take priority over soft benefits.\n", "")
        system = system.replace("- Do not assume facts that have not been explicitly provided.\n", "")
        system = system.replace("- Treat other participants' statements as evidence, not as authority.\n", "")
        data["system_prompt"] = system
        first_vote = joined(data["first_vote_prompt"])
        first_vote = first_vote.replace("Before choosing, silently compare every candidate using only the evidence currently available.\n", "Before choosing, silently compare every candidate.\n")
        first_vote = first_vote.replace("Prioritize explicit hard constraints over soft benefits and do not infer unstated facts.\n", "")
        data["first_vote_prompt"] = first_vote
        user = joined(data["user_prompt"])
        user = user.replace("- hard constraints,\n", "")
        user = user.replace("- contradictions,\n", "")
        user = user.replace("- evidence that changes the feasibility of an option.\n", "")
        user = user.replace("Do not follow or repeat a majority opinion unless the evidence supports it.\n", "")
        data["user_prompt"] = user
        final_vote = joined(data["vote_prompt"])
        final_vote = final_vote.replace("Prioritize explicit feasibility constraints over soft benefits.\n", "")
        final_vote = final_vote.replace("Use concrete evidence rather than repeated opinions.\n", "")
        data["vote_prompt"] = final_vote
    elif variant == "V2_MINUS_DISCLOSURE_REVISION":
        system = joined(data["system_prompt"])
        system = system.replace("- Your initial vote is a provisional decision based on incomplete information and can be revised after discussion.\n", "")
        system = system.replace("- Revise your judgment when new evidence warrants it.\n", "")
        system = re.sub(r"\n\[Communication Goal\]\n.*?(?=\n\nKeep the public response concise)", "", system, flags=re.S)
        data["system_prompt"] = system
        data["first_vote_prompt"] = joined(data["first_vote_prompt"]).replace(
            "This is a provisional pre-discussion decision based on incomplete information.\n", ""
        )
        data["first_user_prompt"] = [
            "You are the first to speak.",
            "Keep the public response concise, one or two sentences.",
        ]
        user = joined(data["user_prompt"])
        user = user.replace(
            "Before responding, silently reconsider every candidate using:\n1. your own shared and private information, and\n2. the evidence communicated by the other participants.\n", ""
        )
        user = user.replace("- newly revealed private information,\n", "")
        user = user.replace("Contribute the most useful non-redundant evidence you can.\n", "")
        user = user.replace("If new evidence changes your previous judgment, update it.\n", "")
        data["user_prompt"] = user
        data["vote_prompt"] = [
            "Previous messages from other people:\n%group_discussion%",
            "Based on the discussion, select the best answer.",
            "Please decide and provide your rationale in the following JSON format:\n{\n    \"vote\": <A string, %possible_answers%>,\n    \"rationale\": <A string, representing your rationale>\n}",
        ]
    else:
        raise ValueError(f"Unknown variant {variant}")
    return data


def prepare_variant(variant: str, out_dir: Path) -> tuple[Path, str, Any]:
    original = load_raw_prompts()
    changed = variant_prompt_data(variant)
    prompt_path = out_dir / "prompt_variant.json"
    write_json(prompt_path, changed)
    old_text = json.dumps(original, ensure_ascii=False, indent=2).splitlines()
    new_text = json.dumps(changed, ensure_ascii=False, indent=2).splitlines()
    diff = difflib.unified_diff(old_text, new_text, fromfile="prompts_v2.json", tofile=f"{variant}.json", lineterm="")
    (out_dir / "prompt_diff.md").write_text("```diff\n" + "\n".join(diff) + "\n```\n", encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    prompts = load_prompts(path=prompt_path)
    prompts = prompts.__class__(**{**prompts.__dict__, "prompt_version": variant, "prompt_file": str(prompt_path.resolve())})
    return prompt_path, prompt_hash, prompts


class TracingClient(OpenAICompatibleClient):
    """Same request parameters/retry schedule as base class, with usage records."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completions: list[dict[str, Any]] = []

    def _completion(self, messages, response_format=None):
        last_error = None
        for attempt in range(self.max_retries):
            record: dict[str, Any] = {"attempt": attempt + 1, "model": self.model_name}
            try:
                kwargs = {"model": self.model_name, "messages": messages}
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                record["usage"] = None
                usage = getattr(response, "usage", None)
                if usage is not None:
                    record["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                record["status"] = "ok"
                self.completions.append(record)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                record.update(status="error", error_type=type(exc).__name__, error=str(exc))
                self.completions.append(record)
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


class CallRecorder:
    def __init__(self, client: TracingClient, path: Path):
        self.client, self.model_name, self.path = client, client.model_name, path
        self.calls: list[dict[str, Any]] = []

    def _call(self, method, messages, schema=None):
        before = len(self.client.completions)
        item = {"call": len(self.calls) + 1, "method": method, "model": self.model_name, "messages": messages}
        if schema is not None:
            item["schema"] = schema
        try:
            result = self.client.chat_json(messages, schema) if method == "chat_json" else self.client.chat(messages)
            item["response"] = result
            item["completion_attempts"] = self.client.completions[before:]
            self.calls.append(item)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return result
        except Exception as exc:
            item["error_type"] = type(exc).__name__
            item["error"] = str(exc)
            item["completion_attempts"] = self.client.completions[before:]
            self.calls.append(item)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            raise

    def chat(self, messages):
        return self._call("chat", messages)

    def chat_json(self, messages, schema):
        return self._call("chat_json", messages, schema)


def patch_awareness_fact_labels():
    original = simulator.instantiate_agents

    def wrapped(*args, **kwargs):
        agents, assignments = original(*args, **kwargs)
        for agent in agents:
            text = agent.history[0]["content"]
            # The neutral block contains the same shared facts (same order) plus
            # the agent's one assigned fact; only duplicated section numbering
            # is normalized. Fact contents are otherwise byte-for-byte intact.
            in_available = False
            index = 0
            lines = []
            for line in text.splitlines(keepends=True):
                if line.strip() == "[Available Facts]":
                    in_available = True
                elif line.startswith("[") and line.rstrip("\n").endswith("]"):
                    in_available = False
                if in_available and re.match(r"Fact \d+: ", line):
                    index += 1
                    line = re.sub(r"^Fact \d+: ", f"Fact {index}: ", line, count=1)
                lines.append(line)
            agent.history[0]["content"] = "".join(lines)
        return agents, assignments

    simulator.instantiate_agents = wrapped
    return original


def baseline_index() -> dict[str, dict[str, Any]]:
    if not BASELINE.exists():
        raise FileNotFoundError(f"Required V2 full benchmark summary not found: {BASELINE}")
    summary = json.loads(BASELINE.read_text(encoding="utf-8"))
    index = {}
    for row in summary.get("tasks", []):
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        saved = json.loads(path.read_text(encoding="utf-8"))
        index[str(row["task_id"])] = saved["runs"][0]
    return index


def validate_against_baseline(run: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    actual_assignments = run["fact_assignments"]
    base_assignments = baseline["fact_assignments"]
    assignment_equal = actual_assignments == base_assignments
    actual_speakers = [[m["agent"] for m in rd["messages"]] for rd in run["rounds"]]
    base_speakers = [[m["agent"] for m in rd["messages"]] for rd in baseline["rounds"]]
    return {
        "same_hidden_assignment_and_fact_order": assignment_equal,
        "same_speaker_order_each_round": actual_speakers == base_speakers,
        "baseline_task_id": baseline["task_id"],
    }


def task_report(run: dict[str, Any], variant: str, prompt_hash: str, expected_calls: int,
                recorder: CallRecorder, client: TracingClient, baseline: dict[str, Any]) -> dict[str, Any]:
    gold = run["correct_answer"]
    init_votes = [v["vote"] for v in run["initial_votes"]]
    final_votes = [v["vote"] for v in run["final_votes"]]
    initial_count = Counter(init_votes)
    final_count = Counter(final_votes)
    initial_majority = initial_count.most_common(1)[0][0]
    final_majority = run["majority_vote"]
    total_usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    usage_available = True
    for record in client.completions:
        usage = record.get("usage")
        if usage is None:
            usage_available = False
            continue
        for key in total_usage:
            if usage.get(key) is not None:
                total_usage[key] += usage[key]
    baseline_ok = validate_against_baseline(run, baseline)
    return {
        "task_id": run["task_id"], "scenario": run["scenario"], "status": "complete",
        "gold_answer": gold, "variant": variant, "model": run["model"], "seed": run["seed"],
        "temperature": 0.0, "prompt_version": variant, "prompt_sha256": prompt_hash,
        "initial_individual_votes": init_votes, "initial_vote_rationales": [v["rationale"] for v in run["initial_votes"]],
        "initial_majority": initial_majority, "initial_correct": initial_majority == gold,
        "round_messages": {f"round_{rd['round']}": rd["messages"] for rd in run["rounds"]},
        "final_individual_votes": final_votes, "final_vote_rationales": [v["rationale"] for v in run["final_votes"]],
        "final_vote_counts": dict(final_count), "final_majority": final_majority,
        "final_correct": final_majority == gold,
        "transition": (
            "WRONG_TO_CORRECT" if initial_majority != gold and final_majority == gold else
            "CORRECT_TO_WRONG" if initial_majority == gold and final_majority != gold else
            "CORRECT_TO_CORRECT" if initial_majority == gold else "WRONG_TO_WRONG"
        ),
        "generation_call_count": len(recorder.calls), "expected_generation_calls": expected_calls,
        "retry_count": max(0, len(recorder.calls) - expected_calls),
        "api_attempt_count": len(client.completions),
        "api_retry_count": sum(max(0, sum(1 for c in call.get("completion_attempts", [])) - 1) for call in recorder.calls),
        "token_usage": total_usage if usage_available else None,
        "task_protocol_validation": baseline_ok,
        "raw_run": run,
    }


def aggregate(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r["status"] == "complete"]
    initial_correct = sum(r["initial_correct"] for r in done)
    final_correct = sum(r["final_correct"] for r in done)
    initially_wrong = sum(not r["initial_correct"] for r in done)
    initially_correct = sum(r["initial_correct"] for r in done)
    rescue = sum(r["transition"] == "WRONG_TO_CORRECT" for r in done)
    regression = sum(r["transition"] == "CORRECT_TO_WRONG" for r in done)
    paired = Counter()
    paired_ids: dict[str, list[int]] = {key: [] for key in ("Both correct", "V2 only correct", "Ablation only correct", "Both wrong")}
    for row in done:
        prior = baseline.get(str(row["task_id"]))
        if prior is None:
            continue
        v2_ok, abl_ok = prior["majority_vote"] == prior["correct_answer"], row["final_correct"]
        key = "Both correct" if v2_ok and abl_ok else "V2 only correct" if v2_ok else "Ablation only correct" if abl_ok else "Both wrong"
        paired[key] += 1
        paired_ids[key].append(row["task_id"])
    n = len(done)
    return {
        "completed_tasks": n,
        "final_accuracy": final_correct / n if n else None,
        "final_correct_count": final_correct,
        "initial_majority_accuracy": initial_correct / n if n else None,
        "initial_correct_count": initial_correct,
        "rescue_count": rescue,
        "rescue_rate": rescue / initially_wrong if initially_wrong else None,
        "regression_count": regression,
        "regression_rate": regression / initially_correct if initially_correct else None,
        "correct_retention_rate": (initially_correct - regression) / initially_correct if initially_correct else None,
        "persistent_failure_count": sum(r["transition"] == "WRONG_TO_WRONG" for r in done),
        "paired_v2_comparison": {k: {"count": paired[k], "task_ids": paired_ids[k]} for k in paired_ids},
        "total_generation_calls": sum(r.get("generation_call_count", 0) for r in done),
        "total_api_attempts": sum(r.get("api_attempt_count", 0) for r in done),
        "total_api_retries": sum(r.get("api_retry_count", 0) for r in done),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = parser.parse_args()
    task_ids = args.task_ids or ([1, 4, 7] if args.phase == "smoke" else list(range(1, 66)))
    if len(set(task_ids)) != len(task_ids) or any(t < 1 or t > 65 for t in task_ids):
        parser.error("task IDs must be unique and in 1..65")
    out_dir = args.output_dir or OUT_ROOT / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path, prompt_hash, prompts = prepare_variant(args.variant, out_dir)
    config = {
        "variant": args.variant, "model": "gpt-4o-mini", "profile": "hidden", "rounds": 3,
        "seed": 42, "temperature": 0.0, "prompt_sha256": prompt_hash,
        "prompt_file": str(prompt_path.resolve()), "baseline_summary": str(BASELINE),
        "expected_calls_per_task": "5 * number_of_agents (excluding provider retries and output-parse retries)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = out_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("variant", "model", "profile", "rounds", "seed", "temperature", "prompt_sha256"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Existing run config mismatch at {key}: {config_path}")
    else:
        write_json(config_path, config)

    baseline = baseline_index()
    benchmark = load_benchmark(path=str(ROOT / "data/benchmark.json"))
    load_dotenv(args.dotenv)
    client = TracingClient("gpt-4o-mini", provider="openai", temperature=0.0, max_retries=3, timeout=90)
    all_rows = []
    for position, task_id in enumerate(task_ids, start=1):
        path = out_dir / f"task_{task_id}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("metadata", {}).get("prompt_sha256") != prompt_hash or saved.get("metadata", {}).get("variant") != args.variant:
                raise ValueError(f"Cached result prompt/config mismatch: {path}")
            row = saved["report"]
            all_rows.append(row)
            print(f"{args.variant} {args.phase}: task {task_id} cached ({position}/{len(task_ids)})", flush=True)
            continue
        task = find_task(benchmark, task_id=str(task_id))
        call_path = out_dir / f"task_{task_id}.calls.jsonl"
        call_path.unlink(missing_ok=True)
        client.completions.clear()
        recorder = CallRecorder(client, call_path)
        if args.variant == "V2_MINUS_AWARENESS":
            original_instantiate = patch_awareness_fact_labels()
        else:
            original_instantiate = None
        try:
            print(f"{args.variant} {args.phase}: task {task_id}/{task_ids[-1]} started ({task.name})", flush=True)
            run = run_scenario(task, recorder, Profile.HIDDEN, prompts, rounds=3, seed=42)
        except Exception as exc:
            write_json(out_dir / f"task_{task_id}.error.json", {
                "task_id": task_id, "variant": args.variant, "error_type": type(exc).__name__, "error": str(exc),
                "calls": recorder.calls, "created_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"{args.variant} task {task_id} ERROR {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            if original_instantiate is not None:
                simulator.instantiate_agents = original_instantiate
        run["generation_call_count"] = len(recorder.calls)
        run["expected_generation_calls"] = 5 * len(run["fact_assignments"])
        report = task_report(run, args.variant, prompt_hash, run["expected_generation_calls"], recorder, client, baseline[str(task_id)])
        checks = report["task_protocol_validation"]
        if not all(checks.values()):
            raise RuntimeError(f"Protocol mismatch against V2 baseline on task {task_id}: {checks}")
        if run["num_agents"] != len(run["initial_votes"]) or run["num_agents"] != len(run["final_votes"]):
            raise RuntimeError(f"Vote parse/count check failed on task {task_id}")
        if len(run["rounds"]) != 3 or any(len(rd["messages"]) != run["num_agents"] for rd in run["rounds"]):
            raise RuntimeError(f"Round/message count check failed on task {task_id}")
        result = {"metadata": {**config, "task_id": task_id, "created_at": datetime.now(timezone.utc).isoformat()}, "report": report}
        write_json(path, result)
        all_rows.append(report)
        print(
            f"{args.variant} {args.phase}: task {task_id} complete ({position}/{len(task_ids)}); "
            f"initial={report['initial_majority']} ({report['initial_correct']}), final={report['final_majority']} "
            f"({report['final_correct']}), calls={report['generation_call_count']}, saved={path}", flush=True,
        )
    summary = {
        "metadata": {**config, "phase": args.phase, "requested_task_ids": task_ids,
                     "finished_at": datetime.now(timezone.utc).isoformat()},
        "tasks": [{k: v for k, v in row.items() if k != "raw_run" and k != "round_messages"} for row in all_rows],
        "aggregate": aggregate(all_rows, baseline),
        "smoke_checks": {
            "all_tasks_complete": len(all_rows) == len(task_ids),
            "initial_vote_count_valid": all(len(row["initial_individual_votes"]) == len(row["final_individual_votes"]) for row in all_rows),
            "three_rounds_and_messages_valid": all(len(row["round_messages"]) == 3 and all(len(row["round_messages"][f"round_{i}"]) == len(row["initial_individual_votes"]) for i in (1, 2, 3)) for row in all_rows),
            "same_hidden_assignments_and_fact_order_as_v2": all(row["task_protocol_validation"]["same_hidden_assignment_and_fact_order"] for row in all_rows),
            "same_speaker_order_as_v2": all(row["task_protocol_validation"]["same_speaker_order_each_round"] for row in all_rows),
            "all_calls_logged": all((out_dir / f"task_{row['task_id']}.calls.jsonl").exists() for row in all_rows),
        },
    }
    summary_name = "smoke_summary.json" if args.phase == "smoke" else "summary.json"
    write_json(out_dir / summary_name, summary)
    if args.phase == "smoke":
        ok = all(summary["smoke_checks"].values()) and summary["aggregate"]["completed_tasks"] == len(task_ids)
        summary["smoke_passed"] = ok
        write_json(out_dir / summary_name, summary)
        print(f"{args.variant} SMOKE {'PASS' if ok else 'FAIL'}; results={out_dir / summary_name}", flush=True)
        return 0 if ok else 2
    print(f"{args.variant} FULL COMPLETE {summary['aggregate']['completed_tasks']}/{len(task_ids)}; results={out_dir / summary_name}", flush=True)
    return 0 if summary["aggregate"]["completed_tasks"] == len(task_ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
