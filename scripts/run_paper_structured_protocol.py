"""Run a paper-based adaptation of HiddenBench's Exchange-then-Decide protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.models import OpenAICompatibleClient
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import AgentState, Profile, format_discussion_transcript, instantiate_agents

PROMPT_PATH = ROOT / "analysis/paper_structured_protocol/prompts.json"
OUTPUT_DEFAULT = ROOT / "runs/paper_structured_protocol_gpt4o_mini_hidden_20261016"
BASELINE_V2 = ROOT / "runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015"
BASELINE_V1 = load_prompts(version="v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(template: list[str], values: dict[str, str]) -> str:
    text = "\n\n".join(template)
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unresolved = [token for token in ("{exchange_number}", "{public_transcript}", "{front_runner}") if token in text]
    if unresolved:
        raise ValueError(f"Unexpanded template placeholders: {unresolved}")
    return text


class UsageClient(OpenAICompatibleClient):
    """Same Chat Completions adapter with request/usage diagnostics per attempt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts: list[dict[str, Any]] = []
        self.logical_requests = 0

    def _completion(self, messages, response_format=None):
        self.logical_requests += 1
        last_error = None
        for attempt in range(self.max_retries):
            record: dict[str, Any] = {
                "attempt": attempt + 1,
                "messages": messages,
                "response_format": response_format,
            }
            try:
                kwargs: dict[str, Any] = {"model": self.model_name, "messages": messages}
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                usage = getattr(response, "usage", None)
                record.update({
                    "status": "ok",
                    "response_model": getattr(response, "model", None),
                    "usage": ({k: getattr(usage, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
                              if usage else None),
                })
                self.attempts.append(record)
                return response.choices[0].message.content or ""
            except Exception as exc:  # provider retry behavior mirrors the shared adapter
                last_error = exc
                record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
                self.attempts.append(record)
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


def initial_majority(votes: list[dict[str, Any]]) -> str:
    return Counter(v["vote"] for v in votes).most_common(1)[0][0]


def valid_cached(path: Path, task_id: int, prompt_hash: str, v1_hash: str, runner_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    meta = saved.get("metadata", {})
    if not (meta.get("task_id") == task_id and meta.get("protocol_prompt_sha256") == prompt_hash
            and meta.get("v1_prompt_sha256") == v1_hash and meta.get("seed") == 42
            and meta.get("model") == "gpt-4o-mini" and meta.get("profile") == "hidden"
            and meta.get("temperature") == 0.0 and meta.get("exchanges") == 2
            and meta.get("decide_passes") == 1 and meta.get("runner_sha256") == runner_hash):
        raise ValueError(f"Cached result has a mismatched manifest; refusing reuse: {path}")
    run = saved.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"Cached result has no run object; refusing reuse: {path}")
    n = run.get("num_agents")
    expected = n * 5 if isinstance(n, int) else None
    votes = (run.get("initial_votes"), run.get("final_votes"))
    agent_names = [a.get("agent") for a in run.get("fact_assignments", [])]
    if not expected or meta.get("call_count_expected") != expected:
        raise ValueError(f"Cached result has invalid expected call count; refusing reuse: {path}")
    if run.get("generation_call_count", 0) < expected or run.get("api_attempt_count", 0) < expected:
        raise ValueError(f"Cached result is incomplete (<5N calls); refusing reuse: {path}")
    if not isinstance(run.get("api_attempts"), list) or sum(a.get("status") == "ok" for a in run["api_attempts"]) < expected:
        raise ValueError(f"Cached result is missing successful API attempts; refusing reuse: {path}")
    if run.get("api_attempt_count") != len(run["api_attempts"]):
        raise ValueError(f"Cached result has inconsistent API attempt count; refusing reuse: {path}")
    stage_calls = run.get("stage_logical_calls", {})
    if any(stage_calls.get(k, 0) < n for k in ("initial_vote", "exchange_1", "exchange_2", "decide", "final_vote")):
        raise ValueError(f"Cached result does not contain all 5N stage calls; refusing reuse: {path}")
    if sum(stage_calls.values()) != run.get("generation_call_count"):
        raise ValueError(f"Cached result has inconsistent stage-call total; refusing reuse: {path}")
    if any(not isinstance(items, list) or len(items) != n or [v.get("agent") for v in items] != agent_names for items in votes):
        raise ValueError(f"Cached result has incomplete or misordered votes; refusing reuse: {path}")
    if len(run.get("exchanges", [])) != 2 or any(
        len(stage.get("messages", [])) != n or [m.get("agent") for m in stage["messages"]] != agent_names
        for stage in run.get("exchanges", [])
    ):
        raise ValueError(f"Cached result has incomplete Exchange messages; refusing reuse: {path}")
    if len(run.get("decide_messages", [])) != n or [m.get("agent") for m in run.get("decide_messages", [])] != agent_names:
        raise ValueError(f"Cached result has incomplete Decide messages; refusing reuse: {path}")
    allowed = run.get("possible_answers", [])
    if any(v.get("vote") not in allowed or not v.get("rationale") for part in votes for v in part):
        raise ValueError(f"Cached result has an invalid vote; refusing reuse: {path}")
    expected_winner = Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0]
    if run.get("majority_vote") != expected_winner or run.get("final_correct") != (expected_winner == run.get("correct_answer")):
        raise ValueError(f"Cached result has inconsistent scoring/tie-break; refusing reuse: {path}")
    if len(run.get("discussion_transcript", [])) != 3 * n:
        raise ValueError(f"Cached result has incomplete public transcript; refusing reuse: {path}")
    return saved


def static_validate() -> dict[str, Any]:
    prompt_data = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    assert prompt_data["protocol_version"] == "paper_structured_protocol_adaptation_v1"
    assert "1 or 2 concrete" in " ".join(prompt_data["exchange_prompt"])
    assert "front-runner may be incorrect" in " ".join(prompt_data["exchange_prompt"])
    assert "initial group front-runner" in " ".join(prompt_data["exchange_prompt"]).lower()
    assert "held fixed for both Exchange stages" in " ".join(prompt_data["exchange_prompt"])
    assert "strongest evidence" in " ".join(prompt_data["decide_prompt"])
    assert "main uncertainty" in " ".join(prompt_data["decide_prompt"])
    assert not any("private fact" in x.lower() or "information asymmetry" in x.lower() for x in prompt_data["exchange_prompt"] + prompt_data["decide_prompt"])
    benchmark = load_benchmark(path=str(ROOT / "data/benchmark.json"))
    assert len(benchmark) == 65
    assignments_checked = 0
    for task_id in range(1, 66):
        task = find_task(benchmark, task_id=str(task_id))
        reference_path = BASELINE_V2 / f"task_{task_id}.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing V2 reference task result: {reference_path}")
        ref = json.loads(reference_path.read_text(encoding="utf-8"))["runs"][0]
        agents, assignments = instantiate_agents(task, _NoCallClient(), Profile.HIDDEN, BASELINE_V1, random.Random(42))
        assert len(agents) == ref["num_agents"]
        assert assignments == ref["fact_assignments"], f"Seed-42 fact assignment mismatch in task {task_id}"
        assignments_checked += 1
        fake_transcript = "Person 1: example evidence."
        exchange = expand(prompt_data["exchange_prompt"], {"exchange_number": "1", "public_transcript": fake_transcript, "front_runner": task.possible_answers[0]})
        decide = expand(prompt_data["decide_prompt"], {"public_transcript": fake_transcript})
        assert "{public_transcript}" not in exchange + decide
    agents_total = sum(len(t.hidden_information) for t in benchmark)
    expected = agents_total * 5  # initial vote + 2 exchange + 1 decide + final vote
    return {
        "status": "PASS",
        "tasks": len(benchmark),
        "task_assignments_matched_to_saved_v2": assignments_checked,
        "agents_total": agents_total,
        "expected_model_calls": expected,
        "call_formula": "sum_agents * (1 initial vote + 2 exchange messages + 1 decide summary + 1 final vote)",
        "prompt_sha256": {"adaptation": sha(PROMPT_PATH), "v1_backbone": sha(ROOT / "src/hiddenbench/prompts.json"), "v2_reference": sha(ROOT / "src/hiddenbench/prompts_v2.json")},
        "runner_sha256": sha(Path(__file__).resolve()),
        "vote_implementation": "AgentState.vote + vote_json_schema from src/hiddenbench/simulator.py",
        "tie_break": "Counter(vote).most_common(1)[0][0] (first encountered option wins ties)",
        "api_called": False,
    }


class _NoCallClient:
    model_name = "static-validation-no-api"


def run_task(task_id: int, client: UsageClient, prompts: dict[str, Any], output_dir: Path, seed: int) -> dict[str, Any]:
    task = find_task(load_benchmark(path=str(ROOT / "data/benchmark.json")), task_id=str(task_id))
    agents, assignments = instantiate_agents(task, client, Profile.HIDDEN, BASELINE_V1, random.Random(seed))
    run: dict[str, Any] = {
        "task_id": task.id, "scenario": task.name, "profile": "hidden", "model": client.model_name,
        "num_agents": len(agents), "correct_answer": task.correct_answer,
        "possible_answers": list(task.possible_answers), "fact_assignments": assignments,
        "initial_votes": [], "exchanges": [], "decide_messages": [], "final_votes": [],
    }
    call_start = len(client.attempts)
    logical_start = client.logical_requests
    stage_calls: dict[str, int] = {}
    initial_prompt = render_prompt(BASELINE_V1.first_vote_prompt, {
        "possible_answers": ", ".join(task.possible_answers),
        "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
    })
    for agent in agents:
        run["initial_votes"].append({"agent": agent.name, **agent.vote(initial_prompt, task.possible_answers)})
    stage_calls["initial_vote"] = client.logical_requests - logical_start

    front_runner = initial_majority(run["initial_votes"])
    transcript: list[dict[str, Any]] = []
    for exchange_no in (1, 2):
        stage_start = client.logical_requests
        stage = {"stage": f"exchange_{exchange_no}", "messages": []}
        for agent in agents:
            private_state = ""
            if exchange_no == 1:
                own_initial = next(v for v in run["initial_votes"] if v["agent"] == agent.name)
                private_state = (
                    f"Your initial vote was: {own_initial['vote']}\n"
                    f"Your initial rationale was: {own_initial['rationale']}\n\n"
                )
            prompt = expand(prompts["exchange_prompt"], {
                "exchange_number": str(exchange_no),
                "public_transcript": format_discussion_transcript(transcript),
                "front_runner": front_runner,
            })
            prompt = private_state + prompt
            response = agent.chat(prompt)
            row = {"agent": agent.name, "prompt": prompt, "response": response}
            stage["messages"].append(row)
            transcript.append({"round": exchange_no, "agent": agent.name, "response": response})
        run["exchanges"].append(stage)
        stage_calls[f"exchange_{exchange_no}"] = client.logical_requests - stage_start

    decide_start = client.logical_requests
    decide = {"stage": "decide", "messages": []}
    for agent in agents:
        prompt = expand(prompts["decide_prompt"], {"public_transcript": format_discussion_transcript(transcript)})
        response = agent.chat(prompt)
        decide["messages"].append({"agent": agent.name, "prompt": prompt, "response": response})
        transcript.append({"round": 3, "agent": agent.name, "response": response})
    run["decide_messages"] = decide["messages"]
    stage_calls["decide"] = client.logical_requests - decide_start

    final_start = client.logical_requests
    final_prompt = render_prompt(BASELINE_V1.vote_prompt, {
        "group_discussion": format_discussion_transcript(transcript),
        "possible_answers": ", ".join(task.possible_answers),
    })
    for agent in agents:
        run["final_votes"].append({"agent": agent.name, **agent.vote(final_prompt, task.possible_answers)})
    stage_calls["final_vote"] = client.logical_requests - final_start
    run["majority_vote"] = Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0]
    run["final_correct"] = run["majority_vote"] == task.correct_answer
    run["discussion_transcript"] = transcript
    call_end = len(client.attempts)
    run["api_attempts"] = client.attempts[call_start:call_end]
    run["generation_call_count"] = client.logical_requests - logical_start
    run["api_attempt_count"] = call_end - call_start
    run["stage_logical_calls"] = stage_calls
    usage = [a.get("usage") for a in run["api_attempts"] if a.get("status") == "ok" and a.get("usage")]
    run["token_usage"] = {key: sum((u.get(key) or 0) for u in usage) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Validate prompt/data/assignments without API calls")
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(1, 66)))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--smoke-only", action="store_true", help="Convenience flag restricting execution to Tasks 1 and 2")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(static_validate(), ensure_ascii=False, indent=2))
        return 0

    if args.smoke_only:
        if args.task_ids != list(range(1, 66)):
            parser.error("--smoke-only cannot be combined with explicit --task-ids")
        args.task_ids = [1, 2]
    if not args.task_ids or len(set(args.task_ids)) != len(args.task_ids) or any(i < 1 or i > 65 for i in args.task_ids):
        parser.error("task IDs must be unique integers in 1..65")

    validation = static_validate()
    load_dotenv(args.dotenv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    protocol_hash = sha(PROMPT_PATH)
    v1_hash = sha(ROOT / "src/hiddenbench/prompts.json")
    runner_hash = sha(Path(__file__).resolve())
    client = UsageClient("gpt-4o-mini", provider="openai", temperature=0.0, max_retries=3, timeout=90)
    summary: list[dict[str, Any]] = []
    for task_id in args.task_ids:
        path = output_dir / f"task_{task_id}.json"
        cached = valid_cached(path, task_id, protocol_hash, v1_hash, runner_hash)
        if cached:
            summary.append({"task_id": task_id, "status": "reused", "majority_vote": cached["run"]["majority_vote"], "correct": cached["run"]["final_correct"]})
            print(f"Task {task_id}: cached result reused", flush=True)
            continue
        print(f"Task {task_id}/65 started", flush=True)
        task_attempt_start = len(client.attempts)
        try:
            run = run_task(task_id, client, prompts, output_dir, seed=42)
            n = len(run["fact_assignments"])
            if run["generation_call_count"] < n * 5 or any(run["stage_logical_calls"].get(k, 0) < n for k in ("initial_vote", "exchange_1", "exchange_2", "decide", "final_vote")):
                raise RuntimeError(f"Task {task_id}: expected 5N complete logical stage calls, got {run.get('stage_logical_calls')}")
        except Exception as exc:
            failure = {"task_id": task_id, "status": "failed", "error_type": type(exc).__name__,
                       "error": str(exc), "api_attempts": client.attempts[task_attempt_start:],
                       "created_at": datetime.now(timezone.utc).isoformat(),
                       "note": "Any previously saved successful task JSON remains intact; rerun resumes by validating completed task files."}
            fail_path = output_dir / f"task_{task_id}.error.json"
            tmp_error = fail_path.with_suffix(".json.tmp")
            tmp_error.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_error, fail_path)
            summary.append({"task_id": task_id, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "error_file": str(fail_path)})
            print(f"Task {task_id}: FAILED {type(exc).__name__}: {exc}; saved={fail_path}", flush=True)
            continue
        meta = {"task_id": task_id, "condition": "Paper-based Structured Protocol Adaptation", "model": client.model_name,
                "profile": "hidden", "seed": 42, "temperature": 0.0, "exchanges": 2, "decide_passes": 1,
                "protocol_prompt_file": str(PROMPT_PATH.relative_to(ROOT)), "protocol_prompt_sha256": protocol_hash,
                "v1_backbone_prompt_file": "src/hiddenbench/prompts.json", "v1_prompt_sha256": v1_hash,
                "runner_sha256": runner_hash,
                "fact_assignment_seed": 42, "evaluation": "existing exact answer equality and first-seen Counter.most_common tie break",
                "call_count_expected": len(run["fact_assignments"]) * 5, "created_at": datetime.now(timezone.utc).isoformat(),
                "static_validation": validation}
        saved = {"metadata": meta, "run": run}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if path.exists():
            tmp.unlink(missing_ok=True)
            raise FileExistsError(f"Refusing to overwrite an existing task result: {path}")
        os.replace(tmp, path)
        summary.append({"task_id": task_id, "status": "complete", "majority_vote": run["majority_vote"], "correct": run["final_correct"], "calls": run["generation_call_count"], "tokens": run["token_usage"]})
        print(f"Task {task_id}: majority={run['majority_vote']}; correct={run['final_correct']}; calls={run['generation_call_count']}; saved={path}", flush=True)
    summary_obj = {"condition": "Paper-based Structured Protocol Adaptation", "metadata": validation, "tasks": summary,
                   "completed": sum(x["status"] != "failed" for x in summary), "failed": sum(x["status"] == "failed" for x in summary)}
    summary_path = output_dir / "summary.json"
    tmp_summary = summary_path.with_suffix(".json.tmp")
    tmp_summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_summary, summary_path)
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2), flush=True)
    return int(summary_obj["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
