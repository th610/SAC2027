"""Run frozen V2 Module A1/A2 ablations with smoke gate and resumable tasks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, format_discussion_transcript, instantiate_agents, run_scenario
from run_v2_ablations import CallRecorder, TracingClient


ROOT = Path(__file__).resolve().parents[1]
FULL_PROMPT = ROOT / "src/hiddenbench/prompts_v2.json"
FULL_RESULTS = ROOT / "runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015"
PROMPTS = {
    "minus_a1": ROOT / "analysis/v2_module_a_ablation/minus_a1.json",
    "minus_a2": ROOT / "analysis/v2_module_a_ablation/minus_a2.json",
}
OUT = ROOT / "runs/v2_module_a_ablation_gpt4o_mini_hidden_20261016"
MODEL = "gpt-4o-mini"
ROUNDS = 3
SEED = 42
TEMP = 0.0
PLACEHOLDER = re.compile(r"%[a-z_]+%")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_full(task_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = FULL_RESULTS / f"task_{task_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved, saved["runs"][0]


def baseline_manifest() -> dict[str, Any]:
    paths = [FULL_PROMPT, FULL_RESULTS / "summary.json"] + [FULL_RESULTS / f"task_{i}.json" for i in range(1, 66)]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def _replace_once(prompt: dict[str, Any], section: str, old: str, new: str) -> None:
    hits = 0
    for i, text in enumerate(prompt[section]):
        if old in text:
            prompt[section][i] = text.replace(old, new, 1)
            hits += 1
    assert hits == 1, (section, old, hits)


def audit_prompt_diffs(full: dict[str, Any], variants: dict[str, Any]) -> None:
    """Assert each prompt differs only by its predeclared A1/A2 text edits."""
    expected = {name: copy.deepcopy(full) for name in variants}
    _replace_once(expected["minus_a1"], "system_prompt",
        "All participants know the same %num_shared_facts% shared facts.\nEach participant also has exactly one unique private fact.\nOther participants do not know your private fact unless it is communicated during discussion.\nYou do not know other participants' private facts until they are communicated during discussion.\n", "")
    _replace_once(expected["minus_a1"], "system_prompt",
        "- Your initial judgment may be wrong because you do not have the other participants' private facts.\n", "")
    _replace_once(expected["minus_a1"], "system_prompt",
        "During discussion, prioritize decision-relevant information that other participants may not yet know.\nEspecially communicate evidence that rules out an option, rescues an option, or materially changes the decision.",
        "During discussion, proactively share decision-relevant information from the facts available to you.\nEspecially communicate evidence that rules out an option, rescues an option, or materially changes the decision.")
    _replace_once(expected["minus_a1"], "first_user_prompt",
        "Share the most decision-relevant information you currently have, especially information that other participants may not know.",
        "Share the most decision-relevant information you currently have.")
    _replace_once(expected["minus_a1"], "user_prompt",
        "1. your own shared and private information, and\n2. the evidence communicated by the other participants.",
        "1. the evidence available to you, and\n2. the evidence communicated by the other participants.")
    _replace_once(expected["minus_a1"], "user_prompt", "newly revealed private information", "newly communicated evidence")
    _replace_once(expected["minus_a2"], "system_prompt",
        "During discussion, prioritize decision-relevant information that other participants may not yet know.\nEspecially communicate evidence that rules out an option, rescues an option, or materially changes the decision.\nAvoid merely repeating an existing consensus.",
        "Avoid merely repeating an existing consensus.")
    _replace_once(expected["minus_a2"], "first_user_prompt",
        "Share the most decision-relevant information you currently have, especially information that other participants may not know.\nExplain briefly which candidate or candidates this evidence affects.",
        "Explain briefly how the available evidence affects the candidate or candidates.")
    _replace_once(expected["minus_a2"], "user_prompt",
        "Contribute the most useful non-redundant evidence you can.\nIf new evidence changes your previous judgment, update it.",
        "If new evidence changes your previous judgment, update it.")
    for name in variants:
        assert variants[name] == expected[name], f"Unexpected prompt edits in {name}"
        p = variants[name]
        assert p["system_prompt"][2:4] == full["system_prompt"][2:4]
        assert p["first_vote_prompt"] == full["first_vote_prompt"]
        assert p["vote_prompt"] == full["vote_prompt"]
        assert p["generation_prompt"] == full["generation_prompt"]
    assert "proactively share decision-relevant information" in variants["minus_a1"]["system_prompt"][5]
    assert "Share the most decision-relevant information" in variants["minus_a1"]["first_user_prompt"][0]
    assert "All participants know the same" in variants["minus_a2"]["system_prompt"][1]
    assert "you do not have the other participants' private facts" in variants["minus_a2"]["system_prompt"][4]


def preflight() -> dict[str, Any]:
    benchmark = load_benchmark(path=str(ROOT / "data/benchmark.json"))
    full = load_prompts(version="v2")
    variants = {name: load_prompts(path=path) for name, path in PROMPTS.items()}
    full_raw = json.loads(FULL_PROMPT.read_text(encoding="utf-8"))
    variant_raw = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in PROMPTS.items()}
    audit_prompt_diffs(full_raw, variant_raw)
    summary = json.loads((FULL_RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] == 65 and summary["failed"] == 0
    assert sum(bool(row["majority_correct"]) for row in summary["tasks"]) == 57
    assert {int(row["task_id"]) for row in summary["tasks"]} == set(range(1, 66))
    assert FULL_RESULTS.resolve() != OUT.resolve()
    assert len({p.resolve() for p in PROMPTS.values()}) == 2
    assert all(not str(OUT.resolve()).startswith(str(p.parent.resolve()) + "/") for p in PROMPTS.values())
    full_matches = Counter()
    for task_id in range(1, 66):
        task = find_task(benchmark, task_id=str(task_id))
        saved, prior = load_full(task_id)
        assert saved["metadata"]["model"] == MODEL
        assert saved["metadata"]["profile"] == "hidden"
        assert saved["metadata"]["rounds"] == ROUNDS
        assert saved["metadata"]["seed"] == SEED
        assert saved["metadata"]["temperature"] == TEMP
        assert prior["task_id"] == task.id and prior["correct_answer"] == task.correct_answer
        assert prior["possible_answers"] == task.possible_answers
        baseline_agents, baseline_assignment = instantiate_agents(task, None, Profile.HIDDEN, full, random.Random(SEED))
        assert baseline_assignment == prior["fact_assignments"]
        assert [a.history[0]["content"] for a in baseline_agents] == [x["prompt"] for x in prior["prompt_diagnostics"]["system_prompts"]]
        full_matches["system"] += 1
        initial = render_prompt(full.first_vote_prompt, {"possible_answers": ", ".join(task.possible_answers)})
        assert initial == prior["prompt_diagnostics"]["initial_vote_prompt"]
        full_matches["initial"] += 1
        iv = prior["initial_votes"][0]
        first = f"Your initial vote was: {iv['vote']}\nYour initial rationale was: {iv['rationale']}\n\n" + full.first_user_prompt
        assert first == prior["rounds"][0]["messages"][0]["prompt"]
        full_matches["r1_first"] += 1
        final = render_prompt(full.vote_prompt, {
            "group_discussion": format_discussion_transcript(prior["rounds"][-1]["messages"]),
            "possible_answers": ", ".join(task.possible_answers),
        })
        assert final == prior["prompt_diagnostics"]["final_vote_prompt"]
        full_matches["final"] += 1
        for name, prompt in variants.items():
            agents, assignment = instantiate_agents(task, None, Profile.HIDDEN, prompt, random.Random(SEED))
            assert assignment == prior["fact_assignments"], (task_id, name, "assignment")
            assert len(agents) == len(prior["initial_votes"])
            rendered = [a.history[0]["content"] for a in agents]
            rendered.append(render_prompt(prompt.first_vote_prompt, {"possible_answers": ", ".join(task.possible_answers)}))
            rendered.append(prompt.first_user_prompt)
            rendered.append(render_prompt(prompt.user_prompt, {"messages": "Person 1: example", "extra": ""}))
            rendered.append(render_prompt(prompt.vote_prompt, {"group_discussion": "Person 1: example", "possible_answers": ", ".join(task.possible_answers)}))
            assert all(not PLACEHOLDER.search(x) for x in rendered), (task_id, name, "placeholder")
            assert prompt.first_vote_prompt.endswith(full.first_vote_prompt.split("\n\n")[-1])
            assert prompt.vote_prompt.endswith(full.vote_prompt.split("\n\n")[-1])
    return {
        "status": "PASS", "tasks": 65, "full_correct": 57,
        "full_prompt_matches_saved": dict(full_matches),
        "model": MODEL, "temperature": TEMP, "seed": SEED, "rounds": ROUNDS,
        "prompt_sha256": {"full": sha(FULL_PROMPT), **{name: sha(path) for name, path in PROMPTS.items()}},
        "baseline_manifest": baseline_manifest(),
    }


def validate_run(run: dict[str, Any], prior: dict[str, Any], task_id: int) -> None:
    assert run["task_id"] == prior["task_id"] and run["correct_answer"] == prior["correct_answer"]
    assert run["possible_answers"] == prior["possible_answers"]
    assert run["fact_assignments"] == prior["fact_assignments"]
    assert len(run["initial_votes"]) == len(prior["initial_votes"])
    assert len(run["rounds"]) == ROUNDS
    assert [[x["agent"] for x in r["messages"]] for r in run["rounds"]] == [[x["agent"] for x in r["messages"]] for r in prior["rounds"]]
    assert all(len(r["messages"]) == len(prior["initial_votes"]) for r in run["rounds"])
    assert all(v["vote"] in run["possible_answers"] and v["rationale"] for v in run["initial_votes"] + run["final_votes"])
    assert len(run["final_votes"]) == len(prior["final_votes"])
    assert run["majority_vote"] == Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0]
    assert all("prompt" in m and "response" in m for r in run["rounds"] for m in r["messages"])


def run_task(name: str, task_id: int, client: TracingClient, manifest: dict[str, Any]) -> str:
    path = OUT / name / f"task_{task_id}.json"
    _, prior = load_full(task_id)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        meta = saved["metadata"]
        expected = {"condition": name, "model": MODEL, "temperature": TEMP, "seed": SEED, "rounds": ROUNDS, "prompt_sha256": sha(PROMPTS[name])}
        assert all(meta.get(k) == v for k, v in expected.items()), (name, task_id, "cache config")
        assert meta["baseline_manifest"] == manifest, (name, task_id, "baseline changed")
        validate_run(saved["run"], prior, task_id)
        assert saved["generation_call_count"] >= len(prior["initial_votes"]) * (ROUNDS + 2)
        print(f"{name} task {task_id}/65 cached", flush=True)
        return "cached"
    task = find_task(load_benchmark(path=str(ROOT / "data/benchmark.json")), task_id=str(task_id))
    call_path = OUT / name / f"task_{task_id}.calls.jsonl"
    call_path.parent.mkdir(parents=True, exist_ok=True)
    if call_path.exists():
        # An interrupted task has no completed JSON cache. Preserve its partial
        # calls for audit, then rerun the task from its original information.
        archive = call_path.with_name(call_path.name + ".interrupted-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        assert not archive.exists(), archive
        call_path.rename(archive)
    client.completions.clear()
    recorder = CallRecorder(client, call_path)
    prompts = load_prompts(path=PROMPTS[name])
    try:
        print(f"{name} task {task_id}/65 started", flush=True)
        run = run_scenario(task, recorder, Profile.HIDDEN, prompts, rounds=ROUNDS, seed=SEED)
        validate_run(run, prior, task_id)
        expected = len(prior["initial_votes"]) * (ROUNDS + 2)
        assert len(recorder.calls) >= expected
        actual_tokens = {k: sum((r.get("usage") or {}).get(k) or 0 for r in client.completions) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        value = {
            "metadata": {"condition": name, "model": MODEL, "temperature": TEMP, "seed": SEED, "rounds": ROUNDS,
                         "prompt_file": str(PROMPTS[name]), "prompt_sha256": sha(PROMPTS[name]),
                         "baseline_manifest": manifest, "created_at": datetime.now(timezone.utc).isoformat()},
            "generation_call_count": len(recorder.calls), "expected_generation_calls": expected,
            "api_attempt_count": len(client.completions),
            "retry_count": len(client.completions) - len(recorder.calls),
            "token_usage": actual_tokens, "run": run,
        }
        write_json(path, value)
        print(f"{name} task {task_id}/65 complete; majority={run['majority_vote']}; correct={run['majority_vote'] == task.correct_answer}; calls={len(recorder.calls)}", flush=True)
        return "complete"
    except Exception as exc:
        write_json(OUT / name / f"task_{task_id}.error.json", {"condition": name, "task_id": task_id,
                   "error_type": type(exc).__name__, "error": str(exc), "call_count": len(recorder.calls),
                   "api_attempt_count": len(client.completions)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "pipeline", "full"), required=True)
    parser.add_argument("--condition", choices=tuple(PROMPTS))
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.phase == "full" and not args.condition:
        parser.error("--condition is required for --phase full")
    check = preflight()
    if args.phase == "preflight":
        print(json.dumps({k: v for k, v in check.items() if k != "baseline_manifest"}, indent=2))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    gate = OUT / "preflight.json"
    if gate.exists():
        assert json.loads(gate.read_text(encoding="utf-8")) == check, "Preflight inputs changed"
    else:
        write_json(gate, check)
    load_dotenv(args.dotenv)
    client = TracingClient(MODEL, provider="openai", temperature=TEMP, max_retries=3, timeout=90)
    manifest = check["baseline_manifest"]
    # Both smoke calls must succeed before any other task starts.
    selected = (args.condition,) if args.phase == "full" else tuple(PROMPTS)
    for name in selected:
        run_task(name, 1, client, manifest)
    if args.phase in {"pipeline", "smoke"}:
        print("SMOKE_GATE_PASS: both conditions completed Task 1", flush=True)
    if args.phase == "smoke":
        return 0
    for name in selected:
        for task_id in range(1, 66):
            run_task(name, task_id, client, manifest)
    assert baseline_manifest() == manifest, "Frozen Full inputs changed during experiment"
    print(f"FULL_COMPLETE: {','.join(selected)} has 65 valid task files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
