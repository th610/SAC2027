"""Run unchanged HiddenBench V2 prompts against one local Qwen3-8B BF16 vLLM server."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import subprocess
import time
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.models import OpenAICompatibleClient
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, format_discussion_transcript, instantiate_agents, run_scenario

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "src/hiddenbench/prompts_v2.json"
BASELINE = ROOT / "runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015"
OUTPUT = ROOT / "runs/v2_full_qwen3_8b_bf16_hidden_20261016"
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PORT = 8011
MODEL_LEN = 16384
MAX_OUTPUT = 1024
SEED = 42
ROUNDS = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def baseline(task_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    p = BASELINE / f"task_{task_id}.json"
    saved = json.loads(p.read_text(encoding="utf-8"))
    return saved, saved["runs"][0]


def baseline_hashes() -> dict[str, str]:
    paths = [PROMPT_PATH, BASELINE / "summary.json"] + [BASELINE / f"task_{i}.json" for i in range(1, 66)]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def load_local_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def gpu_used_mib(gpu: int) -> int:
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])


class CheckedLocalClient(OpenAICompatibleClient):
    """Same vote/chat interface as V2, with explicit local decoding diagnostics."""

    def __init__(self, *, tokenizer: Any, base_url: str):
        super().__init__(MODEL, provider="openai", api_key="EMPTY", base_url=base_url,
                         temperature=0.0, max_retries=3, timeout=180)
        self.tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []
        self.vote_encoding_repairs: list[dict[str, str]] = []
        self.max_input_tokens = 0
        self._tokenizer_lock = Lock()

    def chat_json(self, messages, schema):
        """Repair a strict UTF-8/Latin-1 mojibake round trip in vote labels only."""
        result = super().chat_json(messages, schema)
        vote = str(result.get("vote", ""))
        allowed = schema.get("properties", {}).get("vote", {}).get("enum", [])
        repaired = repair_vote_label(vote, allowed)
        if repaired is not None and repaired != vote:
            self.vote_encoding_repairs.append({"returned_vote": vote, "matched_option": repaired})
            result = dict(result)
            result["vote"] = repaired
        return result


    def _completion(self, messages, response_format=None) -> str:
        with self._tokenizer_lock:
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        input_tokens = len(input_ids["input_ids"] if hasattr(input_ids, "keys") else input_ids)
        self.max_input_tokens = max(self.max_input_tokens, input_tokens)
        if input_tokens + MAX_OUTPUT > MODEL_LEN:
            raise ValueError(f"Context limit guard: {input_tokens}+{MAX_OUTPUT}>{MODEL_LEN}")
        last_error: Exception | None = None
        for attempt in range(3):
            record: dict[str, Any] = {"messages": messages, "input_tokens_measured": input_tokens, "attempt": attempt + 1,
                                      "temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_OUTPUT,
                                      "enable_thinking": False, "json_schema": response_format is not None}
            try:
                response = self.client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=0.0, top_p=1.0,
                    max_tokens=MAX_OUTPUT, response_format=response_format,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                usage = response.usage
                record.update(
                    finish_reason=choice.finish_reason,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    raw_response=content,
                )
                self.calls.append(record)
                if choice.finish_reason == "length":
                    raise ValueError("Completion truncated: finish_reason=length")
                if "<think>" in content or "</think>" in content:
                    raise ValueError("Thinking output found with enable_thinking=False")
                if not content.strip():
                    raise ValueError("Empty model response")
                return content
            except Exception as exc:
                last_error = exc
                if "raw_response" not in record:
                    record.update(error_type=type(exc).__name__, error=str(exc))
                    self.calls.append(record)
                if isinstance(exc, ValueError):
                    raise
                time.sleep(min(2 ** attempt, 8))
        assert last_error is not None
        raise last_error


def repair_vote_label(vote: str, allowed: list[str]) -> str | None:
    """Return an exact allowed option, repairing only Unicode normalization/mojibake."""
    if vote in allowed:
        return vote
    candidates = [unicodedata.normalize("NFC", vote)]
    try:
        candidates.append(unicodedata.normalize("NFC", vote.encode("latin-1").decode("utf-8")))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    matches = {candidate for candidate in candidates if candidate in allowed}
    return next(iter(matches)) if len(matches) == 1 else None


def validate_static(model_dir: Path) -> dict[str, Any]:
    prompt = load_prompts(version="v2")
    benchmark = load_benchmark(path=str(ROOT / "data/benchmark.json"))
    summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] == 65 and summary["failed"] == 0
    assert sum(row["majority_correct"] for row in summary["tasks"]) == 57
    matches = Counter()
    for task_id in range(1, 66):
        task = find_task(benchmark, task_id=str(task_id))
        old_meta, old = baseline(task_id)
        assert old_meta["metadata"]["model"] == "gpt-4o-mini"
        assert old["task_id"] == task.id and old["correct_answer"] == task.correct_answer
        assert old["possible_answers"] == task.possible_answers
        agents, assigned = instantiate_agents(task, None, Profile.HIDDEN, prompt, random.Random(SEED))
        assert assigned == old["fact_assignments"]
        assert [a.history[0]["content"] for a in agents] == [x["prompt"] for x in old["prompt_diagnostics"]["system_prompts"]]
        matches["system_and_facts"] += 1
        initial = render_prompt(prompt.first_vote_prompt, {"possible_answers": ", ".join(task.possible_answers)})
        assert initial == old["prompt_diagnostics"]["initial_vote_prompt"]
        matches["initial_vote"] += 1
        iv = old["initial_votes"][0]
        first = f"Your initial vote was: {iv['vote']}\nYour initial rationale was: {iv['rationale']}\n\n" + prompt.first_user_prompt
        assert first == old["rounds"][0]["messages"][0]["prompt"]
        matches["r1_first"] += 1
        final = render_prompt(prompt.vote_prompt, {
            "group_discussion": format_discussion_transcript(old["rounds"][-1]["messages"]),
            "possible_answers": ", ".join(task.possible_answers),
        })
        assert final == old["prompt_diagnostics"]["final_vote_prompt"]
        matches["final_vote"] += 1
        assert len(agents) == len(old["initial_votes"])
        assert len(old["rounds"]) == ROUNDS
    config_path = model_dir / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert model_config["architectures"] == ["Qwen3ForCausalLM"]
    assert "quantization_config" not in model_config
    assert model_config["torch_dtype"] == "bfloat16"
    assert model_config["max_position_embeddings"] >= MODEL_LEN
    assert model_dir.name == REVISION
    return {"status": "PASS", "task_count": 65, "baseline_correct": 57,
            "matches": dict(matches), "prompt_sha256": sha(PROMPT_PATH),
            "model": MODEL, "revision": REVISION, "precision": "bfloat16", "quantization": None,
            "model_dir": str(model_dir), "model_len": MODEL_LEN, "max_output_tokens": MAX_OUTPUT,
            "baseline_hashes": baseline_hashes()}


def validate_run(run: dict[str, Any], old: dict[str, Any]) -> None:
    assert run["task_id"] == old["task_id"] and run["correct_answer"] == old["correct_answer"]
    assert run["possible_answers"] == old["possible_answers"]
    assert run["fact_assignments"] == old["fact_assignments"]
    assert len(run["rounds"]) == ROUNDS and len(run["final_votes"]) == len(old["final_votes"])
    assert [[m["agent"] for m in rd["messages"]] for rd in run["rounds"]] == [[m["agent"] for m in rd["messages"]] for rd in old["rounds"]]
    assert all(v["vote"] in run["possible_answers"] and v["rationale"] for v in run["initial_votes"] + run["final_votes"])
    assert Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0] == run["majority_vote"]
    assert all(m["response"].strip() for rd in run["rounds"] for m in rd["messages"])


def run_task(task_id: int, tokenizer: Any, static: dict[str, Any], base_url: str) -> dict[str, Any]:
    path = OUTPUT / f"task_{task_id}.json"
    _, old = baseline(task_id)
    expected_meta = {"model": MODEL, "revision": REVISION, "prompt_sha256": static["prompt_sha256"],
                     "seed": SEED, "rounds": ROUNDS, "temperature": 0.0, "enable_thinking": False,
                     "baseline_hashes": static["baseline_hashes"]}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert all(saved["metadata"].get(k) == v for k, v in expected_meta.items()), f"Cache config mismatch: {path}"
        validate_run(saved["run"], old)
        print(f"Task {task_id}/65 cached", flush=True)
        return saved
    task = find_task(load_benchmark(path=str(ROOT / "data/benchmark.json")), task_id=str(task_id))
    client = CheckedLocalClient(tokenizer=tokenizer, base_url=base_url)
    started = time.monotonic()
    try:
        print(f"Task {task_id}/65 started", flush=True)
        run = run_scenario(task, client, Profile.HIDDEN, load_prompts(version="v2"), rounds=ROUNDS, seed=SEED)
        validate_run(run, old)
        expected_calls = len(old["initial_votes"]) * (ROUNDS + 2)
        assert len(client.calls) >= expected_calls
        assert all(c.get("finish_reason") != "length" for c in client.calls)
        saved = {"metadata": {**expected_meta, "created_at": datetime.now(timezone.utc).isoformat()},
                 "generation_call_count": len(client.calls), "expected_generation_calls": expected_calls,
                 "retry_count": len(client.calls) - expected_calls,
                 "elapsed_seconds": time.monotonic() - started,
                 "max_input_tokens": client.max_input_tokens,
                 "vote_encoding_repairs": client.vote_encoding_repairs,
                 "usage": {"prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in client.calls),
                           "completion_tokens": sum(c.get("completion_tokens") or 0 for c in client.calls)},
                 "completion_diagnostics": client.calls, "run": run}
        atomic_json(path, saved)
        print(f"Task {task_id}/65 complete; majority={run['majority_vote']}; correct={run['majority_vote'] == run['correct_answer']}; calls={len(client.calls)}", flush=True)
        return saved
    except Exception as exc:
        atomic_json(OUTPUT / f"task_{task_id}.error.json", {"task_id": task_id,
                    "error_type": type(exc).__name__, "error": str(exc), "calls": client.calls,
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic() - started})
        print(f"Task {task_id}/65 FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise


async def check_server(base_url: str) -> dict[str, Any]:
    from openai import AsyncOpenAI
    async with AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=30) as client:
        models = await client.models.list()
    ids = [x.id for x in models.data]
    assert MODEL in ids, f"Expected {MODEL} at {base_url}; got {ids}"
    return {"base_url": base_url, "served_models": ids}


async def phase_run(ids: list[int], concurrency: int, tokenizer: Any, static: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async def one(task_id: int) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(run_task, task_id, tokenizer, static, base_url)
    return await asyncio.gather(*(one(i) for i in ids))


async def monitor_gpu(gpu: int, stop: asyncio.Event, observations: list[int]) -> None:
    while not stop.is_set():
        try:
            observations.append(await asyncio.to_thread(gpu_used_mib, gpu))
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            continue


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1-p)/n + z*z/(4*n*n)) / d
    return [center-half, center+half]


def report() -> dict[str, Any]:
    rows = []
    for i in range(1, 66):
        p = OUTPUT / f"task_{i}.json"
        if not p.exists():
            continue
        x = json.loads(p.read_text(encoding="utf-8"))
        _, old = baseline(i)
        validate_run(x["run"], old)
        rows.append({"task_id": i, "qwen_correct": x["run"]["majority_vote"] == x["run"]["correct_answer"],
                     "gpt_correct": old["majority_vote"] == old["correct_answer"],
                     "qwen_majority": x["run"]["majority_vote"], "gold": x["run"]["correct_answer"]})
    k = sum(r["qwen_correct"] for r in rows)
    paired = Counter((r["gpt_correct"], r["qwen_correct"]) for r in rows)
    return {"completed": len(rows), "missing_task_ids": sorted(set(range(1,66))-set(r["task_id"] for r in rows)),
            "correct": k, "accuracy": k/len(rows) if rows else None,
            "wilson95": wilson(k,len(rows)) if rows else None,
            "gpt_full_correct": 57, "gpt_full_accuracy": 57/65, "gpt_full_wilson95": wilson(57,65),
            "paired": {"both_correct": paired[(True,True)], "gpt_only": paired[(True,False)],
                       "qwen_only": paired[(False,True)], "both_wrong": paired[(False,False)]},
            "task_rows": rows}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate", "smoke", "full", "report"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{PORT}/v1")
    parser.add_argument("--max-concurrent-tasks", type=int, default=4)
    args = parser.parse_args()
    assert args.max_concurrent_tasks >= 1
    static = validate_static(args.model_dir.resolve())
    if args.phase == "validate":
        print(json.dumps({k: v for k, v in static.items() if k != "baseline_hashes"}, indent=2))
        return 0
    if args.phase == "report":
        print(json.dumps(report(), ensure_ascii=False, indent=2))
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    static_path = OUTPUT / "static_validation.json"
    if static_path.exists():
        assert json.loads(static_path.read_text(encoding="utf-8")) == static
    else:
        atomic_json(static_path, static)
    server = await check_server(args.base_url)
    tokenizer = load_local_tokenizer(args.model_dir)
    print(f"SERVER_CHECK_PASS {server}; max_concurrent_tasks={args.max_concurrent_tasks}", flush=True)
    started = time.monotonic()
    gpu_observations: list[int] = []
    monitor_stop = asyncio.Event()
    monitor = asyncio.create_task(monitor_gpu(1, monitor_stop, gpu_observations))
    try:
        if args.phase == "smoke":
            rows = await phase_run([1,2,3], args.max_concurrent_tasks, tokenizer, static, args.base_url)
            assert all(len(x["run"]["rounds"]) == ROUNDS for x in rows)
            assert all(x["max_input_tokens"] + MAX_OUTPUT <= MODEL_LEN for x in rows)
            assert all(c.get("finish_reason") != "length" and "<think>" not in c.get("raw_response", "") for x in rows for c in x["completion_diagnostics"])
            atomic_json(OUTPUT / "smoke_pass.json", {"task_ids": [1,2,3], "model": MODEL, "revision": REVISION,
                        "prompt_sha256": static["prompt_sha256"], "max_concurrent_tasks": args.max_concurrent_tasks,
                        "elapsed_seconds": time.monotonic()-started,
                        "peak_gpu_1_memory_used_mib": max(gpu_observations) if gpu_observations else None})
            print("SMOKE_PASS: tasks 1-3 complete; full run remains stopped", flush=True)
            return 0
        gate = OUTPUT / "smoke_pass.json"
        assert gate.exists(), "Smoke gate missing; run --phase smoke first"
        g = json.loads(gate.read_text(encoding="utf-8"))
        assert g["model"] == MODEL and g["revision"] == REVISION and g["prompt_sha256"] == static["prompt_sha256"]
        await phase_run(list(range(1,66)), args.max_concurrent_tasks, tokenizer, static, args.base_url)
        assert baseline_hashes() == static["baseline_hashes"], "Original V2 inputs changed"
        result = report()
        assert result["completed"] == 65
        result.update(model=MODEL, revision=REVISION, max_concurrent_tasks=args.max_concurrent_tasks,
                      elapsed_seconds=time.monotonic()-started,
                      peak_gpu_1_memory_used_mib=max(gpu_observations) if gpu_observations else None,
                      gpu_1_memory_used_mib_at_end=gpu_used_mib(1))
        atomic_json(OUTPUT / "summary.json", result)
        print(f"FULL_COMPLETE: {result['correct']}/65, Wilson95={result['wilson95']}", flush=True)
        return 0
    finally:
        monitor_stop.set()
        await monitor


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
