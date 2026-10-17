"""Run unchanged HiddenBench V2 through Gemini GenerateContent API."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.models import extract_json_object
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, format_discussion_transcript, instantiate_agents, run_scenario, vote_json_schema

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "src/hiddenbench/prompts_v2.json"
BASELINE = ROOT / "runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015"
OUTPUT = ROOT / "runs/v2_gemini_3_5_flash_lite_hidden_20261016"
MODEL = "gemini-3.5-flash-lite"
SEED = 42
ROUNDS = 3
TEMPERATURE = 0.0
THINKING_LEVEL = "minimal"
MAX_OUTPUT_TOKENS = 1024
MAX_CONCURRENT_TASKS = 2
MIN_REQUEST_INTERVAL = 0.5  # target no more than 2 request starts/second across tasks
INPUT_USD_PER_M = 0.30
OUTPUT_USD_PER_M = 2.50


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def get_baseline(task_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    saved = json.loads((BASELINE / f"task_{task_id}.json").read_text(encoding="utf-8"))
    return saved, saved["runs"][0]


def baseline_hashes() -> dict[str, str]:
    paths = [PROMPT, BASELINE / "summary.json"] + [BASELINE / f"task_{i}.json" for i in range(1, 66)]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def static_validation() -> dict[str, Any]:
    prompt_set = load_prompts(version="v2")
    benchmark = load_benchmark(path=str(ROOT / "data/benchmark.json"))
    summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] == 65 and summary["failed"] == 0
    assert sum(row["majority_correct"] for row in summary["tasks"]) == 57
    checked = Counter()
    for task_id in range(1, 66):
        task = find_task(benchmark, task_id=str(task_id))
        _, old = get_baseline(task_id)
        assert old["task_id"] == task.id and old["correct_answer"] == task.correct_answer
        assert old["possible_answers"] == task.possible_answers
        agents, assignment = instantiate_agents(task, None, Profile.HIDDEN, prompt_set, random.Random(SEED))
        assert assignment == old["fact_assignments"]
        assert [a.history[0]["content"] for a in agents] == [x["prompt"] for x in old["prompt_diagnostics"]["system_prompts"]]
        checked["facts_and_system"] += 1
        initial = render_prompt(prompt_set.first_vote_prompt, {"possible_answers": ", ".join(task.possible_answers)})
        assert initial == old["prompt_diagnostics"]["initial_vote_prompt"]
        checked["initial_prompt"] += 1
        iv = old["initial_votes"][0]
        first = f"Your initial vote was: {iv['vote']}\nYour initial rationale was: {iv['rationale']}\n\n" + prompt_set.first_user_prompt
        assert first == old["rounds"][0]["messages"][0]["prompt"]
        checked["r1_first_prompt"] += 1
        final = render_prompt(prompt_set.vote_prompt, {
            "group_discussion": format_discussion_transcript(old["rounds"][-1]["messages"]),
            "possible_answers": ", ".join(task.possible_answers)})
        assert final == old["prompt_diagnostics"]["final_vote_prompt"]
        checked["final_prompt"] += 1
        assert len(old["rounds"]) == ROUNDS and len(agents) == len(old["initial_votes"])
    return {"status": "PASS", "tasks": 65, "baseline_correct": 57,
            "matched_records": dict(checked), "model": MODEL, "thinking_level": THINKING_LEVEL,
            "temperature": TEMPERATURE, "prompt_sha256": digest(PROMPT),
            "baseline_hashes": baseline_hashes()}


class GeminiV2Client:
    """Maps simulator chat history to Gemini roles while filtering thought parts."""

    model_name = MODEL

    def __init__(self, api_key: str, *, rate_lock: threading.Lock, rate_state: dict[str, float]):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.calls: list[dict[str, Any]] = []
        self.rate_lock = rate_lock
        self.rate_state = rate_state

    @staticmethod
    def _error_code(exc: Exception) -> int | None:
        for key in ("code", "status_code"):
            val = getattr(exc, key, None)
            if isinstance(val, int):
                return val
        response = getattr(exc, "response", None)
        val = getattr(response, "status_code", None)
        return val if isinstance(val, int) else None

    @staticmethod
    def _normalize(messages: list[dict[str, str]]) -> tuple[str, list[Any]]:
        from google.genai import types
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        return system, contents

    def _generate(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        system, contents = self._normalize(messages)
        config_args: dict[str, Any] = {
            "system_instruction": system or None,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_config": self.types.ThinkingConfig(thinking_level="MINIMAL"),
        }
        if schema is not None:
            config_args.update(response_mime_type="application/json", response_schema=gemini_compatible_schema(schema))
        config = self.types.GenerateContentConfig(**config_args)
        last_error: Exception | None = None
        for attempt in range(5):
            rec: dict[str, Any] = {"model": MODEL, "attempt": attempt + 1, "temperature": TEMPERATURE,
                                   "thinking_level": THINKING_LEVEL, "max_output_tokens": MAX_OUTPUT_TOKENS,
                                   "structured_json": schema is not None,
                                   "mapped_system_instruction": system, "mapped_contents": messages}
            try:
                with self.rate_lock:
                    delay = MIN_REQUEST_INTERVAL - (time.monotonic() - self.rate_state.get("last", 0.0))
                    if delay > 0:
                        time.sleep(delay)
                    self.rate_state["last"] = time.monotonic()
                response = self.client.models.generate_content(model=MODEL, contents=contents, config=config)
                candidate = response.candidates[0] if response.candidates else None
                parts = candidate.content.parts if candidate and candidate.content else []
                public_text = "\n".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()
                usage = response.usage_metadata
                finish = str(candidate.finish_reason) if candidate else "NO_CANDIDATE"
                thought_count = getattr(usage, "thoughts_token_count", None) if usage else None
                rec.update(
                    response_text=public_text,
                    finish_reason=finish,
                    prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
                    candidate_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
                    thoughts_tokens=thought_count,
                    total_tokens=getattr(usage, "total_token_count", None) if usage else None,
                    thought_parts_filtered=sum(bool(getattr(p, "thought", False)) for p in parts),
                )
                self.calls.append(rec)
                if not public_text:
                    raise ValueError("Gemini returned no non-thinking public text")
                if candidate and str(candidate.finish_reason).endswith("MAX_TOKENS"):
                    raise ValueError("Gemini output hit max_output_tokens")
                return public_text
            except Exception as exc:
                last_error = exc
                code = self._error_code(exc)
                rec.update(error_type=type(exc).__name__, error=str(exc), http_code=code)
                if not self.calls or self.calls[-1] is not rec:
                    self.calls.append(rec)
                if code != 429 and (code is None or code < 500):
                    raise
                if attempt == 4:
                    break
                time.sleep(min(2 ** attempt, 16))
        assert last_error is not None
        raise last_error


    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._generate(messages)

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        output = self._generate(messages, schema)
        parsed = extract_json_object(output)
        allowed = schema["properties"]["vote"]["enum"]
        if parsed.get("vote") not in allowed or not str(parsed.get("rationale", "")).strip():
            raise ValueError("Gemini JSON did not satisfy vote/rationale schema")
        return parsed

    def close(self) -> None:
        self.client.close()


def gemini_compatible_schema(schema: Any) -> Any:
    """Remove JSON Schema keywords unsupported by Gemini while preserving validation locally."""
    if isinstance(schema, dict):
        return {key: gemini_compatible_schema(value) for key, value in schema.items()
                if key != "additionalProperties"}
    if isinstance(schema, list):
        return [gemini_compatible_schema(value) for value in schema]
    return schema


def validate_run(run: dict[str, Any], old: dict[str, Any]) -> None:
    assert run["task_id"] == old["task_id"] and run["correct_answer"] == old["correct_answer"]
    assert run["possible_answers"] == old["possible_answers"] and run["fact_assignments"] == old["fact_assignments"]
    assert len(run["rounds"]) == ROUNDS and len(run["final_votes"]) == len(old["final_votes"])
    assert [[m["agent"] for m in r["messages"]] for r in run["rounds"]] == [[m["agent"] for m in r["messages"]] for r in old["rounds"]]
    assert all(v["vote"] in run["possible_answers"] and v["rationale"] for v in run["initial_votes"] + run["final_votes"])
    assert Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0] == run["majority_vote"]
    assert all(m["response"].strip() for r in run["rounds"] for m in r["messages"])


async def run_one(task_id: int, api_key: str, rate_lock: threading.Lock, rate_state: dict[str, float]) -> dict[str, Any]:
    path = OUTPUT / f"task_{task_id}.json"
    _, old = get_baseline(task_id)
    p_hash = digest(PROMPT)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["metadata"]["prompt_sha256"] == p_hash and saved["metadata"]["model"] == MODEL
        validate_run(saved["run"], old)
        return saved
    task = find_task(load_benchmark(path=str(ROOT / "data/benchmark.json")), task_id=str(task_id))
    gemini = GeminiV2Client(api_key, rate_lock=rate_lock, rate_state=rate_state)
    started = time.monotonic()
    try:
        run = await asyncio.to_thread(run_scenario, task, gemini, Profile.HIDDEN,
                                      load_prompts(version="v2"), rounds=ROUNDS, seed=SEED)
        validate_run(run, old)
        expected = len(old["initial_votes"]) * (ROUNDS + 2)
        usage = {k: sum(c.get(k) or 0 for c in gemini.calls) for k in ("prompt_tokens", "candidate_tokens", "thoughts_tokens", "total_tokens")}
        # Gemini output pricing includes thinking tokens; count candidate + thoughts for billed output estimate.
        usage["billed_output_tokens_estimate"] = usage["candidate_tokens"] + usage["thoughts_tokens"]
        cost = usage["prompt_tokens"] * INPUT_USD_PER_M / 1e6 + usage["billed_output_tokens_estimate"] * OUTPUT_USD_PER_M / 1e6
        saved = {
            "metadata": {"model": MODEL, "api": "Google Gemini Developer API", "sdk": "google-genai",
                         "sdk_version": __import__("importlib.metadata", fromlist=["version"]).version("google-genai"),
                         "prompt_file": str(PROMPT), "prompt_sha256": p_hash,
                         "profile": "hidden", "rounds": ROUNDS, "seed": SEED,
                         "thinking_level": THINKING_LEVEL, "temperature": TEMPERATURE,
                         "max_output_tokens": MAX_OUTPUT_TOKENS, "created_at": datetime.now(timezone.utc).isoformat()},
            "generation_call_count": len(gemini.calls), "expected_generation_calls": expected,
            "retry_attempt_count": sum(max(0, c["attempt"] - 1) for c in gemini.calls),
            "token_usage": usage, "estimated_list_price_usd": cost,
            "elapsed_seconds": time.monotonic() - started,
            "call_diagnostics": gemini.calls,
            "run": run,
        }
        atomic_json(path, saved)
        print(f"task {task_id}/65 complete | majority={run['majority_vote']} | correct={run['majority_vote']==run['correct_answer']} | calls={len(gemini.calls)} | tokens={usage['total_tokens']} | est_usd=${cost:.5f}", flush=True)
        return saved
    except Exception as exc:
        atomic_json(OUTPUT / f"task_{task_id}.error.json", {"task_id": task_id, "model": MODEL,
                    "error_type": type(exc).__name__, "error": str(exc), "call_diagnostics": gemini.calls,
                    "elapsed_seconds": time.monotonic() - started})
        print(f"task {task_id}/65 FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        gemini.close()


async def bounded_run(task_ids: list[int], api_key: str, concurrency: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    rate_lock = threading.Lock()
    rate_state: dict[str, float] = {"last": 0.0}
    results: list[dict[str, Any]] = []
    async def worker(task_id: int) -> None:
        async with semaphore:
            results.append(await run_one(task_id, api_key, rate_lock, rate_state))
    await asyncio.gather(*(worker(i) for i in task_ids))
    return results


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h,c+h]


def summarize() -> dict[str, Any]:
    rows=[]
    for i in range(1,66):
        p=OUTPUT/f"task_{i}.json"
        if not p.exists(): continue
        x=json.loads(p.read_text(encoding="utf-8")); r=x["run"]
        _,old=get_baseline(i); validate_run(r,old)
        rows.append({"task_id":i,"gold":r["correct_answer"],"gemini_majority":r["majority_vote"],
                     "gemini_correct":r["majority_vote"]==r["correct_answer"],
                     "gpt_majority":old["majority_vote"],"gpt_correct":old["majority_vote"]==old["correct_answer"],
                     "initial_votes":[v["vote"] for v in r["initial_votes"]],
                     "initial_majority":Counter(v["vote"] for v in r["initial_votes"]).most_common(1)[0][0],
                     "initial_correct":Counter(v["vote"] for v in r["initial_votes"]).most_common(1)[0][0]==r["correct_answer"],
                     "final_votes":[v["vote"] for v in r["final_votes"]],
                     "calls":x["generation_call_count"],"tokens":x["token_usage"],"cost_usd":x["estimated_list_price_usd"]})
    k=sum(x["gemini_correct"] for x in rows); ki=sum(x["initial_correct"] for x in rows)
    pairs=Counter((x["gpt_correct"],x["gemini_correct"]) for x in rows)
    all_calls=sum(x["calls"] for x in rows); all_cost=sum(x["cost_usd"] for x in rows)
    all_usage={key:sum(x["tokens"].get(key,0) for x in rows) for key in ("prompt_tokens","candidate_tokens","thoughts_tokens","total_tokens","billed_output_tokens_estimate")}
    return {"model":MODEL,"completed":len(rows),"failed_task_ids":sorted(set(range(1,66))-set(x["task_id"] for x in rows)),
            "initial_correct":ki,"initial_accuracy":ki/len(rows) if rows else None,
            "initial_wilson95":wilson(ki,len(rows)) if rows else None,
            "final_correct":k,"final_accuracy":k/len(rows) if rows else None,
            "final_wilson95":wilson(k,len(rows)) if rows else None,
            "gpt_baseline_correct":57,"gpt_baseline_accuracy":57/65,"gpt_baseline_wilson95":wilson(57,65),
            "paired":{"both_correct":pairs[(True,True)],"gpt_only":pairs[(True,False)],"gemini_only":pairs[(False,True)],"both_wrong":pairs[(False,False)]},
            "total_generation_calls":all_calls,"total_api_retry_attempts":sum(max(0,json.loads((OUTPUT/f"task_{x['task_id']}.json").read_text())["retry_attempt_count"]) for x in rows),
            "total_token_usage":all_usage,"estimated_list_price_usd":all_cost,
            "task_rows":rows}


async def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("preflight","smoke","full","report"),required=True)
    parser.add_argument("--dotenv",type=Path,default=Path(".env"))
    parser.add_argument("--max-concurrent-tasks",type=int,default=MAX_CONCURRENT_TASKS)
    args=parser.parse_args()
    if args.phase=="report":
        print(json.dumps(summarize(),ensure_ascii=False,indent=2));return 0
    static=static_validation()
    if args.phase=="preflight":
        print(json.dumps({k:v for k,v in static.items() if k!="baseline_hashes"},indent=2));return 0
    load_dotenv(args.dotenv)
    api_key=os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY missing from environment or configured dotenv")
    from google import genai
    client=genai.Client(api_key=api_key)
    model_info=client.models.get(model=MODEL)
    model_record={"name":model_info.name,"display_name":getattr(model_info,"display_name",None),"supported_actions":getattr(model_info,"supported_actions",None)}
    client.close()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    static_path=OUTPUT/"static_validation.json"
    if static_path.exists(): assert json.loads(static_path.read_text())==static
    else: atomic_json(static_path,static)
    model_path=OUTPUT/"model_access.json"
    if model_path.exists(): assert json.loads(model_path.read_text())==model_record
    else: atomic_json(model_path,model_record)
    ids=[1] if args.phase=="smoke" else list(range(1,66))
    start=time.monotonic()
    results=await bounded_run(ids,api_key,args.max_concurrent_tasks)
    tokens={key:sum(x["token_usage"].get(key,0) for x in results) for key in ("prompt_tokens","candidate_tokens","thoughts_tokens","total_tokens","billed_output_tokens_estimate")}
    cost=sum(x["estimated_list_price_usd"] for x in results)
    total_calls=sum(x["generation_call_count"] for x in results)
    if args.phase=="smoke":
        full_calls=sum(len(get_baseline(i)[1]["initial_votes"])*(ROUNDS+2) for i in range(1,66))
        scale=full_calls/total_calls
        est_input=tokens["prompt_tokens"]*scale
        est_output=tokens["billed_output_tokens_estimate"]*scale
        estimate=est_input*INPUT_USD_PER_M/1e6+est_output*OUTPUT_USD_PER_M/1e6
        smoke={"task_id":1,"status":"PASS","calls":total_calls,"tokens":tokens,"actual_smoke_estimated_list_price_usd":cost,
               "full_expected_calls":full_calls,"simple_linear_full_token_estimate":{"input":est_input,"output_including_thoughts":est_output,"usd":estimate},
               "caveat":"Task 1 linear extrapolation is a rough budget estimate; per-task conversation lengths vary.",
               "elapsed_seconds":time.monotonic()-start,"model_info":model_record}
        atomic_json(OUTPUT/"smoke_cost_estimate.json",smoke)
        print(json.dumps(smoke,ensure_ascii=False,indent=2),flush=True)
        return 0
    result=summarize();result.update(model_info=model_record,max_concurrent_tasks=args.max_concurrent_tasks,elapsed_seconds=time.monotonic()-start)
    atomic_json(OUTPUT/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="task_rows"},ensure_ascii=False,indent=2),flush=True)
    return 0 if result["completed"]==65 else 1


if __name__=="__main__":
    raise SystemExit(asyncio.run(main()))
