#!/usr/bin/env python3
"""Fresh Gemini 3.5 Flash-Lite evaluation of the Paper-based EDA protocol."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, os, sys, threading, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_paper_structured_protocol as core

MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
THINKING_LEVEL = "minimal"
MAX_OUTPUT_TOKENS = 1024
MAX_ATTEMPTS = 5
MIN_REQUEST_INTERVAL = 0.5
SMOKE_DEFAULT = ROOT / "runs/paper_structured_protocol_gemini_3_5_flash_lite_hidden_20261017_smoke"
FULL_DEFAULT = ROOT / "runs/paper_structured_protocol_gemini_3_5_flash_lite_hidden_20261017"
DOTENV = Path(".env")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def gemini_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [gemini_schema(x) for x in schema]
    return schema


class GeminiEDAClient:
    model_name = MODEL

    def __init__(self, api_key: str):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.attempts: list[dict[str, Any]] = []
        self.logical_requests = 0
        self._rate_lock = threading.Lock()
        self._rate_state = {"last": 0.0}

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for name in ("code", "status_code"):
            value = getattr(exc, name, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        return value if isinstance(value, int) else None

    def _request(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        self.logical_requests += 1
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for message in messages:
            if message["role"] == "system":
                continue
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(self.types.Content(role=role, parts=[self.types.Part.from_text(text=message["content"])]))
        config_args: dict[str, Any] = {
            "system_instruction": system or None,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_config": self.types.ThinkingConfig(thinking_level="MINIMAL"),
        }
        if schema is not None:
            config_args.update(response_mime_type="application/json", response_schema=gemini_schema(schema))
        config = self.types.GenerateContentConfig(**config_args)
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record: dict[str, Any] = {
                "attempt": attempt, "model_requested": MODEL, "temperature": TEMPERATURE,
                "thinking_level": THINKING_LEVEL, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "structured_json": schema is not None, "messages": messages,
                "mapped_system_instruction": system, "started_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                with self._rate_lock:
                    delay = MIN_REQUEST_INTERVAL - (time.monotonic() - self._rate_state["last"])
                    if delay > 0:
                        time.sleep(delay)
                    self._rate_state["last"] = time.monotonic()
                response = self.client.models.generate_content(model=MODEL, contents=contents, config=config)
                candidate = response.candidates[0] if response.candidates else None
                parts = candidate.content.parts if candidate and candidate.content else []
                public_text = "\n".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()
                usage = getattr(response, "usage_metadata", None)
                finish = str(candidate.finish_reason) if candidate else "NO_CANDIDATE"
                record.update({
                    "response_id": getattr(response, "response_id", None),
                    "response_model_version": getattr(response, "model_version", None),
                    "raw_response": public_text, "finish_reason": finish,
                    "thought_parts_filtered": sum(bool(getattr(p, "thought", False)) for p in parts),
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                        "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                        "thoughts_tokens": getattr(usage, "thoughts_token_count", None) if usage else None,
                        "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
                    } if usage else None,
                })
                if not public_text:
                    raise ValueError("Gemini returned no non-thinking public text")
                if candidate and str(candidate.finish_reason).endswith("MAX_TOKENS"):
                    raise ValueError("Gemini output hit max_output_tokens")
                record.update(status="ok", completed_at=datetime.now(timezone.utc).isoformat())
                self.attempts.append(record)
                return public_text
            except Exception as exc:
                last_error = exc
                code = self._status_code(exc)
                record.update(status="error", error_type=type(exc).__name__, error=str(exc), http_code=code,
                              completed_at=datetime.now(timezone.utc).isoformat())
                self.attempts.append(record)
                retryable = code == 429 or (code is not None and code >= 500)
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(min(2 ** (attempt - 1), 16))
        assert last_error is not None
        raise last_error

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._request(messages)

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        from hiddenbench.models import extract_json_object
        parsed = extract_json_object(self._request(messages, schema))
        allowed = schema["properties"]["vote"]["enum"]
        if parsed.get("vote") not in allowed or not str(parsed.get("rationale", "")).strip():
            raise ValueError("Gemini JSON failed local vote/rationale validation")
        return parsed

    def close(self) -> None:
        self.client.close()


def frozen_manifest(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": "Paper-based Structured Protocol Adaptation (EDA)",
        "model": MODEL, "api": "Google Gemini Developer API", "sdk": "google-genai",
        "sdk_version": importlib.metadata.version("google-genai"),
        "temperature": TEMPERATURE, "thinking_level": THINKING_LEVEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "seed": 42, "profile": "hidden",
        "exchanges": 2, "decide_passes": 1, "fresh_all_task_calls": True,
        "protocol_prompt_sha256": sha(core.PROMPT_PATH),
        "v1_backbone_sha256": sha(ROOT / "src/hiddenbench/prompts.json"),
        "dataset_sha256": sha(ROOT / "data/benchmark.json"),
        "runner_sha256": sha(Path(__file__).resolve()),
        "static_validation": validation,
    }


def valid_cached(path: Path, task_id: int, manifest: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("metadata") != manifest or saved.get("task_id") != task_id:
        raise ValueError(f"Cache manifest mismatch, refusing reuse: {path}")
    run = saved.get("run", {})
    n = run.get("num_agents")
    if not isinstance(n, int) or saved.get("generation_call_count") != 5 * n:
        raise ValueError(f"Incomplete call count in cache: {path}")
    if len(run.get("initial_votes", [])) != n or len(run.get("final_votes", [])) != n:
        raise ValueError(f"Incomplete ballots in cache: {path}")
    if len(run.get("exchanges", [])) != 2 or any(len(s.get("messages", [])) != n for s in run["exchanges"]):
        raise ValueError(f"Incomplete exchanges in cache: {path}")
    if len(run.get("decide_messages", [])) != n or len(run.get("discussion_transcript", [])) != 3 * n:
        raise ValueError(f"Incomplete discussion in cache: {path}")
    if run.get("majority_vote") != Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0]:
        raise ValueError(f"Tie-break/majority mismatch in cache: {path}")
    return saved


def run_phase(phase: str, out: Path, smoke_dir: Path, dotenv: Path) -> int:
    validation = core.static_validate()
    if phase == "static":
        print(json.dumps(frozen_manifest(validation), ensure_ascii=False, indent=2))
        return 0
    load_dotenv(dotenv)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing in configured environment/dotenv")
    if phase == "full":
        gate_path = smoke_dir / "smoke_pass.json"
        if not gate_path.is_file():
            raise RuntimeError(f"Smoke gate missing: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("protocol_prompt_sha256") != sha(core.PROMPT_PATH) or gate.get("model") != MODEL:
            raise RuntimeError("Smoke gate does not match protocol/model")
    out.mkdir(parents=True, exist_ok=True)
    manifest = frozen_manifest(validation)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Manifest differs; refusing to continue in {out}")
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    if phase == "smoke":
        smoke_manifest_path = out / "smoke_manifest.json"
        atomic_json(smoke_manifest_path, manifest)
    ids = [1, 2] if phase == "smoke" else list(range(1, 66))
    tasks = []
    for task_id in ids:
        dest = out / f"task_{task_id}.json"
        cached = valid_cached(dest, task_id, manifest) if phase == "full" else None
        if cached:
            saved = cached
            print(f"Task {task_id}: verified cache", flush=True)
        else:
            client = GeminiEDAClient(api_key)
            started = time.monotonic()
            try:
                print(f"Task {task_id}/{len(ids)} started", flush=True)
                run = core.run_task(task_id, client, json.loads(core.PROMPT_PATH.read_text(encoding="utf-8")), out, seed=42)
                n = run["num_agents"]
                expected = n * 5
                if client.logical_requests != expected or run["generation_call_count"] != expected:
                    raise RuntimeError(f"Expected 5N={expected} calls, got logical={client.logical_requests}, run={run['generation_call_count']}")
                if run["majority_vote"] != Counter(v["vote"] for v in run["final_votes"]).most_common(1)[0][0]:
                    raise RuntimeError("Majority vote/tie-break validation failed")
                if any(v["vote"] not in run["possible_answers"] for v in run["initial_votes"] + run["final_votes"]):
                    raise RuntimeError("Parsed vote outside answer options")
                # The core function stores Gemini attempts under this adapter, with normalized usage keys.
                attempts = run["api_attempts"]
                if sum(a.get("status") == "ok" for a in attempts) < expected:
                    raise RuntimeError("Fewer than 5N successful API attempts")
                saved = {
                    "task_id": task_id, "metadata": manifest, "run": run,
                    "generation_call_count": client.logical_requests,
                    "expected_generation_calls": expected,
                    "api_attempt_count": len(attempts),
                    "retry_count": len(attempts) - client.logical_requests,
                    "token_usage": run["token_usage"],
                    "elapsed_seconds": time.monotonic() - started,
                    "response_ids": [a.get("response_id") for a in attempts if a.get("status") == "ok"],
                }
                if dest.exists():
                    raise FileExistsError(f"Refusing overwrite: {dest}")
                atomic_json(dest, saved)
                print(f"Task {task_id}: majority={run['majority_vote']} correct={run['final_correct']} calls={client.logical_requests} attempts={len(attempts)} tokens={run['token_usage']['total_tokens']}", flush=True)
            except Exception as exc:
                error = {"task_id": task_id, "model": MODEL, "error_type": type(exc).__name__,
                         "error": str(exc), "api_attempts": client.attempts,
                         "created_at": datetime.now(timezone.utc).isoformat()}
                atomic_json(out / f"task_{task_id}.error.json", error)
                tasks.append({"task_id": task_id, "status": "failed", "error": str(exc)})
                print(f"Task {task_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
                client.close()
                continue
            client.close()
        run = saved["run"]
        tasks.append({"task_id": task_id, "status": "complete", "correct": run["majority_vote"] == run["correct_answer"],
                      "initial_correct": Counter(v["vote"] for v in run["initial_votes"]).most_common(1)[0][0] == run["correct_answer"],
                      "majority_vote": run["majority_vote"], "gold": run["correct_answer"],
                      "calls": saved["generation_call_count"], "tokens": saved["token_usage"]})
    completed = [x for x in tasks if x["status"] == "complete"]
    summary = {"condition": "Paper-based Structured Protocol Adaptation (EDA)", "model": MODEL,
               "phase": phase, "task_count": len(ids), "completed": len(completed),
               "failed": len(tasks) - len(completed), "failed_task_ids": [x["task_id"] for x in tasks if x["status"] == "failed"],
               "correct": sum(x["correct"] for x in completed), "accuracy": (sum(x["correct"] for x in completed) / len(completed)) if completed else None,
               "initial_correct": sum(x["initial_correct"] for x in completed),
               "total_logical_calls": sum(x["calls"] for x in completed), "tasks": tasks,
               "created_at": datetime.now(timezone.utc).isoformat(), "manifest": manifest}
    atomic_json(out / "summary.json", summary)
    if phase == "smoke":
        if len(completed) != 2 or summary["failed"]:
            raise RuntimeError("2-task smoke failed; full run is blocked")
        atomic_json(out / "smoke_pass.json", {"status": "PASS", "task_ids": [1, 2], "model": MODEL,
                                               "protocol_prompt_sha256": sha(core.PROMPT_PATH),
                                               "calls": summary["total_logical_calls"], "fresh_full_output": True})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tasks", "manifest")}, ensure_ascii=False, indent=2), flush=True)
    return int(summary["failed"] > 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("static", "smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-dir", type=Path, default=SMOKE_DEFAULT)
    parser.add_argument("--dotenv", type=Path, default=DOTENV)
    args = parser.parse_args()
    out = args.smoke_dir.resolve() if args.phase == "smoke" else (args.output_dir or FULL_DEFAULT).resolve()
    return run_phase(args.phase, out, args.smoke_dir.resolve(), args.dotenv)


if __name__ == "__main__":
    raise SystemExit(main())
