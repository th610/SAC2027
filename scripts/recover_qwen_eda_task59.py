#!/usr/bin/env python3
"""Recover Qwen EDA Task 59 with an auditable exact Unicode-mojibake repair."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import run_paper_structured_protocol_qwen3_bf16 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paper_structured_protocol_qwen3_8b_bf16_hidden_20261017_task59_recovery"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


class ExactEncodingRepairClient(core.UsageClient):
    """Only reverse UTF-8 bytes misread as Latin-1 when that yields an allowed vote."""

    def __init__(self):
        super().__init__()
        self.normalizations = []

    def chat_json(self, messages, schema):
        parsed = super().chat_json(messages, schema)
        value = str(parsed.get("vote", ""))
        allowed = schema.get("properties", {}).get("vote", {}).get("enum", [])
        if value in allowed:
            return parsed
        try:
            repaired = value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return parsed
        if repaired not in allowed:
            return parsed
        event = {
            "logical_request": self.logical_requests,
            "raw_vote_value": value,
            "repaired_vote_value": repaired,
            "repair": "latin-1 encode followed by UTF-8 decode",
            "accepted_only_because_repaired_value_exactly_matches_allowed_option": True,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.normalizations.append(event)
        parsed = dict(parsed)
        parsed["vote"] = repaired
        return parsed

    def close(self):
        self.client.close()


def main() -> int:
    validation = core.static_validate()
    destination = OUT / "task_59.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite recovery result: {destination}")
    client = ExactEncodingRepairClient()
    started = datetime.now(timezone.utc)
    try:
        print("Re-running Qwen EDA Task 59 with original prompt/model/sampling settings", flush=True)
        run = core.run_task(59, client, json.loads(core.PROMPT_PATH.read_text(encoding="utf-8")), OUT, seed=42)
        n = run["num_agents"]
        if run["generation_call_count"] != n * 5:
            raise RuntimeError(f"Expected 5N={n*5} logical calls, got {run['generation_call_count']}")
        if any(v["vote"] not in run["possible_answers"] for v in run["initial_votes"] + run["final_votes"]):
            raise RuntimeError("Recovered run still has a vote outside allowed options")
        if not client.normalizations:
            raise RuntimeError("Task 59 did not exercise the expected encoding repair; refusing special recovery output")
        run["vote_encoding_normalizations"] = client.normalizations
        run["normalization_count"] = len(client.normalizations)
        meta = {
            "task_id": 59,
            "condition": "Paper-based Structured Protocol Adaptation",
            "model": core.MODEL,
            "profile": "hidden",
            "seed": 42,
            "temperature": core.TEMPERATURE,
            "top_p": core.TOP_P,
            "max_tokens": core.MAX_TOKENS,
            "enable_thinking": core.ENABLE_THINKING,
            "revision": core.REVISION,
            "precision": "bfloat16",
            "quantization": None,
            "endpoint": core.BASE_URL,
            "exchanges": 2,
            "decide_passes": 1,
            "protocol_prompt_file": str(core.PROMPT_PATH.relative_to(ROOT)),
            "protocol_prompt_sha256": core.sha(core.PROMPT_PATH),
            "v1_backbone_prompt_file": "src/hiddenbench/prompts.json",
            "v1_prompt_sha256": core.sha(ROOT / "src/hiddenbench/prompts.json"),
            "runner_sha256": core.sha(Path(core.__file__).resolve()),
            "recovery_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fact_assignment_seed": 42,
            "vote_encoding_repair": "Exact Latin-1-to-UTF-8 reversal only when the repaired value equals an allowed option; raw output retained in api_attempts.",
            "static_validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        artifact = {
            "metadata": meta,
            "run": run,
            "generation_call_count": run["generation_call_count"],
            "expected_generation_calls": n * 5,
            "api_attempt_count": run["api_attempt_count"],
            "retry_count": run["api_attempt_count"] - run["generation_call_count"],
            "token_usage": run["token_usage"],
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(destination, artifact)
        summary = {
            "task_id": 59,
            "status": "complete_with_exact_unicode_repair",
            "majority_vote": run["majority_vote"],
            "gold": run["correct_answer"],
            "correct": run["final_correct"],
            "generation_calls": run["generation_call_count"],
            "api_attempts": run["api_attempt_count"],
            "normalization_count": len(client.normalizations),
        }
        atomic_json(OUT / "recovery_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        atomic_json(OUT / "recovery_error.json", {
            "task_id": 59, "error_type": type(exc).__name__, "error": str(exc),
            "api_attempts": client.attempts, "normalizations": client.normalizations,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"RECOVERY FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
