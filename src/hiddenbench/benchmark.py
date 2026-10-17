from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal


DEFAULT_HF_REPO = "YuxuanLi1225/HiddenBench"
DEFAULT_BENCHMARK_FILENAME = "benchmark.json"


@dataclass(frozen=True)
class BenchmarkTask:
    """A single HiddenBench decision task."""

    id: int | str
    name: str
    description: str
    shared_information: list[str]
    hidden_information: list[str]
    possible_answers: list[str]
    correct_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "BenchmarkTask":
        validate_task_mapping(item)
        known = {
            "id",
            "name",
            "description",
            "shared_information",
            "hidden_information",
            "possible_answers",
            "correct_answer",
        }
        metadata = {key: value for key, value in item.items() if key not in known}
        return cls(
            id=item["id"],
            name=item["name"],
            description=item["description"],
            shared_information=list(item["shared_information"]),
            hidden_information=list(item["hidden_information"]),
            possible_answers=list(item["possible_answers"]),
            correct_answer=item["correct_answer"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "shared_information": list(self.shared_information),
            "hidden_information": list(self.hidden_information),
            "possible_answers": list(self.possible_answers),
            "correct_answer": self.correct_answer,
        }
        data.update(self.metadata)
        return data


def validate_task_mapping(item: dict[str, Any]) -> None:
    required = {
        "id",
        "name",
        "description",
        "shared_information",
        "hidden_information",
        "possible_answers",
        "correct_answer",
    }
    missing = required - set(item)
    if missing:
        raise ValueError(f"Benchmark task is missing required fields: {sorted(missing)}")

    if not isinstance(item["name"], str) or not item["name"].strip():
        raise ValueError("Task name must be a non-empty string")
    if not isinstance(item["description"], str) or not item["description"].strip():
        raise ValueError(f"Task {item['name']!r} has an empty description")

    for field_name in ("shared_information", "hidden_information", "possible_answers"):
        value = item[field_name]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Task {item['name']!r} field {field_name!r} must be a non-empty list")
        if not all(isinstance(entry, str) and entry.strip() for entry in value):
            raise ValueError(f"Task {item['name']!r} field {field_name!r} must contain non-empty strings")

    if len(set(item["possible_answers"])) != len(item["possible_answers"]):
        raise ValueError(f"Task {item['name']!r} has duplicate possible answers")
    if item["correct_answer"] not in item["possible_answers"]:
        raise ValueError(f"Task {item['name']!r} correct_answer is not in possible_answers")


def validate_benchmark(items: Iterable[dict[str, Any]]) -> list[BenchmarkTask]:
    tasks = [BenchmarkTask.from_mapping(item) for item in items]
    seen_ids: set[int | str] = set()
    for task in tasks:
        if task.id in seen_ids:
            raise ValueError(f"Duplicate task id: {task.id!r}")
        seen_ids.add(task.id)
    return tasks


def default_data_dir() -> Path:
    bundled = Path(__file__).resolve().parent / "data"
    if bundled.exists():
        return bundled
    package_root = Path(__file__).resolve().parents[2]
    return package_root / "data"


def load_json_benchmark(path: str | Path | None = None) -> list[BenchmarkTask]:
    benchmark_path = Path(path) if path is not None else default_data_dir() / DEFAULT_BENCHMARK_FILENAME
    with benchmark_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Benchmark file must contain a JSON list: {benchmark_path}")
    return validate_benchmark(raw)


def load_hf_benchmark(repo: str = DEFAULT_HF_REPO, split: str = "train") -> list[BenchmarkTask]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the Hugging Face extra with `pip install hiddenbench[hf]`.") from exc

    dataset = load_dataset(repo, split=split)
    rows = [dict(row) for row in dataset]
    return validate_benchmark(rows)


def load_benchmark(
    source: Literal["local", "hf"] = "local",
    path: str | Path | None = None,
    hf_repo: str = DEFAULT_HF_REPO,
    split: str = "train",
) -> list[BenchmarkTask]:
    """Load HiddenBench tasks from local JSON or Hugging Face."""

    if source == "local":
        return load_json_benchmark(path)
    if source == "hf":
        return load_hf_benchmark(hf_repo, split=split)
    raise ValueError("source must be either 'local' or 'hf'")


def find_task(tasks: list[BenchmarkTask], task_id: int | str | None = None, name: str | None = None) -> BenchmarkTask:
    if task_id is None and name is None:
        raise ValueError("Provide task_id or name")
    for task in tasks:
        if task_id is not None and str(task.id) == str(task_id):
            return task
        if name is not None and task.name == name:
            return task
    label = f"id={task_id}" if task_id is not None else f"name={name}"
    raise KeyError(f"No benchmark task found for {label}")
