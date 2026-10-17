"""HiddenBench: collective reasoning under distributed information."""

from hiddenbench.benchmark import BenchmarkTask, load_benchmark
from hiddenbench.metrics import score_results
from hiddenbench.models import ModelClient, OpenAICompatibleClient, build_model_client
from hiddenbench.simulator import Profile, run_evaluation, run_scenario

__all__ = [
    "BenchmarkTask",
    "ModelClient",
    "OpenAICompatibleClient",
    "Profile",
    "build_model_client",
    "load_benchmark",
    "run_evaluation",
    "run_scenario",
    "score_results",
]

__version__ = "0.1.0"
