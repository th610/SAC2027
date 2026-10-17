from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from hiddenbench.benchmark import BenchmarkTask
from hiddenbench.models import Message, ModelClient
from hiddenbench.prompts import PromptSet, load_prompts, render_prompt


class Profile(StrEnum):
    HIDDEN = "hidden"
    FULL = "full"


@dataclass
class AgentState:
    name: str
    model_client: ModelClient
    history: list[Message]
    is_special: bool = False

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        response = self.model_client.chat(self.history)
        self.history.append({"role": "assistant", "content": str(response)})
        return str(response)

    def vote(self, message: str, possible_answers: list[str], retries: int = 3) -> dict[str, str]:
        schema = vote_json_schema(possible_answers)
        local_history = list(self.history)
        local_history.append({"role": "user", "content": message})
        for attempt in range(retries):
            response = self.model_client.chat_json(local_history, schema)
            vote = str(response.get("vote", "")).strip()
            rationale = str(response.get("rationale", "")).strip()
            if vote in possible_answers and rationale:
                return {"vote": vote, "rationale": rationale}
            local_history.append({"role": "assistant", "content": str(response)})
            local_history.append(
                {
                    "role": "user",
                    "content": f"The vote must be exactly one of {possible_answers}. Return valid JSON again.",
                }
            )
        raise ValueError(f"Model returned invalid vote {vote!r}; expected one of {possible_answers}")


def vote_json_schema(possible_answers: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "vote": {"type": "string", "enum": possible_answers},
            "rationale": {"type": "string"},
        },
        "required": ["vote", "rationale"],
        "additionalProperties": False,
    }


def format_facts(facts: list[str]) -> str:
    return "".join(f"Fact {index}: {fact}\n" for index, fact in enumerate(facts, start=1))


def format_discussion_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "No discussion has occurred."
    return "\n".join(f"{entry['agent']}: {entry['response']}" for entry in messages)


def instantiate_agents(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    prompts: PromptSet,
    rng: random.Random,
    *,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
) -> tuple[list[AgentState], list[dict[str, Any]]]:
    profile = Profile(profile)
    hidden_info = list(task.hidden_information)
    rng.shuffle(hidden_info)
    num_agents = len(hidden_info)
    num_special_agents = min(num_agents, math.ceil(num_agents * special_agent_ratio))
    special_indices = set(rng.sample(range(num_agents), num_special_agents)) if num_special_agents else set()

    agents: list[AgentState] = []
    fact_assignments: list[dict[str, Any]] = []
    for index in range(num_agents):
        facts = list(task.shared_information)
        if profile == Profile.HIDDEN:
            facts.append(hidden_info[index])
            assigned_hidden = [hidden_info[index]]
        else:
            facts.extend(hidden_info)
            assigned_hidden = list(hidden_info)
        rng.shuffle(facts)
        system_prompt = render_prompt(
            load_prompts().system_prompt if profile == Profile.FULL and prompts.prompt_version == "v2"
            else prompts.system_prompt,
            {
                "description": task.description,
                "information": format_facts(facts),
                "shared_information": format_facts([f for f in facts if f in task.shared_information]),
                "private_information": format_facts(assigned_hidden),
                "num_agents": str(num_agents),
                "num_shared_facts": str(len(task.shared_information)),
                "extra": extra_prompt if index in special_indices else "",
            },
        )
        agent = AgentState(
            name=f"Person {index + 1}",
            model_client=model_client,
            history=[{"role": "system", "content": system_prompt}],
            is_special=index in special_indices,
        )
        agents.append(agent)
        fact_assignments.append(
            {
                "agent": agent.name,
                "is_special": agent.is_special,
                "visible_facts": list(facts),
                "assigned_hidden_information": assigned_hidden,
            }
        )
    return agents, fact_assignments


def collect_initial_votes(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    prompts: PromptSet | None = None,
    *,
    seed: int | None = None,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    prompt_set = prompts or load_prompts()
    rng = random.Random(seed)
    agents, assignments = instantiate_agents(
        task,
        model_client,
        profile,
        prompt_set,
        rng,
        extra_prompt=extra_prompt,
        special_agent_ratio=special_agent_ratio,
    )
    vote_prompt = render_prompt(
        prompt_set.first_vote_prompt,
        {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    votes = []
    for agent in agents:
        response = agent.vote(vote_prompt, task.possible_answers)
        votes.append({"agent": agent.name, **response})
    return votes, assignments


def collect_final_votes(
    agents: list[AgentState],
    prompts: PromptSet,
    possible_answers: list[str],
    discussion_messages: list[dict[str, Any]],
    *,
    vote_template: str | None = None,
) -> list[dict[str, str]]:
    group_discussion = format_discussion_transcript(discussion_messages)
    vote_prompt = render_prompt(
        vote_template if vote_template is not None else prompts.vote_prompt,
        {
            "group_discussion": group_discussion,
            "possible_answers": ", ".join(possible_answers),
        },
    )
    votes = []
    for agent in agents:
        response = agent.vote(vote_prompt, possible_answers)
        votes.append({"agent": agent.name, **response})
    return votes


def run_scenario(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    prompts: PromptSet | None = None,
    *,
    rounds: int = 15,
    seed: int | None = None,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
    final_vote_scope: str = "last_round",
    discussion_scope: str = "latest_other_messages",
    round_vote_scope: str | None = None,
    checkpoint_vote_prompt: str | None = None,
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if final_vote_scope not in {"last_round", "full_transcript"}:
        raise ValueError("final_vote_scope must be 'last_round' or 'full_transcript'")
    if discussion_scope not in {"latest_other_messages", "full_transcript"}:
        raise ValueError("discussion_scope must be 'latest_other_messages' or 'full_transcript'")
    if round_vote_scope not in {None, "cumulative"}:
        raise ValueError("round_vote_scope must be None or 'cumulative'")
    if round_vote_scope is not None and not checkpoint_vote_prompt:
        raise ValueError("Private round tracking requires an explicit checkpoint vote prompt")

    profile = Profile(profile)
    prompt_set = prompts or load_prompts()
    coordinator_enabled = bool(prompt_set.coordinator_system_prompt)
    coordinator_v4 = prompt_set.prompt_version in {"v4", "v4.1"}
    if coordinator_enabled:
        if profile != Profile.HIDDEN:
            raise ValueError("Coordinator experiments require Hidden profile")
        round_vote_scope = "cumulative"
        checkpoint_vote_prompt = prompt_set.checkpoint_vote_prompt
        if not checkpoint_vote_prompt or not prompt_set.coordinator_user_prompt:
            raise ValueError("Coordinator experiments require checkpoint and agent guidance prompts")
    rng = random.Random(seed)
    agents, assignments = instantiate_agents(
        task,
        model_client,
        profile,
        prompt_set,
        rng,
        extra_prompt=extra_prompt,
        special_agent_ratio=special_agent_ratio,
    )

    run_data: dict[str, Any] = {
        "task_id": task.id,
        "scenario": task.name,
        "profile": profile.value,
        "model": model_client.model_name,
        "num_agents": len(agents),
        "num_special": sum(agent.is_special for agent in agents),
        "correct_answer": task.correct_answer,
        "possible_answers": list(task.possible_answers),
        "fact_assignments": assignments,
        "prompt_version": prompt_set.prompt_version,
        "prompt_file": prompt_set.prompt_file,
        "system_prompt_version": "v1" if profile == Profile.FULL and prompt_set.prompt_version == "v2"
        else prompt_set.prompt_version,
        "prompt_diagnostics": {
            "system_prompts": [{"agent": a.name, "prompt": a.history[0]["content"]} for a in agents],
        },
        "initial_votes": [],
        "rounds": [],
        "discussion_transcript": [],
        "final_votes": [],
        "majority_vote": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }
    if coordinator_enabled:
        run_data.update(coordinator_enabled=True, round_vote_tracking=True,
                        system_prompt_version="v2" if not coordinator_v4 else prompt_set.prompt_version,
                        coordinator_calls=rounds - 1,
                        expected_generation_calls=len(agents) * (1 + 2 * rounds) + rounds - 1)

    initial_vote_prompt = render_prompt(
        prompt_set.first_vote_prompt,
        {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    run_data["prompt_diagnostics"]["initial_vote_prompt"] = initial_vote_prompt
    for agent in agents:
        response = agent.vote(initial_vote_prompt, task.possible_answers)
        run_data["initial_votes"].append({"agent": agent.name, **response})

    transcript: list[dict[str, Any]] = []
    previous_v4_state: dict[str, Any] | None = None
    for round_index in range(rounds):
        round_data = {"round": round_index + 1, "messages": [], "votes": []}
        if round_index == 0:
            previous_messages: list[str] = []
            for agent_index, agent in enumerate(agents):
                initial_vote = run_data["initial_votes"][agent_index]
                initial_prefix = (
                    f"Your initial vote was: {initial_vote['vote']}\n"
                    f"Your initial rationale was: {initial_vote['rationale']}\n\n"
                )
                if not previous_messages:
                    prompt = initial_prefix + prompt_set.first_user_prompt
                else:
                    prompt = initial_prefix + render_prompt(
                        prompt_set.user_prompt,
                        {
                            "messages": "\n".join(previous_messages),
                            "extra": extra_prompt if agent.is_special else "",
                        },
                    )
                response = agent.chat(prompt)
                previous_messages.append(f"{agent.name}: {response}")
                message = {"agent": agent.name, "prompt": prompt, "response": response}
                round_data["messages"].append(message)
                transcript.append({"round": round_index + 1, "agent": agent.name, "response": response})
        else:
            for agent in agents:
                current_index = agents.index(agent)
                previous_messages = []
                for offset in range(1, len(agents)):
                    other = agents[(current_index + offset) % len(agents)]
                    previous_messages.append(f"{other.name}: {other.history[-1]['content']}")
                guidance = ""
                if coordinator_enabled:
                    if coordinator_v4:
                        guidance = run_data["rounds"][-1]["coordinator"]["agent_guidance"]
                    else:
                        guidance = run_data["rounds"][-1]["coordinator_annotation"]["formatted_annotation"]
                if discussion_scope == "full_transcript":
                    previous_messages = [format_discussion_transcript(transcript)]
                prompt = render_prompt(
                    prompt_set.coordinator_user_prompt if coordinator_enabled else prompt_set.user_prompt,
                    {
                        "messages": "\n".join(previous_messages),
                        "extra": extra_prompt if agent.is_special else "",
                        "coordinator_annotation": guidance,
                        "coordinator_guidance": guidance,
                    },
                )
                response = agent.chat(prompt)
                message = {"agent": agent.name, "prompt": prompt, "response": response}
                round_data["messages"].append(message)
                transcript.append({"round": round_index + 1, "agent": agent.name, "response": response})

        if round_vote_scope == "cumulative":
            # Measurement calls use local history copies in AgentState.vote().
            # Their outputs never enter agent histories or the public transcript.
            round_data["votes"] = collect_final_votes(
                agents, prompt_set, task.possible_answers, transcript,
                vote_template=checkpoint_vote_prompt,
            )
            round_data["checkpoint_vote_prompt"] = render_prompt(
                checkpoint_vote_prompt,
                {"group_discussion": format_discussion_transcript(transcript),
                 "possible_answers": ", ".join(task.possible_answers)},
            )
        elif round_index == rounds - 1:
            vote_messages = round_data["messages"] if final_vote_scope == "last_round" else transcript
            run_data["prompt_diagnostics"]["final_vote_prompt"] = render_prompt(
                prompt_set.vote_prompt,
                {"group_discussion": format_discussion_transcript(vote_messages),
                 "possible_answers": ", ".join(task.possible_answers)},
            )
            round_data["votes"] = collect_final_votes(agents, prompt_set, task.possible_answers, vote_messages)
        if coordinator_v4 and round_index < rounds - 1:
            if prompt_set.prompt_version == "v4.1":
                from hiddenbench.coordinator_v4_1 import run_coordinator_v4_1 as run_coordinator_v4
            else:
                from hiddenbench.coordinator_v4 import run_coordinator_v4

            # The function's signature accepts only this public allowlist; no
            # assignments, votes, rationale, correctness labels, or task object.
            coordinator = run_coordinator_v4(
                description=task.description, answers=tuple(task.possible_answers),
                shared_information=tuple(task.shared_information), num_agents=len(agents),
                round_number=round_index + 1, public_messages=list(transcript),
                previous_state=previous_v4_state, model=model_client, prompts=prompt_set,
            )
            round_data["coordinator"] = coordinator
            previous_v4_state = coordinator["internal_state"]
        elif coordinator_enabled and round_index < rounds - 1:
            from hiddenbench.coordinator import PublicCoordinatorInput, run_coordinator

            # Construct from public allowlist fields only. The coordinator
            # function cannot accept a task, AgentState, assignments or votes.
            public_input = PublicCoordinatorInput(
                description=task.description, possible_answers=tuple(task.possible_answers),
                num_agents=len(agents), shared_information=tuple(task.shared_information),
                round_number=round_index + 1,
                public_discussion=format_discussion_transcript(transcript),
            )
            round_data["coordinator_annotation"] = run_coordinator(public_input, model_client, prompt_set)
        run_data["rounds"].append(round_data)

    run_data["discussion_transcript"] = transcript
    run_data["final_votes"] = run_data["rounds"][-1]["votes"]
    vote_counts = Counter(vote["vote"] for vote in run_data["final_votes"])
    if vote_counts:
        run_data["majority_vote"] = vote_counts.most_common(1)[0][0]
    if round_vote_scope is not None:
        from hiddenbench.round_tracking import add_vote_tracking

        run_data["round_vote_scope"] = round_vote_scope
        add_vote_tracking(run_data)
    if coordinator_v4:
        from hiddenbench.coordinator_v4 import add_v4_diagnostics

        add_v4_diagnostics(run_data)
    return run_data


def _scenario_seed(base_seed: int | None, task_index: int, duplication_index: int) -> int | None:
    if base_seed is None:
        return None
    return base_seed + task_index * 10_000 + duplication_index


def run_evaluation(
    benchmark: list[BenchmarkTask],
    model_client: ModelClient,
    profile: Profile | str,
    *,
    rounds: int = 15,
    duplications: int = 1,
    seed: int | None = None,
    max_workers: int = 1,
    prompts: PromptSet | None = None,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    profile = Profile(profile)
    prompt_set = prompts or load_prompts()
    jobs = [
        (task_index, duplication_index, task)
        for task_index, task in enumerate(benchmark)
        for duplication_index in range(duplications)
    ]

    metadata = {
        "benchmark": "HiddenBench",
        "prompt_version": prompt_set.prompt_version,
        "prompt_file": prompt_set.prompt_file,
        "profile": profile.value,
        "model": model_client.model_name,
        "rounds": rounds,
        "duplications": duplications,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if prompt_set.coordinator_system_prompt:
        metadata.update(coordinator_enabled=True, round_vote_tracking=True,
                        coordinator_calls=rounds - 1, round_vote_scope="cumulative",
                        expected_generation_calls=sum(len(task.hidden_information) * (1 + 2 * rounds) + rounds - 1 for _, _, task in jobs))
    results: dict[str, Any] = {"metadata": metadata, "runs": []}

    def run_job(job: tuple[int, int, BenchmarkTask]) -> dict[str, Any]:
        task_index, duplication_index, task = job
        return run_scenario(
            task,
            model_client,
            profile,
            prompt_set,
            rounds=rounds,
            seed=_scenario_seed(seed, task_index, duplication_index),
            extra_prompt=extra_prompt,
            special_agent_ratio=special_agent_ratio,
        )

    if max_workers <= 1:
        sequential_runs = []
        for index, job in enumerate(jobs, start=1):
            run = run_job(job)
            sequential_runs.append(run)
            if progress_callback is not None:
                progress_callback(index, len(jobs), run)
        results["runs"] = sequential_runs
        return results

    ordered: list[dict[str, Any] | None] = [None] * len(jobs)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(run_job, job): index for index, job in enumerate(jobs)}
        for future in as_completed(future_to_index):
            run = future.result()
            ordered[future_to_index[future]] = run
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(jobs), run)
    results["runs"] = [item for item in ordered if item is not None]
    return results
