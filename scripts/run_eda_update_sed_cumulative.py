#!/usr/bin/env python3
"""Isolated GPT-4o-mini EDA leader-update and SED transcript-scope variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_paper_structured_protocol as eda
from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, format_discussion_transcript, instantiate_agents, run_scenario

MODEL = 'gpt-4o-mini-2024-07-18'
TEMPERATURE = 0.0
SEED = 42
EDA_PROMPT = ROOT / 'analysis/paper_structured_protocol/prompts.json'
V1_PROMPT = ROOT / 'src/hiddenbench/prompts.json'
V2_PROMPT = ROOT / 'src/hiddenbench/prompts_v2.json'
DATA = ROOT / 'data/benchmark.json'
BASE_EDA = ROOT / 'runs/paper_structured_protocol_gpt4o_mini_hidden_20261016'
BASE_SED = ROOT / 'runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015'
BASE_SED_FRESH = ROOT / 'runs/prompt_v2_gpt4o_mini_hidden_fresh_sed_20261017'
DEFAULT_OUT = ROOT / 'runs/eda_update_sed_cumulative_gpt4o_mini_20261017'
LEADER_OLD = 'The initial group front-runner, determined from the pre-discussion votes and held fixed for both Exchange stages, is {front_runner}.'
LEADER_NEW = 'The current group front-runner, updated from private votes after Exchange 1, is {front_runner}.'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def run_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['run'] if isinstance(data.get('run'), dict) else data['runs'][0]


def majority(votes: list[dict[str, Any]]) -> str:
    return Counter(v['vote'] for v in votes).most_common(1)[0][0]


def check_sources() -> dict[str, Any]:
    benchmark = load_benchmark(path=str(DATA))
    if len(benchmark) != 65:
        raise ValueError('Expected exactly 65 benchmark tasks')
    raw_eda = json.loads(EDA_PROMPT.read_text(encoding='utf-8'))
    if raw_eda['exchange_prompt'].count(LEADER_OLD) != 1:
        raise ValueError('EDA leader sentence changed; review the minimal edit')
    for task_id in range(1, 66):
        task = find_task(benchmark, task_id=str(task_id))
        baseline = [run_object(folder / f'task_{task_id}.json') for folder in (BASE_EDA, BASE_SED, BASE_SED_FRESH)]
        for run in baseline:
            if run['correct_answer'] != task.correct_answer or run['possible_answers'] != task.possible_answers:
                raise ValueError(f'Task {task_id}: baseline gold/options mismatch')
        for version, run in (('v1', baseline[0]), ('v2', baseline[1]), ('v2', baseline[2])):
            _, assignments = instantiate_agents(task, None, Profile.HIDDEN, load_prompts(version=version), random.Random(SEED))
            if assignments != run['fact_assignments']:
                raise ValueError(f'Task {task_id}: {version} baseline fact assignment mismatch')
    return {'task_count': 65, 'model_requested': MODEL, 'temperature': TEMPERATURE, 'seed': SEED,
            'dataset_sha256': digest(DATA), 'eda_prompt_sha256': digest(EDA_PROMPT),
            'v1_prompt_sha256': digest(V1_PROMPT), 'v2_prompt_sha256': digest(V2_PROMPT),
            'runner_sha256': digest(Path(__file__)), 'baseline_assignments_checked': 195,
            'leader_method': 'private per-agent JSON vote after Exchange 1 using unchanged V1 vote prompt and public Exchange 1 transcript; first-seen plurality',
            'eda_exchange2_only_prompt_edit': {'before': LEADER_OLD, 'after': LEADER_NEW},
            'max_retries': 3, 'timeout_seconds': 90, 'top_p': 'omitted', 'max_output_tokens': 'omitted',
            'request_seed': 'omitted; seed 42 is task/fact assignment only'}


def run_eda_update(task, client, raw_prompts):
    agents, assignments = instantiate_agents(task, client, Profile.HIDDEN, load_prompts(version='v1'), random.Random(SEED))
    run = {'task_id': task.id, 'scenario': task.name, 'profile': 'hidden', 'model': client.model_name,
           'num_agents': len(agents), 'correct_answer': task.correct_answer, 'possible_answers': list(task.possible_answers),
           'fact_assignments': assignments, 'initial_votes': [], 'leader_update_votes': [], 'exchanges': [],
           'decide_messages': [], 'final_votes': []}
    stage_calls = {}
    start = client.logical_requests
    initial_prompt = render_prompt(eda.BASELINE_V1.first_vote_prompt,
        {'possible_answers': ', '.join(task.possible_answers),
         'group_discussion': 'No discussion has occurred yet. This is your initial vote based solely on the information available to you.'})
    for agent in agents:
        run['initial_votes'].append({'agent': agent.name, **agent.vote(initial_prompt, task.possible_answers)})
    stage_calls['initial_vote'] = client.logical_requests - start
    initial_leader = majority(run['initial_votes'])
    transcript = []
    for exchange_no in (1, 2):
        if exchange_no == 2:
            vote_start = client.logical_requests
            # AgentState.vote copies the private chat history: votes are neither public nor persistent.
            current_prompt = render_prompt(eda.BASELINE_V1.vote_prompt,
                {'group_discussion': format_discussion_transcript(transcript),
                 'possible_answers': ', '.join(task.possible_answers)})
            for agent in agents:
                run['leader_update_votes'].append({'agent': agent.name, **agent.vote(current_prompt, task.possible_answers)})
            stage_calls['leader_update_vote'] = client.logical_requests - vote_start
            current_leader = majority(run['leader_update_votes'])
            run['leader_update_vote_prompt'] = current_prompt
        else:
            current_leader = initial_leader
        stage_start = client.logical_requests
        stage = {'stage': f'exchange_{exchange_no}', 'leader': current_leader, 'messages': []}
        template = raw_prompts['exchange_prompt'] if exchange_no == 1 else [LEADER_NEW if line == LEADER_OLD else line for line in raw_prompts['exchange_prompt']]
        for agent in agents:
            private_state = ''
            if exchange_no == 1:
                own = next(v for v in run['initial_votes'] if v['agent'] == agent.name)
                private_state = f"Your initial vote was: {own['vote']}\nYour initial rationale was: {own['rationale']}\n\n"
            prompt = private_state + eda.expand(template,
                {'exchange_number': str(exchange_no), 'public_transcript': format_discussion_transcript(transcript),
                 'front_runner': current_leader})
            response = agent.chat(prompt)
            stage['messages'].append({'agent': agent.name, 'prompt': prompt, 'response': response})
            transcript.append({'round': exchange_no, 'agent': agent.name, 'response': response})
        run['exchanges'].append(stage)
        stage_calls[f'exchange_{exchange_no}'] = client.logical_requests - stage_start
    decide_start = client.logical_requests
    for agent in agents:
        prompt = eda.expand(raw_prompts['decide_prompt'], {'public_transcript': format_discussion_transcript(transcript)})
        response = agent.chat(prompt)
        run['decide_messages'].append({'agent': agent.name, 'prompt': prompt, 'response': response})
        transcript.append({'round': 3, 'agent': agent.name, 'response': response})
    stage_calls['decide'] = client.logical_requests - decide_start
    final_start = client.logical_requests
    final_prompt = render_prompt(eda.BASELINE_V1.vote_prompt,
        {'group_discussion': format_discussion_transcript(transcript), 'possible_answers': ', '.join(task.possible_answers)})
    for agent in agents:
        run['final_votes'].append({'agent': agent.name, **agent.vote(final_prompt, task.possible_answers)})
    stage_calls['final_vote'] = client.logical_requests - final_start
    run['initial_leader'] = initial_leader
    run['updated_leader'] = current_leader
    run['majority_vote'] = majority(run['final_votes'])
    run['final_correct'] = run['majority_vote'] == task.correct_answer
    run['discussion_transcript'] = transcript
    run['stage_logical_calls'] = stage_calls
    return run


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def summarize(folder: Path, condition: str) -> dict[str, Any]:
    rows, tokens, transitions = [], Counter(), Counter()
    for task_id in range(1, 66):
        path = folder / f'task_{task_id}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        run = data['run']
        gold = run['correct_answer']
        initial = majority(run['initial_votes']) == gold
        final = majority(run['final_votes']) == gold
        if run['majority_vote'] != majority(run['final_votes']):
            raise ValueError(f'Task {task_id}: plurality mismatch')
        transitions[('C' if initial else 'W') + '→' + ('C' if final else 'W')] += 1
        rows.append({'task_id': task_id, 'initial_correct': initial, 'final_correct': final,
                     'initial_majority': majority(run['initial_votes']), 'final_majority': run['majority_vote']})
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            tokens[key] += data['token_usage'][key]
    result = {'condition': condition, 'completed': len(rows), 'correct': sum(r['final_correct'] for r in rows),
              'transitions': {key: transitions[key] for key in ('W→C', 'C→W', 'C→C', 'W→W')},
              'token_usage': dict(tokens), 'task_rows': rows, 'paired': {}}
    if len(rows) == 65:
        for name, base in (('EDA', BASE_EDA), ('SED_original', BASE_SED), ('SED_fresh', BASE_SED_FRESH)):
            pair = Counter()
            for row in rows:
                prior = run_object(base / f"task_{row['task_id']}.json")
                base_correct = majority(prior['final_votes']) == prior['correct_answer']
                pair[(base_correct, row['final_correct'])] += 1
            b, c = pair[(True, False)], pair[(False, True)]
            result['paired'][name] = {'both_correct': pair[(True, True)], 'baseline_only_correct_b': b,
                                       'variant_only_correct_c': c, 'both_wrong': pair[(False, False)],
                                       'exact_mcnemar_two_sided_p': exact_mcnemar(b, c)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['eda_update', 'sed_cumulative'], required=True)
    parser.add_argument('--task-ids', nargs='+', type=int, default=list(range(1, 66)))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--dotenv', type=Path, default=Path('.env'))
    parser.add_argument('--static-only', action='store_true')
    args = parser.parse_args()
    validation = check_sources()
    if args.static_only:
        print(json.dumps(validation, indent=2)); return 0
    if not args.task_ids or len(set(args.task_ids)) != len(args.task_ids) or any(t < 1 or t > 65 for t in args.task_ids):
        parser.error('Task IDs must be distinct values in 1..65')
    load_dotenv(args.dotenv)
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY missing')
    folder = args.out.resolve() / args.condition
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != validation:
            raise ValueError(f'Run manifest mismatch: {manifest}')
    else:
        atomic_json(manifest, validation)
    benchmark = load_benchmark(path=str(DATA))
    raw_eda = json.loads(EDA_PROMPT.read_text(encoding='utf-8'))
    failures = []
    for task_id in args.task_ids:
        path = folder / f'task_{task_id}.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            run = saved['run']
            task = find_task(benchmark, task_id=str(task_id))
            expected = len(task.hidden_information) * (6 if args.condition == 'eda_update' else 5)
            if (saved['metadata'] != validation or run['correct_answer'] != task.correct_answer or
                saved['generation_call_count'] != expected or saved['successful_call_count'] != expected or
                len(run['initial_votes']) != len(task.hidden_information) or
                len(run['final_votes']) != len(task.hidden_information) or
                run['majority_vote'] != majority(run['final_votes'])):
                raise ValueError(f'Cached task validation failed: {path}')
            print(f'Task {task_id}: verified cached result', flush=True)
            continue
        task = find_task(benchmark, task_id=str(task_id))
        client = eda.UsageClient(MODEL, provider='openai', temperature=TEMPERATURE, max_retries=3, timeout=90)
        started = time.monotonic()
        try:
            print(f'Task {task_id}: started ({args.condition})', flush=True)
            run = (run_eda_update(task, client, raw_eda) if args.condition == 'eda_update' else
                   run_scenario(task, client, Profile.HIDDEN, load_prompts(version='v2'), rounds=3, seed=SEED,
                                discussion_scope='full_transcript', final_vote_scope='full_transcript'))
            n = run['num_agents']
            expected = n * (6 if args.condition == 'eda_update' else 5)
            if client.logical_requests != expected:
                raise RuntimeError(f'Task {task_id}: expected {expected} calls, got {client.logical_requests}')
            if args.condition == 'eda_update' and (len(run['leader_update_votes']) != n or len(run['exchanges']) != 2):
                raise RuntimeError('EDA stage completion failed')
            if args.condition == 'sed_cumulative' and (len(run['rounds']) != 3 or any(len(r['messages']) != n for r in run['rounds'])):
                raise RuntimeError('SED stage completion failed')
            attempts = client.attempts
            successful = [a for a in attempts if a['status'] == 'ok']
            if len(successful) != expected or any(a.get('response_model') != MODEL for a in successful):
                raise RuntimeError('Missing successful call or model snapshot mismatch')
            if any(a.get('usage') is None for a in successful):
                raise RuntimeError('API token usage missing')
            usage = {key: sum(a['usage'].get(key) or 0 for a in successful)
                     for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            saved = {'metadata': validation, 'run': run, 'generation_call_count': client.logical_requests,
                     'api_attempt_count': len(attempts), 'successful_call_count': len(successful),
                     'retry_count': len(attempts) - len(successful), 'token_usage': usage,
                     'elapsed_seconds': time.monotonic() - started, 'api_attempts': attempts,
                     'created_at': datetime.now(timezone.utc).isoformat()}
            atomic_json(path, saved)
            print(f"Task {task_id}: correct={run['majority_vote'] == run['correct_answer']} calls={expected} tokens={usage['total_tokens']}", flush=True)
        except Exception as exc:
            failures.append(task_id)
            atomic_json(folder / f'task_{task_id}.error.json',
                        {'task_id': task_id, 'type': type(exc).__name__, 'error': str(exc),
                         'api_attempts': client.attempts, 'created_at': datetime.now(timezone.utc).isoformat()})
            print(f'Task {task_id}: FAILED {type(exc).__name__}: {exc}', flush=True)
        finally:
            client.client.close()
    report = summarize(folder, args.condition)
    report['failed_task_ids'] = failures
    atomic_json(folder / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'task_rows'}, ensure_ascii=False, indent=2), flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
