#!/usr/bin/env python3
"""Recompute all released HiddenBench metrics offline using only the JSONL evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('md', 'eda', 'sed')
EXPECTED = {
    'gpt_md': 17, 'gpt_eda': 45, 'gpt_sed': 57,
    'gpt_sed_fresh': 58, 'gpt_eda_update': 41,
    'gpt_sed_cumulative': 54, 'gpt_sed_minus_a1': 50,
    'gpt_sed_minus_a2': 52, 'qwen_md': 15, 'qwen_eda': 36,
    'qwen_sed': 60, 'gemini_md': 24, 'gemini_eda': 27,
    'gemini_sed': 58,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plurality(votes: list[dict]) -> tuple[str, bool]:
    count = Counter(v['vote'] for v in votes)
    winner, maximum = count.most_common(1)[0]
    return winner, sum(n == maximum for n in count.values()) > 1


def norm(value: str) -> str:
    return re.sub(r'\W+', ' ', value.casefold()).strip()


def mention(fact: str, message: str) -> bool:
    # Same lexical rule as the saved project's coordinator_v4 disclosure diagnostic.
    fact_normal = norm(fact)
    tokens = set(fact_normal.split())
    msg_normal = norm(message)
    return fact_normal in msg_normal or bool(tokens and len(tokens & set(msg_normal.split())) / len(tokens) >= 0.5)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_all(manifest: dict) -> dict[str, dict[int, dict]]:
    all_runs = {}
    for condition, info in manifest['runs'].items():
        path = ROOT / info['release_file']
        if digest(path) != info['release_sha256']:
            raise ValueError(f'Released JSONL checksum mismatch: {path}')
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        by_id = {int(r['task_id']): r for r in records}
        if len(records) != 65 or set(by_id) != set(range(1, 66)):
            raise ValueError(f'{condition}: expected exactly Task 1–65')
        for task_id, r in by_id.items():
            n = r['num_agents']
            if len(r['fact_assignments']) != n or len(r['initial_votes']) != n or len(r['final_votes']) != n:
                raise ValueError(f'{condition} Task {task_id}: incomplete agents/votes')
            if len(r['stages']) != 3 or any(len(s['messages']) != n for s in r['stages']):
                raise ValueError(f'{condition} Task {task_id}: incomplete public stages')
            for vote in r['initial_votes'] + r['final_votes']:
                if vote['vote'] not in r['possible_answers']:
                    raise ValueError(f'{condition} Task {task_id}: invalid vote')
            if plurality(r['final_votes'])[0] != r['final_plurality_recorded']:
                raise ValueError(f'{condition} Task {task_id}: plurality mismatch')
        all_runs[condition] = by_id
    first = next(iter(all_runs.values()))
    for condition, run in all_runs.items():
        for task_id in range(1, 66):
            a, b = first[task_id], run[task_id]
            for field in ('gold', 'possible_answers', 'num_agents', 'fact_assignments'):
                if a[field] != b[field]:
                    raise ValueError(f'{condition} Task {task_id}: {field} differs from reference')
    return all_runs


def load_recovery_usage(manifest: dict) -> dict:
    """Load separately logged Qwen EDA parse-recovery API usage."""
    path = ROOT / 'results' / 'qwen_eda_recovery_usage.json'
    if not path.exists():
        return {'api_attempts': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    expected_hash = manifest.get('supporting_history_artifacts', {}).get('results/qwen_eda_recovery_usage.json')
    if expected_hash is None or digest(path) != expected_hash:
        raise ValueError(f'Recovery usage checksum mismatch: {path}')
    record = json.loads(path.read_text(encoding='utf-8'))
    attempts = record['attempts']
    totals = {k: sum(a['usage'][k] for a in attempts)
              for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    if len(attempts) != 3 or totals != record['expected_totals']:
        raise ValueError('Qwen EDA recovery usage does not match its recorded totals')
    return {'api_attempts': len(attempts), **totals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis' / 'generated')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    if digest(ROOT / manifest['dataset']['release_file']) != manifest['dataset']['sha256']:
        raise ValueError('Benchmark checksum mismatch')
    runs = load_all(manifest)
    qwen_eda_recovery = load_recovery_usage(manifest)
    accuracy_rows, transition_rows, disclosure_rows, token_rows = [], [], [], []
    task_rows = []
    for condition, tasks in runs.items():
        transitions = Counter()
        correct = initial_correct = initial_ties = final_ties = 0
        tokens = Counter()
        missing_tokens = 0
        first_stage_count = Counter()
        total_facts = 0
        for task_id in range(1, 66):
            r = tasks[task_id]
            initial, initial_tie = plurality(r['initial_votes'])
            final, final_tie = plurality(r['final_votes'])
            init_ok, final_ok = initial == r['gold'], final == r['gold']
            correct += final_ok
            initial_correct += init_ok
            initial_ties += initial_tie
            final_ties += final_tie
            transitions[('C' if init_ok else 'W') + '_to_' + ('C' if final_ok else 'W')] += 1
            task_rows.append({'condition': condition, 'task_id': task_id, 'gold': r['gold'],
                              'initial_plurality': initial, 'initial_correct': int(init_ok),
                              'initial_tie': int(initial_tie), 'final_plurality': final,
                              'final_correct': int(final_ok), 'final_tie': int(final_tie)})
            token = r['tokens']
            if not all(isinstance(token.get(k), int) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                missing_tokens += 1
            else:
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    tokens[k] += token[k]
            for assigned in r['fact_assignments']:
                total_facts += 1
                owner = assigned['agent']
                facts = assigned['assigned_hidden_information']
                if len(facts) != 1:
                    raise ValueError(f'{condition} Task {task_id}: expected one hidden fact per owner')
                first_stage = 0
                for stage_id, stage in enumerate(r['stages'], 1):
                    if any(m['agent'] == owner and mention(facts[0], m['response']) for m in stage['messages']):
                        first_stage = stage_id
                        break
                first_stage_count[first_stage] += 1
        accuracy_rows.append({'condition': condition, 'tasks': 65, 'initial_correct': initial_correct,
                              'final_correct': correct, 'final_accuracy': correct / 65,
                              'initial_ties': initial_ties, 'final_ties': final_ties})
        transition_rows.append({'condition': condition,
                                'wrong_to_correct': transitions['W_to_C'],
                                'correct_to_wrong': transitions['C_to_W'],
                                'correct_to_correct': transitions['C_to_C'],
                                'wrong_to_wrong': transitions['W_to_W']})
        disclosure_rows.append({'condition': condition, 'assignments': total_facts,
                                'first_stage_1': first_stage_count[1],
                                'first_stage_2': first_stage_count[2],
                                'first_stage_3': first_stage_count[3],
                                'never_detected': first_stage_count[0],
                                'ever_detected': total_facts - first_stage_count[0],
                                'mention_rate': (total_facts - first_stage_count[0]) / total_facts,
                                'stage_3_label': 'Decide' if '_eda' in condition else 'Round 3'})
        extra_usage = qwen_eda_recovery if condition == 'qwen_eda' else {
            'api_attempts': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        if not missing_tokens and condition == 'qwen_eda':
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                tokens[k] += extra_usage[k]
        token_rows.append({'condition': condition, 'tasks': 65,
                           'tasks_missing_numeric_usage': missing_tokens,
                           'prompt_tokens': tokens['prompt_tokens'] if not missing_tokens else 'not recorded',
                           'completion_tokens': tokens['completion_tokens'] if not missing_tokens else 'not recorded',
                           'total_tokens': tokens['total_tokens'] if not missing_tokens else 'not recorded',
                           'additional_recovery_api_attempts': extra_usage['api_attempts'],
                           'source': ('saved API usage fields plus three separately logged parse-recovery attempts'
                                      if condition == 'qwen_eda' else tasks[1]['tokens']['source'])})
    comparisons = []
    for model in ('gpt', 'qwen', 'gemini'):
        comparisons.extend([(f'{model}_md', f'{model}_eda'),
                            (f'{model}_md', f'{model}_sed'),
                            (f'{model}_eda', f'{model}_sed')])
    comparisons.extend([('gpt_sed', 'gpt_sed_fresh'),
                        ('gpt_eda', 'gpt_eda_update'),
                        ('gpt_sed', 'gpt_sed_cumulative'),
                        ('gpt_sed_fresh', 'gpt_sed_cumulative'),
                        ('gpt_sed', 'gpt_sed_minus_a1'),
                        ('gpt_sed', 'gpt_sed_minus_a2'),
                        ('gpt_eda', 'gpt_sed_cumulative'),
                        ('gpt_sed', 'gpt_eda_update')])
    paired_rows = []
    for baseline, variant in comparisons:
        count = Counter()
        for task_id in range(1, 66):
            a, b = runs[baseline][task_id], runs[variant][task_id]
            count[(plurality(a['final_votes'])[0] == a['gold'], plurality(b['final_votes'])[0] == b['gold'])] += 1
        b_only, v_only = count[(True, False)], count[(False, True)]
        paired_rows.append({'baseline': baseline, 'variant': variant,
                            'both_correct': count[(True, True)], 'baseline_only_b': b_only,
                            'variant_only_c': v_only, 'both_wrong': count[(False, False)],
                            'mcnemar_exact_two_sided_p': exact_mcnemar(b_only, v_only)})
    write_csv(out / 'accuracy.csv', accuracy_rows)
    write_csv(out / 'paired_mcnemar.csv', paired_rows)
    write_csv(out / 'transitions.csv', transition_rows)
    write_csv(out / 'private_fact_mention.csv', disclosure_rows)
    write_csv(out / 'token_usage.csv', token_rows)
    write_csv(out / 'task_outcomes.csv', task_rows)
    mismatches = {r['condition']: {'expected': EXPECTED[r['condition']], 'actual': r['final_correct']}
                  for r in accuracy_rows if r['final_correct'] != EXPECTED[r['condition']]}
    verification = {'runs_checked': len(runs), 'task_records_checked': sum(len(t) for t in runs.values()),
                    'same_gold_options_agents_assignments': True, 'source_checksum_validation': True,
                    'correct_count_mismatches': mismatches,
                    'paper_tables_1_to_6': 'see TABLE_AUDIT.md; table values checked against supplied anonymous manuscript',
                    'qwen_eda_recovery_usage': qwen_eda_recovery,
                    'analysis_script_sha256': digest(Path(__file__)),
                    'no_model_calls': True}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
