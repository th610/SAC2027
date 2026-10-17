"""Build the anonymous, offline HiddenBench evidence package from local saved artifacts.

This copies no raw API request logs. It never imports an experiment runner or calls a model.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
ROOT = RELEASE.parent
RUNS = {
    'gpt_md': 'runs/prompt_v1_gpt4o_mini_hidden_tasks_1_65_20261015',
    'gpt_eda': 'runs/paper_structured_protocol_gpt4o_mini_hidden_20261016',
    'gpt_sed': 'runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015',
    'gpt_sed_fresh': 'runs/prompt_v2_gpt4o_mini_hidden_fresh_sed_20261017',
    'gpt_eda_update': 'runs/eda_update_sed_cumulative_gpt4o_mini_20261017/eda_update',
    'gpt_sed_cumulative': 'runs/eda_update_sed_cumulative_gpt4o_mini_20261017/sed_cumulative',
    'gpt_sed_minus_a1': 'runs/v2_module_a_ablation_gpt4o_mini_hidden_20261016/minus_a1',
    'gpt_sed_minus_a2': 'runs/v2_module_a_ablation_gpt4o_mini_hidden_20261016/minus_a2',
    'qwen_md': 'runs/v1_minimal_qwen3_8b_bf16_hidden_20261016',
    'qwen_eda': 'runs/paper_structured_protocol_qwen3_8b_bf16_hidden_20261017_verified',
    'qwen_sed': 'runs/v2_full_qwen3_8b_bf16_hidden_20261016',
    'gemini_md': 'runs/gemini_3_5_flash_lite_base_hidden_20261016',
    'gemini_eda': 'runs/paper_structured_protocol_gemini_3_5_flash_lite_hidden_20261017',
    'gemini_sed': 'runs/v2_gemini_3_5_flash_lite_hidden_20261016',
}
PROMPTS = {
    'md': 'src/hiddenbench/prompts.json',
    'sed': 'src/hiddenbench/prompts_v2.json',
    'eda': 'analysis/paper_structured_protocol/prompts.json',
    'sed_minus_a1': 'analysis/v2_module_a_ablation/minus_a1.json',
    'sed_minus_a2': 'analysis/v2_module_a_ablation/minus_a2.json',
}
RUN_PROMPTS = {
    'gpt_md': ['md'], 'gpt_eda': ['md', 'eda'], 'gpt_sed': ['sed'],
    'gpt_sed_fresh': ['sed'], 'gpt_eda_update': ['md', 'eda'],
    'gpt_sed_cumulative': ['sed'], 'gpt_sed_minus_a1': ['sed_minus_a1'],
    'gpt_sed_minus_a2': ['sed_minus_a2'], 'qwen_md': ['md'],
    'qwen_eda': ['md', 'eda'], 'qwen_sed': ['sed'],
    'gemini_md': ['md'], 'gemini_eda': ['md', 'eda'], 'gemini_sed': ['sed'],
}
SOURCES = [
    'src/hiddenbench/__init__.py', 'src/hiddenbench/benchmark.py',
    'src/hiddenbench/metrics.py', 'src/hiddenbench/models.py',
    'src/hiddenbench/prompts.py', 'src/hiddenbench/simulator.py',
    'scripts/run_prompt_v2_tasks.py', 'scripts/run_paper_structured_protocol.py',
    'scripts/run_v2_gpt4o_mini_fresh_sed.py',
    'scripts/run_eda_update_sed_cumulative.py',
    'scripts/run_v2_module_a_ablation.py', 'scripts/run_v2_ablations.py',
    'scripts/run_v1_qwen3_bf16_parallel.py',
    'scripts/run_v2_qwen3_bf16_parallel.py',
    'scripts/run_paper_structured_protocol_qwen3_bf16.py',
    'scripts/recover_qwen_eda_task59.py',
    'scripts/run_gemini_base_minimal.py',
    'scripts/run_v2_gemini_3_5_flash_lite.py',
    'scripts/run_paper_structured_protocol_gemini_eda.py',
]
# Run history is recorded here and in saved logs; this field is never inferred from a score.
HISTORY = {
    'gpt_sed': ['Tasks 1–20 copied from an earlier completed V2 run; Tasks 21–65 completed later.',
                'The old 65-task artifacts do not contain numeric token usage.'],
    'gpt_sed_fresh': ['Fresh, cache-isolated rerun; all 65 tasks generated again.'],
    'qwen_eda': ['Tasks 1–2 reused from the EDA smoke run.',
                 'Task 59 final artifact was recovered after a failed parse/encoding attempt; separate failed-attempt logs remain in the local source project.'],
    'gpt_eda_update': ['Extra private checkpoint vote after Exchange 1; 6N logical calls rather than 5N.'],
    'gpt_sed_cumulative': ['SED prompt unchanged; round 2/3 and final vote see accumulated public transcript.'],
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_run(data: dict) -> dict:
    return data['run'] if isinstance(data.get('run'), dict) else data['runs'][0]


def model_settings(metadata: dict) -> dict:
    keep = ('model', 'revision', 'temperature', 'top_p', 'max_tokens',
            'max_output_tokens', 'seed', 'rounds', 'exchanges', 'decide_passes',
            'thinking_level', 'enable_thinking', 'precision', 'quantization',
            'api', 'sdk', 'sdk_version', 'prompt_sha256', 'protocol_prompt_sha256',
            'v1_prompt_sha256', 'runner_sha256', 'fresh_no_cache')
    return {k: metadata[k] for k in keep if k in metadata}


def token_usage(data: dict, run: dict) -> dict:
    raw = data.get('token_usage') or data.get('usage') or run.get('token_usage')
    if not isinstance(raw, dict):
        return {'prompt_tokens': None, 'completion_tokens': None,
                'total_tokens': None, 'source': 'not recorded'}
    prompt = raw.get('prompt_tokens')
    completion = raw.get('completion_tokens', raw.get('candidate_tokens'))
    total = raw.get('total_tokens')
    if total is None and isinstance(prompt, int) and isinstance(completion, int):
        total = prompt + completion
    return {'prompt_tokens': prompt, 'completion_tokens': completion,
            'total_tokens': total,
            'source': 'saved API/vLLM usage fields; Qwen MD/SED total is prompt + completion' if prompt is not None else 'not recorded',
            'thoughts_tokens_recorded': raw.get('thoughts_tokens') if 'thoughts_tokens' in raw else None}


def stages(run: dict) -> list[dict]:
    if 'exchanges' in run:
        sequence = [(f'Exchange {i}', stage['messages']) for i, stage in enumerate(run['exchanges'], 1)]
        sequence.append(('Decide', run['decide_messages']))
    else:
        sequence = [(f'Round {i}', stage['messages']) for i, stage in enumerate(run['rounds'], 1)]
    return [{'name': label,
             'messages': [{'agent': m['agent'], 'response': m['response']} for m in messages]}
            for label, messages in sequence]


def compact(task_id: int, data: dict, run: dict) -> dict:
    return {
        'task_id': task_id,
        'gold': run['correct_answer'],
        'possible_answers': run['possible_answers'],
        'num_agents': run['num_agents'],
        'fact_assignments': [{'agent': a['agent'],
                              'assigned_hidden_information': a['assigned_hidden_information']}
                             for a in run['fact_assignments']],
        'initial_votes': [{'agent': v['agent'], 'vote': v['vote']} for v in run['initial_votes']],
        'stages': stages(run),
        'final_votes': [{'agent': v['agent'], 'vote': v['vote']} for v in run['final_votes']],
        'final_plurality_recorded': run['majority_vote'],
        'leader_update_votes': [{'agent': v['agent'], 'vote': v['vote']}
                                for v in run.get('leader_update_votes', [])],
        'tokens': token_usage(data, run),
    }


def copy_source(relative: str) -> dict:
    original = ROOT / relative
    raw = original.read_bytes()
    if relative.startswith('scripts/'):
        target = RELEASE / relative
    else:
        target = RELEASE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    text = raw.decode('utf-8')
    # The only textual adjustment to historical runners removes the local credential-file path.
    sanitized = re.sub(r'(?<=Path\(")[^"\n]*\/\.env', '.env', text)
    sanitized = re.sub(r"(?<=Path\(')[^'\n]*\/\.env", '.env', sanitized)
    output = sanitized.encode('utf-8')
    target.write_bytes(output)
    return {'source_file': relative, 'source_sha256': sha(raw),
            'release_file': relative, 'release_sha256': sha(output),
            'changed_for_anonymity': raw != output,
            'change': 'local dotenv default path replaced with .env' if raw != output else None}


def main() -> None:
    benchmark = ROOT / 'data/benchmark.json'
    benchmark_bytes = benchmark.read_bytes()
    (RELEASE / 'data').mkdir(parents=True, exist_ok=True)
    (RELEASE / 'data/benchmark.json').write_bytes(benchmark_bytes)
    manifest = {
        'package': 'SAC 2027 IRMAS anonymous SED reproducibility package',
        'scope': 'offline recomputation from saved 65-task experiment artifacts',
        'dataset': {'release_file': 'data/benchmark.json', 'sha256': sha(benchmark_bytes),
                    'source': 'YuxuanLi1225/HiddenBench, MIT', 'task_count': 65},
        'prompts': {}, 'code': [], 'runs': {},
        'scoring': {'plurality': "Counter(vote).most_common(1)[0][0] (first vote wins ties)",
                    'correct': 'selected answer == gold answer',
                    'private_fact_mention': 'owner utterance: normalized fact substring OR >=50% unique-token overlap'},
        'known_history': HISTORY,
        'paper_table_validation': 'pending: final manuscript not available in the local project',
    }
    for name, relative in PROMPTS.items():
        raw = (ROOT / relative).read_bytes()
        target = RELEASE / 'prompts' / (name + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        historical_path = RELEASE / relative
        historical_path.parent.mkdir(parents=True, exist_ok=True)
        historical_path.write_bytes(raw)
        manifest['prompts'][name] = {'source_file': relative,
                                     'source_sha256': sha(raw),
                                     'release_file': str(target.relative_to(RELEASE)),
                                     'release_sha256': sha(target.read_bytes()),
                                     'historical_path_copy': relative}
    manifest['code'] = [copy_source(path) for path in SOURCES]
    # The upstream MIT notice must remain intact when its code/data is redistributed.
    shutil.copyfile(ROOT / 'LICENSE', RELEASE / 'licenses/HIDDENBENCH_MIT.txt')
    for condition, source_dir in RUNS.items():
        rows = []
        source_hashes = {}
        settings_seen = set()
        snapshots = set()
        logical = 0
        retries = 0
        for task_id in range(1, 66):
            source = ROOT / source_dir / f'task_{task_id}.json'
            raw = source.read_bytes()
            data = json.loads(raw)
            run = load_run(data)
            if str(run['task_id']) != str(task_id):
                raise ValueError(f'Task ID mismatch: {source}')
            source_hashes[str(task_id)] = sha(raw)
            settings_seen.add(json.dumps(model_settings(data.get('metadata', {})), sort_keys=True))
            logical += data.get('generation_call_count') or run.get('generation_call_count') or 0
            retries += data.get('retry_count') or data.get('retry_attempt_count') or 0
            attempts = data.get('api_attempts') or run.get('api_attempts') or []
            snapshots.update(a.get('response_model') for a in attempts if a.get('response_model'))
            rows.append(compact(task_id, data, run))
        path = RELEASE / 'results' / f'{condition}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows), encoding='utf-8')
        manifest['runs'][condition] = {
            'source_directory': source_dir,
            'prompt_versions': RUN_PROMPTS[condition],
            'prompt_sha256_current_files': {name: manifest['prompts'][name]['source_sha256']
                                            for name in RUN_PROMPTS[condition]},
            'source_task_sha256': source_hashes,
            'release_file': str(path.relative_to(RELEASE)),
            'release_sha256': sha(path.read_bytes()),
            'task_count': len(rows), 'settings_observed': [json.loads(s) for s in sorted(settings_seen)],
            'response_model_ids_observed': sorted(snapshots),
            'logical_calls_recorded': logical if logical else None,
            'retries_recorded': retries if logical else None,
            'history': HISTORY.get(condition, []),
        }
    manifest['supporting_history_artifacts'] = {
        relative: sha((ROOT / relative).read_bytes())
        for relative in (
            'runs/paper_structured_protocol_qwen3_8b_bf16_hidden_20261017_task59_recovery/recovery_summary.json',
            'runs/paper_structured_protocol_qwen3_8b_bf16_hidden_20261017_verified/task_59.error.attempt1.json',
            'runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015/summary.json',
        ) if (ROOT / relative).is_file()
    }
    manifest['runs']['gpt_eda_update']['runtime_prompt_edit'] = {
        'stage': 'Exchange 2 only',
        'original': 'The initial group front-runner, determined from the pre-discussion votes and held fixed for both Exchange stages, is {front_runner}.',
        'variant': 'The current group front-runner, updated from private votes after Exchange 1, is {front_runner}.',
    }
    (RELEASE / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Built', len(manifest['runs']), 'runs,', sum(r['task_count'] for r in manifest['runs'].values()), 'task records')


if __name__ == '__main__':
    main()
