# SED: anonymous HiddenBench reproducibility package

This package supports **offline recomputation** of the HiddenBench results used in an ACM SAC 2027 IRMAS submission. It contains 65 tasks and 14 saved model/protocol runs. It does not contain API credentials, account details, or the submitters' names. No model call is needed to check the released results.

## Study and conditions

The study compares Minimal Deliberation (MD), a paper based Exchange–Decide Adaptation (EDA), and Structured Evidence Deliberation (SED) on the same 65 HiddenBench hidden profile tasks. Four additional GPT analyses are supplied: a cache free SED rerun, EDA with an updated leader, SED with cumulative public transcript, and SED with either awareness (A1) or disclosure (A2) instructions removed. These five additions are separate from the original three model comparison.

| Model | MD | EDA | SED |
|---|---:|---:|---:|
| GPT-4o-mini | 17/65 | 45/65 | 57/65 |
| Qwen3-8B | 15/65 | 36/65 | 60/65 |
| Gemini 3.5 Flash-Lite | 24/65 | 27/65 | 58/65 |

GPT additional runs: SED Fresh 58/65, EDA-update 41/65, SED-cumulative 54/65, SED−A1 50/65, SED−A2 52/65. The original GPT SED result combines 20 previously completed task artifacts with 45 later tasks; Fresh SED is a separate 65-task run.

## Contents

- `data/benchmark.json`: the official 65-task HiddenBench data, byte identical to the local source file.
- `prompts/`: exact JSON files used for MD, SED, EDA, SED−A1, and SED−A2. Identical copies also appear at their historical paths under `src/hiddenbench/` and `analysis/` so the archived runners can resolve them. EDA-update changes one Exchange 2 leader sentence at runtime; the exact replacement is recorded in `MANIFEST.json` and `scripts/run_eda_update_sed_cumulative.py`.
- `scripts/`, `src/hiddenbench/`: copies of the existing experiment runner and simulator code. Only local `.env` default paths were replaced for anonymity. `MANIFEST.json` records both original and released SHA-256 values.
- `results/*.jsonl`: one compact, anonymous 65-task artifact per run. Records retain gold labels, allowed options, hidden fact assignments, public messages, actual initial/final ballots, and recorded token usage. Raw API request histories and provider IDs are omitted.
- `analysis/recompute.py`: independent, standard-library-only metric calculation from the JSONL records.
- `analysis/generated/`: CSV and verification JSON produced by the offline calculation.
- `MANIFEST.json`: prompt/code/data hashes, all 910 original task-file SHA-256 values, recorded run settings, and cache/recovery notes.
- `MISSING_ARTIFACTS.md`: unavailable evidence and interpretation limits.
- `TABLE_AUDIT.md`: comparison of supplied manuscript Tables 1–6 against released artifacts.

## Installation and result recomputation

Python 3.11 or newer is sufficient for the offline analysis. From this directory:

```bash
python3 analysis/recompute.py
```

This writes `accuracy.csv`, `paired_mcnemar.csv`, `transitions.csv`, `private_fact_mention.csv`, `token_usage.csv`, `task_outcomes.csv`, and `verification.json` to `analysis/generated/`. To choose another output directory:

```bash
python3 analysis/recompute.py --output-dir generated_check
```

The script checks 14 × 65 records, all recorded final pluralities, shared gold/options/agent assignments, package checksums, and the independently computed correct counts. Exact two-sided McNemar p-values use the paired discordant tasks. The vote rule is `Counter(votes).most_common(1)[0][0]`: the first encountered option wins a tie. Private Fact Mention uses the owner's public utterances and the original lexical diagnostic: normalized full-fact substring or at least 50% overlap among the fact's unique words. This is a lexical proxy, not a human semantic judgment. EDA's third public stage is **Decide**, while MD/SED's is Round 3.

Check the release for common machine-path, email, and credential patterns with `python3 analysis/check_anonymity.py`.

For inspecting the historical runners, create a separate environment and install their dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-inference.txt
```

The archived runners document how the model calls were made. Their old cache/reference paths are not present in this minimal release, so the supported reproduction command is the offline command above. Running new inference would additionally require provider credentials or a local Qwen vLLM server and the original run reference directories.

## Paper result mapping

The supplied anonymous manuscript was checked against the released records. See `TABLE_AUDIT.md` for table-by-table values and limits. The CSVs below reproduce the paper tables as follows:

| Result type | Released file |
|---|---|
| Three-model MD/EDA/SED accuracy and GPT variants | `analysis/generated/accuracy.csv` |
| Task-paired exact McNemar | `analysis/generated/paired_mcnemar.csv` |
| Initial-to-final group transitions | `analysis/generated/transitions.csv` |
| Private Fact Mention | `analysis/generated/private_fact_mention.csv` |
| Prompt/completion token usage | `analysis/generated/token_usage.csv` |
| Task-level ballots, correctness, and ties | `analysis/generated/task_outcomes.csv` |

Table mapping: Table 1 prompt excerpts (`prompts/sed.json`); Table 2 run settings and protocol scope (`MANIFEST.json`); Table 3 token usage (`token_usage.csv`); Table 4 final accuracy and paired tests (`accuracy.csv`, `paired_mcnemar.csv`); Table 5 endpoint transitions (`transitions.csv`); Table 6 ablation and lexical mention metrics (`accuracy.csv`, `paired_mcnemar.csv`, `private_fact_mention.csv`). `analysis/generated/verification.json` records the offline checks. Table 3 Qwen EDA totals include three separately logged Task 59 parse-recovery API responses; they add 1,527 prompt and 309 completion tokens, but do not add logical task calls.

## Provenance, license, and anonymity

The source dataset is [HiddenBench](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench) and the original [HiddenBench code repository](https://github.com/Yassellee/HiddenBench_ICML) identifies the benchmark as MIT licensed. The original license notice is preserved in `licenses/HIDDENBENCH_MIT.txt` as required for redistributed upstream material. Its copyright attribution identifies the upstream benchmark, **not the authors of this anonymous submission**.

The package omits API keys, account identifiers, emails, absolute machine paths, raw request headers, and submitter identity. Historical runner copies have only the local `.env` path replaced; `MANIFEST.json` records this change. The compact results are derived from saved JSON without changing their votes, public messages, hidden facts, or recorded token counts. Source JSON SHA-256 values permit comparison with a private archive of the original artifacts.
