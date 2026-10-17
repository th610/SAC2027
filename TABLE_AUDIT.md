# Manuscript Table Audit

**Status: PASS WITH NOTES.** The supplied anonymous manuscript (`SAC_2027_paper__789.pdf`, SHA-256 recorded in `MANIFEST.json`) was compared with released task records, prompt files, run metadata, and offline recomputation. No model calls were made. The manuscript PDF itself is not included in the release.

| Paper table | Content checked | Recomputed / source value | Result |
|---|---|---|---|
| Table 1 | Selected verbatim SED prompt excerpts and stage labels | Quoted excerpts occur in `prompts/sed.json` and `src/hiddenbench/prompts_v2.json` (System, first-speaker, discussion, final ballot). | Match |
| Table 2 | MD/EDA/SED stages, message scope, guidance; model and generation settings | Run manifests support 65 tasks, assignment seed 42, 3 public messages per agent, 1,265 logical requests per condition; EDA is 2 Exchanges + Decide. Model settings are listed per run in `MANIFEST.json`. | Match with evidence limit below |
| Table 3 | Logged prompt/completion tokens (millions) | GPT MD: not recorded; GPT EDA 1.716/0.072; GPT SED Fresh 1.691/0.076; Qwen MD 0.911/0.058; Qwen EDA **1.610/0.068** including 3 recovery requests; Qwen SED 1.540/0.064; Gemini MD 0.914/0.060; Gemini EDA 1.772/0.077; Gemini SED 1.633/0.070. | Match after recovery addendum |
| Table 4 | Final accuracy and task-paired SED-vs-MD / SED-vs-EDA tests | GPT 17/45/57, Qwen 15/36/60, Gemini 24/27/58. Discordants and p-values match `paired_mcnemar.csv`; MD–SED p-values are Holm-adjusted, SED–EDA values are unadjusted in the table. | Match |
| Table 5 | Initial-to-final correctness transitions | GPT MD 16/0/1/48; EDA 45/1/0/19; SED 53/0/4/8; Fresh 54/0/4/7. Qwen EDA 36/5/0/24. Gemini EDA 23/5/4/33. Order: W→C / C→W / C→C / W→W. | Match |
| Table 6 | GPT SED prompt variants, exact McNemar vs SED, lexical mention counts | SED 57/65, 167/253; −A1 50/65, p=.092, 102/253; −A2 52/65, p=.227, 74/253; EDA 45/65, p=.00418, 132/253. | Match |

## Notes and limits

- Table 3's Qwen EDA total is not merely the sum of the 65 final task files. The original Task 59 parse-failure log records three additional API responses with usage: 1,527 prompt tokens and 309 completion tokens. These are added to `analysis/generated/token_usage.csv` via `results/qwen_eda_recovery_usage.json`; they are not extra logical task calls. The paper's rounded cells remain 1.61M / 0.068M.
- Table 2 says that no GPT completion exceeded 1,024 tokens. The released GPT task files preserve aggregate token usage, not per-request completion usage, so this statement cannot be independently checked from the release. The recorded GPT runners omit an explicit output cap and rely on the API default; do not present the per-request maximum as independently verified by this package.
- Historical GPT MD and original GPT SED prompt/completion usage are not recorded. GPT SED Fresh is a separate 65-task run and is the Table 3 SED token source.
- Table 1 uses selected prompt excerpts; all checked excerpts are present verbatim in the released SED prompt. This does not assert that the table reproduces the full prompt.
- Table 4 uses the original GPT SED run (20 prior artifacts + 45 later artifacts), while the Fresh run is shown separately in Table 5.
- Table 6's lexical measure is a string-overlap proxy, not semantic annotation or evidence-uptake measurement.
