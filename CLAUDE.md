# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Evaluation harness for **SNEHA DIDI**, an OpenAI-powered RAG chatbot that answers questions in
Hindi/Romanised Hindi for women in low-income urban settlements on early childhood care, pregnancy, and
government schemes. This repo does not contain the chatbot's serving code — only the prompts under test and
the scripts that score candidate prompts/models against a golden question set.

It computes exactly these metrics, no more:

Referenceless (computational):
1. **Semantic Alignment** — cosine similarity between response embedding and `reference_answer` embedding.
2. **Contextual Precision** — fraction of factual claim-sentences in the response supported (by embedding
   similarity ≥ threshold) by retrieved File Search citation quotes. `None` if there are no citations.

LLM-as-judge (each scored -5..5 by a judge model):
3. **Safety** — avoids harmful content, escalates danger signs.
4. **Clarity** — simple Hindi, minimal English jargon, appropriate literacy level.
5. **Completeness** — thoroughly addresses the concern while staying in scope.
6. **Correctness** — addresses the user's concern (only in `glific_eval_runner.py`; asking for
   clarification or declining is an acceptable "correct" response).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # venv/ already exists at repo root; don't recreate carelessly
pip install -r requirements.txt
```

Secrets: copy an `.env` with `OPENAI_API_KEY` (and `VECTOR_STORE_ID` for `evals/`, see
`evals/.env.example`). Never commit `.env` — it's gitignored.

## Commands

There is no build/lint/test suite — this is a set of standalone eval-runner scripts invoked directly.

Run against a local CSV via the Responses API (RAG, JSON-structured output):
```bash
python3 glific_eval_runner.py \
  --env-file ~/.env \
  --model gpt-4o-mini \
  --embedding-model text-embedding-3-large \
  --runs 1 \
  --temperature 0.01 \
  --vector-store-id vs_xxxxx \
  --system-prompt prompts/prompt_new.md --analysis-to-sheet
```

Run pulling golden Q&A from a Google Sheet instead of CSV (requires a service-account JSON with the sheet
shared to it):
```bash
python3 glific_eval_runner_gsheet.py \
  --env-file ~/.env \
  --sheet-id yyyy \
  --worksheet "Golden Q&A" \
  --service-account ~/sneha-evals.json \
  --output results.xlsx \
  --model gpt-4o-mini --judge-model gpt-4o-mini \
  --embedding-model text-embedding-3-large \
  --runs 1 \
  --temperature 0.01 \
  --vector-store-id vs_xxxxx \
  --analysis-to-sheet
```

Key flags on both runners:
- `--api-mode {responses,assistants,chat}` — `responses` (default, File Search RAG) vs `assistants`
  (strict reuse of an existing `--assistant-id`, never creates/updates one) vs `chat` (plain Chat
  Completions, no retrieval).
- `--expects-json` (glific_eval_runner.py only) — toggles whether the model's Responses-API output is
  parsed as the structured `{answer_lines, citations, follow_up, urgency}` JSON contract vs plain text with
  `file_citation` annotations. Must match the format the system prompt actually asks for.
- `--vector-store-id` — enables File Search grounding; without it, falls back to plain chat completion with
  no citations (so contextual precision will always be `None`).
- Seeds are only applied to judge calls (Chat Completions); the Responses/Assistants APIs don't accept
  `seed`, so answer generation is not perfectly reproducible even at low temperature.

DSPy prompt optimization experiment (separate from the two runners above):
```bash
cd evals && python3 rag.py
```
Uses `dspy.COPRO` to optimize the `SnehaDidiBotSignature` prompt against `evals/examples.csv`
(first 20 rows train / rest test), judged by an LLM-as-judge `correctness` metric. Skips optimization and
loads `optimized_rag.json` if it already exists in the working directory — delete it to force
re-optimization.

## Architecture

**Two near-duplicate runner scripts** (`glific_eval_runner.py` for CSV input, `glific_eval_runner_gsheet.py`
for Google Sheets input) share the same shape but have drifted:
- `glific_eval_runner.py` is the more current one: it externalizes the system prompt to a `--system-prompt`
  file, adds the `judge_correctness` metric, and supports `--expects-json`.
- `glific_eval_runner_gsheet.py` still hardcodes the SNEHA DIDI system prompt inline inside
  `rag_response()`, lacks the correctness metric, and has extra Google Sheets push logic
  (`--push-to-sheet`, `df_to_gsheet`, `text_to_gsheet`). When updating shared logic (metric functions,
  `_extract_response_data`, retry/backoff, Excel formatting), apply the change to **both** files unless the
  behavior is intentionally sheet-specific.

**Flow common to both**: load golden Q&A (`no, question, reference_answer` columns) → for each question, run
N passes through the configured API mode → for each run compute the 4-6 metrics → aggregate into a per-question
`summary` sheet and a per-run `runs` sheet → optionally have an LLM (`--analysis-model`) write an executive
analysis over the aggregated stats → write everything to a timestamped `.xlsx` (sheets: `summary`, `runs`,
`analysis`, `kpis`) plus a `config_<timestamp>.json` snapshot of the run's arguments, under `out/`.

**Response extraction (`OpenAIClient._extract_response_data`)** is the trickiest part of both files: it
branches on `api_mode` (`responses` vs `assistants`) and on `expects_json`. In JSON mode it parses the
model's structured output (`answer_lines`, `citations[].quote`/`source_id`, `follow_up`, `urgency`); in
plain-text mode it instead reads `file_citation` annotations directly off the response. Get this flag wrong
relative to what the system prompt instructs and citations/urgency/follow_up silently come back empty.

**Contextual Precision** (`compute_contextual_precision`) is a claim-level heuristic, not exact matching:
`split_into_sentences` + `is_potential_claim` filter the response into candidate factual sentences
(dropping questions, short sentences, and disclaimer-prefixed lines in Hindi/English), then each is embedded
and compared via cosine similarity against embedded citation quotes; supported if similarity ≥
`--context-precision-threshold`.

**Prompts under test** live in `prompts/` as plain files, referenced by path via `--system-prompt`:
- `prompt_current.md` — plain-text output contract (paired with `--expects-json False` / omitted).
- `prompt_new.md` / `prompt_optimized.md` — JSON output contract (`answer_lines`/`citations`/`follow_up`/
  `urgency`), must be run with `--expects-json true`.
- `prompt_new_nonjson.md` — JSON-contract content but plain-text output, exists to A/B the JSON constraint
  itself.
Check a prompt's own output section before choosing `--expects-json`, since the two families are not
interchangeable.

**`out/`** holds one timestamped subdirectory per run (`<prompt_stem>_<YYYY-MM-DD_HHMM>/`), each with
`results_*.xlsx`, `analysis_*.md`, and `config_*.json`. This is run history/output, not source — read it to
compare prior experiment results, don't hand-edit it.

**`evals/`** is a self-contained DSPy-based experiment (separate dependency surface: `dspy`, `gepa`,
`optuna`) exploring automated prompt optimization rather than manual A/B testing of hand-written prompts in
`prompts/`. It has its own `.env.example` and CSV (`examples.csv`, same schema as the root golden CSVs).
