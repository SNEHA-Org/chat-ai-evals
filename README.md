python3 glific_eval_runner.py \
  --env-file ~/.env \
  --model gpt-4o-mini \
  --embedding-model text-embedding-3-large \
  --runs 1 \
  --temperature 0.01 \
  --vector-store-id vs_67a9de6638888191beb37c06f84e1a88 \
  --system-prompt prompts/prompt_new.md --analysis-to-sheet

python3 glific_eval_runner_gsheet.py \
  --env-file ~/.env \
  --sheet-id yyyy \
  --worksheet "Golden Q&A" \
  --service-account ~/sneha-evals.json \
  --output results.xlsx \
  --model gpt-4o-mini --judge-model gpt-4o-mini\
  --embedding-model text-embedding-3-large \
  --runs 1 \
  --temperature 0.01 \
  --vector-store-id vs_xxxxx \
  --analysis-to-sheet