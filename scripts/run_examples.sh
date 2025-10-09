
#!/usr/bin/env bash


# For GSHEET runs
# Responses API with RAG
python glific_eval_runner.py   --env-file .env   --sheet-id SHEET_ID   --worksheet "Sheet1"   --service-account /path/to/service_account.json   --output results.xlsx   --runs 20   --vector-store-id vs_abc123   --push-to-sheet   --analysis-to-sheet

python3 glific_eval_runner.py \
  --env-file ~/vinod.env \
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

  (prompt is embedded in the python file)


# Assistants API (strict; requires existing assistant)
python glific_eval_runner.py   --env-file .env   --sheet-id SHEET_ID   --worksheet "Sheet1"   --service-account /path/to/service_account.json   --output results.xlsx   --runs 20   --api-mode assistants   --assistant-id asst_12345abc   --vector-store-id vs_abc123   --push-to-sheet   --analysis-to-sheet



#For CSV RUNS

