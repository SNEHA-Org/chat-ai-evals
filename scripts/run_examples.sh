
#!/usr/bin/env bash

# Responses API with RAG
python glific_eval_runner.py   --env-file .env   --sheet-id SHEET_ID   --worksheet "Sheet1"   --service-account /path/to/service_account.json   --output results.xlsx   --runs 20   --vector-store-id vs_abc123   --push-to-sheet   --analysis-to-sheet

# Assistants API (strict; requires existing assistant)
python glific_eval_runner.py   --env-file .env   --sheet-id SHEET_ID   --worksheet "Sheet1"   --service-account /path/to/service_account.json   --output results.xlsx   --runs 20   --api-mode assistants   --assistant-id asst_12345abc   --vector-store-id vs_abc123   --push-to-sheet   --analysis-to-sheet
