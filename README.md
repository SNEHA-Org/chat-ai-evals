
# Glific Evals Package
See README in your previous message for full instructions. This package includes:
- `glific_eval_runner.py`
- `requirements.txt`
- `README.md`
- `.env.example`
- `scripts/run_examples.sh`
- `prompts/prefix.txt`, `prompts/system.txt`


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