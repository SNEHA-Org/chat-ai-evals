## What this does ?
Evaluator with the following metrics ONLY:

Referenceless (computational) metrics
1) Semantic Alignment (cosine similarity): response vs golden answer (reference_answer column).
2) Contextual Precision: fraction of factual claims in the response that are supported by retrieved context
   (citations/quotes from File Search via Vector Store). If no citations are present, this metric is None.

LLM-as-judge (referenced) metrics
3) Safety (-5..5): avoids harmful content; escalates danger signs appropriately.
4) Clarity (-5..5): easy to understand at appropriate literacy levels; simple Hindi; minimal English jargon.
5) Completeness (-5..5): thoroughly addresses the user's concern while staying in scope.

Notes
- Input: CSV file with columns: no, question, reference_answer
- Runs N passes per question (default 20) using either:
    (A) Responses API (default). If --vector-store-id is provided, uses File Search grounding and extracts citations.
    (B) Assistants API (--api-mode assistants). STRICT reuse: requires --assistant-id and will NOT create/update.
      In Assistants mode the constructed prompts are ignored; it sends ONLY 'question' as the user message.
- Outputs Excel with timestamped directory structure.
- Analysis tab (optional) summarizes these metrics only.


## USAGE: 
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