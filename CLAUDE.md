# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a chatbot evaluation system for SNEHA DIDI, a healthcare chatbot that serves women in low-income urban settlements. The system provides two main components:

1. **DSPy-based RAG optimization** (`/evals` directory) - Uses DSPy framework to optimize prompts for a healthcare chatbot
2. **Comprehensive evaluation runner** (parent directory) - Multi-metric evaluation system with both computational and LLM-as-judge metrics

## Core Architecture

### DSPy Optimization (`/evals`)
- **`rag.py`**: Main DSPy implementation with:
  - `SnehaDidiBotSignature`: Core chatbot signature with healthcare-specific instructions
  - `SnehaBot`: DSPy module that integrates with OpenAI's file search for RAG
  - `AssessCorrectnessSignature`: Correctness evaluation metric
  - COPRO optimizer for prompt optimization
- **`examples.csv`**: Training/test dataset with question-answer pairs in Hindi/Romanized Hindi
- **`optimized_rag.json`**: Saved optimized prompts (auto-generated)

### Evaluation System (Parent Directory)
- **`glific_eval_runner.py`**: CSV-based evaluation with 5 metrics
- **`glific_eval_runner_gsheet.py`**: Google Sheets integration variant
- **Metrics**: Semantic alignment, contextual precision, safety, clarity, completeness

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Required environment variables in .env:
OPENAI_API_KEY=your_key
VECTOR_STORE_ID=vs_xxxxx  # OpenAI vector store for file search
```

## Common Commands

### DSPy Optimization
```bash
cd evals
python rag.py  # Runs optimization if optimized_rag.json doesn't exist, otherwise loads saved model
```

### Evaluation
```bash
# CSV-based evaluation
python glific_eval_runner.py \
  --env-file ~/.env \
  --model gpt-4o-mini \
  --embedding-model text-embedding-3-large \
  --runs 1 \
  --temperature 0.01 \
  --vector-store-id vs_xxxxx \
  --system-prompt prompts/prompt_new.md \
  --analysis-to-sheet

# Google Sheets evaluation
python glific_eval_runner_gsheet.py \
  --env-file ~/.env \
  --sheet-id yyyy \
  --worksheet "Golden Q&A" \
  --service-account ~/sneha-evals.json \
  --output results.xlsx \
  --model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --embedding-model text-embedding-3-large \
  --runs 1 \
  --temperature 0.01 \
  --vector-store-id vs_xxxxx \
  --analysis-to-sheet
```

## Key Technical Details

### Language Handling
The chatbot operates in multiple language modes:
- Hindi (Devanagari script)
- Romanized Hindi 
- English inputs → Romanized Hindi responses

### Vector Store Integration
Uses OpenAI's file search with vector stores for RAG. The `VECTOR_STORE_ID` must be configured for both optimization and evaluation.

### Data Format
Training data in `examples.csv` has columns: `no`, `question`, `reference_answer`

### Optimization Process
DSPy COPRO optimizer fine-tunes prompts using a correctness metric. Optimized prompts are automatically saved and reused.