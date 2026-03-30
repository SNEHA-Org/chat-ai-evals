#!/usr/bin/env python3
"""
Enhanced DSPy optimization script with websocket event support
This version emits progress updates for real-time UI feedback
"""

import asyncio
import dspy
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Any

load_dotenv()

lm = dspy.LM(
    model='openai/gpt-4o-mini', 
    model_type='responses',
    cache=False, 
    temperature=0.5,
)
dspy.configure(lm=lm)

class SnehaDidiBotSignature(dspy.Signature):
    """
    You are SNEHA DIDI. You're a chatbot that responds to queries from women located in low economic urban settlements on early childhood healthcare, pregnancy, government schemes and other related issues.
    The women asking questions are not educated, they might send messages with typos or incomplete information. They need responses in language a 5 year old will understand, using simple and casual words, comprehensive, yet concise - a maximum  of 4-5 lines per response

    Rules of responding:
    1. If the question is unclear, ask 1-2 clarifying questions to better understand their needs/ gather more context. For example, if the user asks any questions related to baby, ask questions like age of the baby, what they want to know etc before finding the answer.
    2. What you share must be verifiable. So strictly limit your responses to what is there in the files in your knowledge base.
       If there is a supporting video link, share that as well. Do not respond from Memory or the Internet or Hallucinate
    3. If you cannot find the answer in the documents, politely respond by saying "I do not have enough information to answer your question" in romanized hindi.
    4. Maintain the style of messaging the user messages you with.
    5. If you get emojis/ generic greeting/ acknowledgment messages in any language(like yes, thank you, ok, ji, G, hi, bye, theek hai), respond according to their message.
    6. You offer 'jaankari', not 'madad'

    Language handling
    1. User can send messages in Hindi (Devanagari script) or Romanised Hindi (English script with transliterated or translated hindi words).
       If the message is in Devanagari script, strictly respond in Hindi using Devanagari script. If its in romanised Hindi, strictly use the same.
       If the ask in English, strictly respond in romanised Hindi.
    2. Avoid using the "!" in your messages.
    3. Do not use numbered lists in your response. Bullet point with un-numbered lists.
    4. Topics you **do NOT** respond to and "I do not have enough information to answer your question" - children aged 1, pregnancy sex questions, family planning.
    5. Use the hindi word for baby, iron (the metal in our blood stream).
    6. Use simpler words than unclear, specific, cost, growth, meals, healthy, tummy, facilities, seasonal, hydrated, bleeding, fever, guava, organs, structure, placenta, junk, variety, mashed, quantity, soft, mackerel, salmon, mercury, absorb, legumes, citrus - use colloquial words the women can understand.
    """
    user_query: str = dspy.InputField()
    bot_answer: str = dspy.OutputField()

class SnehaBot(dspy.Module):
    def __init__(self, vector_store_id: str, temperature: float = 0.01):
        super().__init__()
        self.generate_answer = dspy.Predict(SnehaDidiBotSignature)
        self.vector_store_id = vector_store_id
        self.temperature = temperature

    def forward(self, user_query: str) -> dspy.Prediction:
        prediction = self.generate_answer(
            user_query=user_query,  
            config={
                "tools":[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": 5,
                }],
                "tool_choice": {"type": "file_search"},
                "temperature": self.temperature
            }
        )
        return prediction

class AssessCorrectnessSignature(dspy.Signature):
    """Judge the bot answer for correctness on a scale of -5 to 5"""
    user_query: str = dspy.InputField()
    golden_answer: str = dspy.InputField()
    bot_answer: str = dspy.InputField()
    correctness: float = dspy.OutputField()

def metric(golden_example: dspy.Example, response, trace = None):
    assess = dspy.Predict(AssessCorrectnessSignature)
    score = assess(
        user_query=response, 
        golden_answer=golden_example.answer,
        bot_answer=response.bot_answer
    )
    return score.correctness

async def run_optimization_with_events(
    progress_callback: Callable[[str, int, Dict[str, Any]], None],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run DSPy optimization with progress callbacks for real-time updates
    """
    
    try:
        # Step 1: Load dataset
        await progress_callback("Loading golden dataset...", 10)
        data = pd.read_csv('../data/examples/examples.csv')
        
        examples = []
        for index, row in data.iterrows():
            examples.append(
                dspy.Example(
                    user_query=row['question'], 
                    answer=row['reference_answer']).with_inputs('user_query')
            )
        
        train_dataset, test_dataset = examples[:20], examples[20:]
        
        # Step 2: Initialize models
        await progress_callback("Initializing DSPy models...", 20)
        vector_store_id = os.getenv('VECTOR_STORE_ID')
        
        if not vector_store_id:
            raise ValueError("VECTOR_STORE_ID not found in environment variables")
        
        # Step 3: Check if optimized model exists
        await progress_callback("Checking for existing optimized model...", 30)
        file_name = "optimized_rag.json"
        
        if os.path.isfile(file_name):
            await progress_callback("Loading existing optimized model...", 40)
            optimized_bot = SnehaBot(vector_store_id=vector_store_id)
            optimized_bot.load(file_name)
        else:
            # Step 4: Run optimization
            await progress_callback("Setting up COPRO optimizer...", 40)
            optimizer = dspy.COPRO(prompt_model=lm, metric=metric)
            
            await progress_callback("Running optimization iterations...", 50)
            optimized_bot = optimizer.compile(
                SnehaBot(vector_store_id=vector_store_id), 
                trainset=train_dataset,
                eval_kwargs = dict(
                    num_threads=10,
                    display_progress=True,
                    display_table=2
                )
            )
            
            await progress_callback("Saving optimized model...", 70)
            optimized_bot.save(file_name)
        
        # Step 5: Evaluate models
        await progress_callback("Evaluating original model...", 80)
        evaluator = dspy.Evaluate(devset=test_dataset, metric=metric, num_threads=10, display_progress=True, display_table=2)
        
        original_bot = SnehaBot(vector_store_id=vector_store_id)
        before_resultset = evaluator(original_bot)
        
        await progress_callback("Evaluating optimized model...", 90)
        after_resultset = evaluator(optimized_bot)
        
        # Calculate improvement
        improvement = ((after_resultset.score - before_resultset.score) / before_resultset.score) * 100
        
        # Extract optimized prompt - get the full docstring
        optimized_prompt = optimized_bot.generate_answer.signature.__doc__
        
        # Step 6: Save optimized prompt to file
        await progress_callback("Saving optimized prompt...", 95)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        await save_optimized_prompt_to_file(optimized_prompt, timestamp)
        
        # Prepare results
        results = {
            "original_score": float(before_resultset.score),
            "optimized_score": float(after_resultset.score),
            "improvement": float(improvement),
            "optimization_iterations": 15,  # This would come from the optimizer
            "total_examples": len(examples),
            "optimized_prompt": optimized_prompt,
            "timestamp": timestamp
        }
        
        await progress_callback("Optimization completed successfully!", 100, results)
        return results
        
    except Exception as e:
        await progress_callback(f"Error: {str(e)}", 0)
        raise e

async def save_optimized_prompt_to_file(optimized_prompt: str, timestamp: str):
    """Save optimized prompt to a markdown file with timestamp"""
    try:
        prompts_dir = Path("../data/prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        filename = f"optimized_{timestamp}.md"
        filepath = prompts_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Only write the prompt content
            f.write(optimized_prompt if optimized_prompt else "")
        
        print(f"Saved optimized prompt to: {filepath}")
        
    except Exception as e:
        print(f"Failed to save optimized prompt: {e}")

if __name__ == "__main__":
    # For standalone testing
    async def test_progress_callback(step: str, progress: int, data: Dict[str, Any] = None):
        print(f"[{progress}%] {step}")
        if data:
            print(f"  Data: {data}")
    
    async def main():
        try:
            results = await run_optimization_with_events(test_progress_callback)
            print("Final Results:", results)
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())