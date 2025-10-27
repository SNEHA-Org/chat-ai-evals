import dspy
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

lm = dspy.LM(
    model='openai/gpt-4o-mini', 
    model_type='responses',
    cache=False, 
    temperature=0.5,
)
dspy.configure(lm=lm)

# load the golden question/answer pairs in dspy.Examples
data = pd.read_csv('goldenset/examples.csv')

examples = []
for index, row in data.iterrows():
    examples.append(
        dspy.Example(
            user_query=row['question'], 
            answer=row['reference_answer']).with_inputs('user_query')
        )
    
train_dataset, test_dataset = examples[:20], examples[20:]

# Sneha bot
class SnehaDidiBotSignature(dspy.Signature):
    """
    You are SNEHA DIDI. You’re a chatbot that responds to queries from women located in low economic urban settlements on early childhood healthcare, pregnancy, government schemes and other related issues.
    The women asking questions are not educated, they might send messages with typos or incomplete information. They need responses in language a 5 year old will understand, using simple and casual words, comprehensive, yet concise - a maximum  of 4-5 lines per response

    Rules of responding:
    1. If the question is unclear, ask 1-2 clarifying questions to better understand their needs/ gather more context. For example, if the user asks any questions related to baby, ask questions like age of the baby, what they want to know etc before finding the answer.
    2. What you share must be verifiable. So strictly limit your responses to what is there in the files in your knowledge base.
       If there is a supporting video link, share that as well. Do not respond from Memory or the Internet or Hallucinate
    3. If you cannot find the answer in the documents, politely respond by saying “I do not have enough information to answer your question” in romanized hindi.
    4. Maintain the style of messaging the user messages you with.
    5. If you get emojis/ generic greeting/ acknowledgment messages in any language(like yes, thank you, ok, ji, G, hi, bye, theek hai), respond according to their message.
    6. You offer 'jaankari', not 'madad'

    Language handling
    1. User can send messages in Hindi (Devanagari script) or Romanised Hindi (English script with transliterated or translated hindi words).
       If the message is in Devanagari script, strictly respond in Hindi using Devanagari script. If its in romanised Hindi, strictly use the same.
       If the ask in English, strictly respond in romanised Hindi.
    2. Avoid using the \"!\" in your messages.
    3. Do not use numbered lists in your response. Bullet point with un-numbered lists.
    4. Topics you **do NOT** respond to and “I do not have enough information to answer your question” - children aged 1, pregnancy sex questions, family planning.
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
                    "max_num_results": 5,  # cap results
                }],
                "tool_choice": {"type": "file_search"},  # force retrieval
                "temperature": self.temperature
            }
        )
        
        return prediction
    

# Optimzer
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


## use the saved optimized prompt if found
file_name = "optimized_rag.json"
if os.path.isfile(file_name):
    print("Loading the optimized prompt")
    optimized_bot = SnehaBot(vector_store_id=os.getenv('VECTOR_STORE_ID'))
    optimized_bot.load(file_name)
else:
    optimizer = dspy.COPRO(prompt_model=lm, metric=metric)

    optimized_bot = optimizer.compile(
        SnehaBot(vector_store_id=os.getenv('VECTOR_STORE_ID')), 
        trainset=train_dataset,
        eval_kwargs = dict(
            num_threads=10,          # Use 16 threads for parallel evaluation
            display_progress=True,   # Show a progress bar during evaluation
            display_table=2        # Do not display the evaluation table
        )
    )

    optimized_bot.save(file_name)


# Evaluate the before and after optimization
evaluator = dspy.Evaluate(devset=test_dataset, metric=metric, num_threads=10, display_progress=True, display_table=2)

before_resultset = evaluator(SnehaBot(vector_store_id=os.getenv('VECTOR_STORE_ID')))

print(before_resultset.items())

after_resultset = evaluator(optimized_bot)

print(after_resultset.items())

print((after_resultset.score - before_resultset.score) / before_resultset.score )