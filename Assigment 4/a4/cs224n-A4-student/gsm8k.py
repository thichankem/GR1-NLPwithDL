import re
import json
from typing import Dict
import os

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

INVALID_ANS = "[invalid]"

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in 
# ------------------------------------------- #



def eval_model_on_gsm8k() -> None:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.
    
    See example_usage.py for how to query models and handle responses.
    The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer' fields.
    
    Think about: What metric will you use to evaluate performance? How will you 
    handle cases where the model's output cannot be parsed?
    """
    # TODO complete for question 2bi

    pass



def superior_prompt_template(question: str) -> str:
    """
    Design your own prompt template that outperforms standard_prompt_template on model A.
    
    Args:
        question: gsm8k question.
    Returns:
        Your improved prompt for the model.
    
    Look at standard_prompt_template() to understand the baseline approach. What 
    aspects of how you prompt the model might affect its reasoning or accuracy?
    
    NOTE: Your prompt must still produce output in the "#### <answer>" format
    so that standard_output_extractor() can parse the response.
    """
    # TODO complete for question 2bii

    pass

def eval_model_on_gsm8k_with_improved_prompt() -> None:
    """
    Evaluate model A using your superior_prompt_template.
    """
    # TODO complete for question 2bii

    pass

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    #eval_model_on_gsm8k()
    #eval_model_on_gsm8k_with_improved_prompt()