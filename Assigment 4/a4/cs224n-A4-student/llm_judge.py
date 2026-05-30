import re
import json
import os
from typing import List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model


# You may find these constants useful for structuring the judge's output.
MODEL_E_PREFERED_TAG = "<MODEL_E_BETTER>"
MODEL_F_PREFERED_TAG = "<MODEL_F_BETTER>"
NO_PREFERENCE_FOUND_TAG = "<NO_PREFERENCE_FOUND>"


def load_alpaca_data() -> List[Dict[str, str]]:

    dataset = []
    with open("./data/alpaca_eval_first_30.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)

    return dataset

def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    """
    Construct a prompt for an LLM judge to evaluate two model responses.

    Args:
        query: the question given to the two models (from AlpacaEval)
        response_E: output from model E on query
        response_F: output from model F on query
    Returns:
        Prompt for the LLM judge.
    
    Consider: The judge is an LLM that will output free-form text. How will you 
    design the prompt so that you can reliably determine which response it preferred?
    Your llm_judge_template and extract_llm_judge_preference should work together.
    """
    # TODO complete for question 3b

    pass

def extract_llm_judge_preference(judge_output: str) -> str:
    """
    Extract the judge's preference from its output.

    Args:
        judge_output: the string sampled from the LLM judge.
    Returns:
        A string representing which response the judge preferred.
    
    This function should work in tandem with your llm_judge_template design.
    What if the judge's output is malformed or ambiguous?
    """
    # TODO complete for question 3b

    pass

def run_llm_judge_eval():
    """
    Run the LLM-as-a-judge evaluation comparing models E and F on AlpacaEval data.
    Use model Z as the judge.
    
    For each AlpacaEval instruction, you'll need responses from both models E and F,
    then have the judge compare them.
    
    Remember to save your results (model responses + judge outputs) - you will 
    need them for Parts C and D.
    """
    # TODO complete for question 3b

    pass
    
def plot_model_output_lengths() -> None:
    """
    For Part D: Plot histograms of response lengths for preferred vs. not-preferred outputs.
    """
    # TODO complete for question 3d

    pass

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    #run_llm_judge_eval()
    #plot_model_output_lengths()
