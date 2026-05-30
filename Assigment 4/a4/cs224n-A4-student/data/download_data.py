import json
from datasets import load_dataset
import requests
import re


def download_gsm8k_100():
    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    # Take first 100 problems
    first_100 = dataset.select(range(100))

    # Function to extract answer from GSM8K answer text
    def extract_answer(answer_text):
        """Extract the integer answer after #### from GSM8K answer text"""
        # Look for #### followed by optional whitespace and then digits
        pattern = r'####\s*(\d+)'
        match = re.search(pattern, answer_text)
        if match:
            return int(match.group(1))
        return None

    # Save to JSONL file
    with open("gsm8k_first_100.jsonl", "w") as f:
        for item in first_100:
            # Add answer field with extracted integer
            item_with_answer = dict(item)
            numerical_answer = extract_answer(item['answer'])
            assert numerical_answer is not None
            item_with_answer['numerical_answer'] = numerical_answer
            json.dump(item_with_answer, f)
            f.write("\n")

    print("Saved first 100 GSM8K problems to gsm8k_first_100.jsonl")


def download_alpaca_eval_30():
    """Download and save first 100 samples from Alpaca Eval dataset"""
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json"
    
    # Fetch the JSON data from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Parse the JSON data
    data = response.json()
    
    # Take first 100 samples
    first_100 = data[:30]
    
    # Save to JSONL file
    instructions = set()
    with open("alpaca_eval_first_30.jsonl", "w") as f:
        for item in first_100:

            instruction = item["instruction"]
            assert instruction not in instructions
            entry = {
                "instruction" : instruction
            }

            json.dump(entry, f)
            f.write("\n")
    
    print(f"Saved first 100 Alpaca Eval samples to alpaca_eval_first_100.jsonl")
    print(f"Total samples in dataset: {len(data)}")

if __name__=="__main__":

    #download_gsm8k_100()
    download_alpaca_eval_30()
