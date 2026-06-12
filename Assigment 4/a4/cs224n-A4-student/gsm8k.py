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


DATA_PATH = "./data/gsm8k_first_100.jsonl"


def load_gsm8k_data(path: str = DATA_PATH) -> List[Dict]:
    """Load the gsm8k jsonl file into a list of {'question', 'numerical_answer'} dicts."""
    dataset: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def _normalise_answer(answer: str) -> str:
    """Normalise a numerical answer string so that '72', '72.0' and '72.00' compare equal."""
    answer = str(answer).strip().replace(",", "")
    try:
        value = float(answer)
        # Render integers without a trailing '.0'
        if value == int(value):
            return str(int(value))
        return str(value)
    except (ValueError, TypeError):
        return answer


def score_model_on_gsm8k(
    model_id: str,
    dataset: List[Dict],
    prompt_template=standard_prompt_template,
) -> float:
    """
    Run a string-match benchmark of a single model over the dataset.

    For every question we build a prompt, query the model, extract the numerical
    answer with `standard_output_extractor`, and compare it (after normalisation)
    to the gold `numerical_answer`. Unparseable outputs count as wrong.

    Returns the accuracy (fraction of correct answers) in [0, 1].
    """
    num_correct = 0

    for example in tqdm(dataset, desc=f"Evaluating model {model_id}"):
        question = example["question"]
        gold = _normalise_answer(example["numerical_answer"])

        prompt = prompt_template(question)
        query = Query(turns=[{"user": prompt}])

        response: QueryResponse = query_model(model_id=model_id, query=query)

        predicted = _normalise_answer(standard_output_extractor(response.text))

        if predicted != INVALID_ANS and predicted == gold:
            num_correct += 1

    accuracy = num_correct / len(dataset)
    return accuracy


def eval_model_on_gsm8k() -> None:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.

    Metric: exact string match accuracy between the extracted numerical answer and
    the gold answer. Outputs that cannot be parsed are treated as incorrect.
    """
    dataset = load_gsm8k_data()

    results: Dict[str, float] = {}
    for model_id in ["A", "B"]:
        accuracy = score_model_on_gsm8k(model_id, dataset, standard_prompt_template)
        results[model_id] = accuracy
        print(f"Model {model_id}: accuracy = {accuracy:.3f} ({int(accuracy * len(dataset))}/{len(dataset)})")

    return results


def superior_prompt_template(question: str) -> str:
    """
    An improved prompt for model A.

    Versus the standard prompt, this version (1) removes the artificial "two or
    fewer steps of reasoning" constraint and instead invites explicit
    step-by-step chain-of-thought, and (2) gives a one-shot worked example so the
    model both reasons more carefully and reliably emits the required
    "#### <answer>" format that `standard_output_extractor` parses.
    """
    prompt = f"""You are a careful math tutor. Solve the word problem by reasoning step by step.
Work through the arithmetic explicitly, then state the final numerical answer.
The very last line of your response MUST be exactly in the format "#### <numerical_answer>",
where <numerical_answer> is a single number with no units, words, or extra symbols.

Example
Problem: A shop sold 3 apples on Monday and twice as many on Tuesday. How many apples in total?
Reasoning: Tuesday it sold 2 x 3 = 6 apples. Total = 3 + 6 = 9 apples.
#### 9

Now solve this problem.
Problem: {question}
""".strip()

    return prompt


def eval_model_on_gsm8k_with_improved_prompt() -> None:
    """
    Evaluate model A using your superior_prompt_template and compare against the
    standard prompt, saving a bar chart of the two accuracies.
    """
    dataset = load_gsm8k_data()

    standard_acc = score_model_on_gsm8k("A", dataset, standard_prompt_template)
    improved_acc = score_model_on_gsm8k("A", dataset, superior_prompt_template)

    print(f"Model A standard prompt: accuracy = {standard_acc:.3f}")
    print(f"Model A improved prompt: accuracy = {improved_acc:.3f}")

    # Bar chart of the two prompts for model A.
    plt.figure(figsize=(5, 5))
    plt.bar(["Standard prompt", "Improved prompt"], [standard_acc, improved_acc],
            color=["#888888", "#2c7fb8"])
    plt.ylabel("GSM8K accuracy")
    plt.ylim(0, 1)
    plt.title("Model A: standard vs. improved prompt")
    for i, acc in enumerate([standard_acc, improved_acc]):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig("gsm8k_prompt_comparison.png")
    plt.close()

    return {"standard": standard_acc, "improved": improved_acc}


if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    #eval_model_on_gsm8k()
    #eval_model_on_gsm8k_with_improved_prompt()