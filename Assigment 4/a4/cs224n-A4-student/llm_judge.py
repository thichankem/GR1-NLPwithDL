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

RESULTS_PATH = "./judge_results.json"


def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    """
    Construct a prompt for an LLM judge to compare two model responses.

    We give the judge the user instruction and both candidate answers (labelled E
    and F), ask it to reason briefly, and force it to end with exactly one of the
    sentinel tags so that `extract_llm_judge_preference` can parse the decision
    deterministically.
    """
    prompt = f"""You are an impartial judge evaluating the quality of two AI assistant responses to a user instruction.
Compare the two responses on helpfulness, relevance, accuracy, and level of detail. Do not let the order
of the responses or their length bias your decision. Be objective.

[User instruction]
{query}

[Response E]
{response_E}

[Response F]
{response_F}

First, briefly explain your reasoning in one or two sentences. Then, on the final line of your output,
write exactly ONE of the following tags to indicate which response is better:
{MODEL_E_PREFERED_TAG}  (if Response E is better)
{MODEL_F_PREFERED_TAG}  (if Response F is better)
{NO_PREFERENCE_FOUND_TAG}  (only if they are genuinely equal in quality)
""".strip()

    return prompt


def extract_llm_judge_preference(judge_output: str) -> str:
    """
    Extract the judge's preference from its output.

    Returns one of MODEL_E_PREFERED_TAG, MODEL_F_PREFERED_TAG or
    NO_PREFERENCE_FOUND_TAG. If the output is malformed / ambiguous (e.g. it
    mentions both tags, or neither), we fall back to NO_PREFERENCE_FOUND_TAG.
    """
    has_e = MODEL_E_PREFERED_TAG in judge_output
    has_f = MODEL_F_PREFERED_TAG in judge_output

    if has_e and not has_f:
        return MODEL_E_PREFERED_TAG
    if has_f and not has_e:
        return MODEL_F_PREFERED_TAG
    return NO_PREFERENCE_FOUND_TAG


def run_llm_judge_eval(judge_model_id: str = "Z") -> List[Dict]:
    """
    Run the LLM-as-a-judge evaluation comparing models E and F on AlpacaEval,
    using model Z as the judge. Saves all responses + judge outputs to disk.
    """
    dataset = load_alpaca_data()

    records: List[Dict] = []
    e_wins = f_wins = ties = 0

    for example in tqdm(dataset, desc="LLM judge eval"):
        query_text = example.get("instruction") or example.get("query") or example["prompt"]

        # Sample a response from each model under test.
        resp_e = query_model(model_id="E", query=Query(turns=[{"user": query_text}])).text
        resp_f = query_model(model_id="F", query=Query(turns=[{"user": query_text}])).text

        # Ask the judge to compare them.
        judge_prompt = llm_judge_template(query_text, resp_e, resp_f)
        judge_output = query_model(
            model_id=judge_model_id, query=Query(turns=[{"user": judge_prompt}])
        ).text

        preference = extract_llm_judge_preference(judge_output)
        if preference == MODEL_E_PREFERED_TAG:
            e_wins += 1
        elif preference == MODEL_F_PREFERED_TAG:
            f_wins += 1
        else:
            ties += 1

        records.append({
            "query": query_text,
            "response_E": resp_e,
            "response_F": resp_f,
            "judge_output": judge_output,
            "preference": preference,
        })

    n = len(dataset)
    print(f"Model E win rate: {e_wins / n:.3f} ({e_wins}/{n})")
    print(f"Model F win rate: {f_wins / n:.3f} ({f_wins}/{n})")
    print(f"Ties / no preference: {ties}/{n}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Bar chart of win counts.
    plt.figure(figsize=(5, 5))
    plt.bar(["Model E", "Model F", "Tie"], [e_wins, f_wins, ties],
            color=["#2c7fb8", "#d95f0e", "#999999"])
    plt.ylabel("Number of wins (out of %d)" % n)
    plt.title("LLM-as-judge: E vs F")
    plt.tight_layout()
    plt.savefig("llm_judge_winrate.png")
    plt.close()

    return records


def plot_model_output_lengths() -> None:
    """
    For Part D: histogram of the lengths of the judge-preferred outputs vs the
    judge-not-preferred outputs (one datapoint per problem), on the same axis.
    Length is measured in characters.
    """
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    preferred_lengths: List[int] = []
    not_preferred_lengths: List[int] = []

    for r in records:
        len_e, len_f = len(r["response_E"]), len(r["response_F"])
        if r["preference"] == MODEL_E_PREFERED_TAG:
            preferred_lengths.append(len_e)
            not_preferred_lengths.append(len_f)
        elif r["preference"] == MODEL_F_PREFERED_TAG:
            preferred_lengths.append(len_f)
            not_preferred_lengths.append(len_e)
        # ties contribute no clear preferred/not-preferred pair, so we skip them

    plt.figure(figsize=(7, 5))
    bins = 15
    plt.hist(preferred_lengths, bins=bins, alpha=0.6, label="Preferred output", color="#2c7fb8")
    plt.hist(not_preferred_lengths, bins=bins, alpha=0.6, label="Not-preferred output", color="#d95f0e")
    plt.xlabel("Response length (characters)")
    plt.ylabel("Count")
    plt.title("Lengths of preferred vs. not-preferred outputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("llm_judge_length_histogram.png")
    plt.close()


if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    #run_llm_judge_eval()
    #plot_model_output_lengths()
