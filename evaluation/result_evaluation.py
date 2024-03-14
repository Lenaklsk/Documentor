import json
import os

from config import EVALUATION_RESULTS_JUNG_UND_NAIV_DIR, EVALUATION_RESULTS_WIKIPEDIA_DIR

model_simple_names = [
    "mistral_instruct_v01",
    "mistral_instruct_v02",
    "llama2_chat",
    "neural_chat",
]

results = {}

for model_name in model_simple_names:
    with open(os.path.join(EVALUATION_RESULTS_WIKIPEDIA_DIR, model_name + "_results.json"), 'r') as file:
        eval_results = json.load(file)
    correctness_scores = [r["correctness_result"]["score"] for r in eval_results]
    faithfulness_scores = [r["faith_result"]["score"] for r in eval_results]

    avg_correctness_score = float('%.2f' % float(sum(correctness_scores) / len(correctness_scores)))
    faith_good_percentage = float('%.2f' % float(sum(faithfulness_scores) / len(faithfulness_scores)))

    results[model_name] = {"corr": avg_correctness_score, "faith": faith_good_percentage}

print(results)
print()
