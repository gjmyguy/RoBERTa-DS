# eval_summary.py
import json
from utils import convert_to_json
from metric.evaluator import get_evaluator
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)


def eval_summarization(input_json, print_result=True):

    with open(input_json, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    src_list = [d["src"] for d in data_json]
    ref_list = [d["ref"] for d in data_json]
    output_list = [d["sys"] for d in data_json]


    data = convert_to_json(output_list=output_list,
                           src_list=src_list,
                           ref_list=ref_list)


    task = "summarization"
    evaluator = get_evaluator(task)


    eval_scores = evaluator.evaluate(data, print_result=print_result)

    return eval_scores

