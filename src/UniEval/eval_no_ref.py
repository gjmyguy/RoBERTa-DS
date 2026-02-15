import json
from utils import convert_to_json
from metric.evaluator import get_evaluator

task = 'summarization'

with open("", "r", encoding="utf-8") as f:
    dataset = json.load(f)

src_list = [item["patient"] for item in dataset]


output_list = [item["predict_kimi"] for item in dataset]

ref_list = [None] * len(dataset)

data = convert_to_json(output_list=output_list, src_list=src_list, ref_list=ref_list)

evaluator = get_evaluator(task)

eval_scores = evaluator.evaluate(
    data, 
    dims=["coherence", "consistency", "fluency"], 
    overall=True, 
    print_result=True
)


