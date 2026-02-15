import json
from transformers import BertTokenizer,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("")

def clean_text(text: str) -> str:

    return text.replace("[CLS]", "").replace("[SEP]", "").strip()

def convert_and_eval(input_file: str, unieval_file: str):

    data_out = []
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    ratios = []
    for item in data:
        src = item.get("patient", item.get("src", ""))
        ref = item.get("compressed_text", item.get("ref", ""))

        sys = None
        for k, v in item.items():
            if k.startswith("predict"):
                sys = v
                break
        if sys is None:
            sys = item.get("sys", "")
        sys = clean_text(sys)

        data_out.append({"src": src, "ref": ref, "sys": sys})

        if src and sys:
            src_tokens = tokenizer.tokenize(src)
            sys_tokens = tokenizer.tokenize(sys)
            ratio = len(sys_tokens) / len(src_tokens) if len(src_tokens) > 0 else 0
            item["compression_ratio"] = round(ratio, 3)
            ratios.append(ratio)
        else:
            item["compression_ratio"] = None

    with open(unieval_file, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0

