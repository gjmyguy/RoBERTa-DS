# -*- coding: utf-8 -*-
import json
import time
from tqdm import tqdm
from compress_cn import compress_batch 
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = ""
OUTPUT_FILE = ""
TEXT_KEY = "patient"
OUTPUT_KEY = "predict_my"
RATIO = 0.6
BATCH_SIZE = 4

QWEN_MODEL = ""
DEVICE = "cuda"

print("🔹 Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="bfloat16",
)
model.eval()


def polish_with_qwen_batch(originals, drafts):

    instruction = """"""

    results = []
    for original, draft in zip(originals, drafts):
        if not draft.strip():
            results.append("")
            continue

        text = f"【Original Text】：{original}\n\n【Draft】：{draft}"
        messages = [{"role": "user", "content": f"{instruction}{text}"}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
            
            # do_sample=False
        )

        cut = model_inputs.input_ids.shape[1]
        decoded = tokenizer.decode(generated_ids[0, cut:], skip_special_tokens=True).strip()
        results.append(decoded)

    return results


def main():
    start_time = time.time()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing", unit="batch"):
        batch_records = data[i:i + BATCH_SIZE]
        texts = [str(r.get(TEXT_KEY, "")) for r in batch_records]

        try:
            drafts = compress_batch(texts, ratio=RATIO)
        except Exception as e:
            print(f"⚠️ 出错: {e}")
            drafts = [""] * len(texts)

        drafts_clean = [d.replace("[CLS]", "").replace("[SEP]", "").strip() for d in drafts]

        polished_list = polish_with_qwen_batch(texts, drafts_clean)

        for r, polished in zip(batch_records, polished_list):
            r[OUTPUT_KEY] = polished

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    elapsed_time = time.time() - start_time



if __name__ == "__main__":
    main()
