import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 路径
model_name_or_path = ""
test_file = ""
output_file = ""

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

instruction_template = """"""

# 读取数据
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

if isinstance(test_data, dict):
    test_data = [test_data]

results = []

start_time = time.time()

for idx, item in enumerate(tqdm(test_data, desc="Running", unit="sample")):
    text = item["patient"]

    user_prompt = instruction_template.format(text=text)

    messages = [{"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 生成预测
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=60,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)


    item["predict_qwen"] = response.strip()
    results.append(item)

total_time = time.time() - start_time

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

