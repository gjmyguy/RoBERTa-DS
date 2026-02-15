
import json
import time
import torch
import nltk
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import openai  # ✅ 新增 import


nltk.download('punkt', quiet=True)
DAMPING, THRESHOLD, MAX_ITER = 0.85, 1e-5, 50

# ---------- ExtUP ----------
def extup_summary(doc, model, k=None):
    sents = nltk.sent_tokenize(doc)
    if len(sents) <= 3:
        return sents
    if k is None:
        k = max(1, int(len(sents) * 0.7))
    sent_emb = model.encode(sents, convert_to_tensor=True)
    doc_emb = model.encode(doc, convert_to_tensor=True)
    W = util.cos_sim(sent_emb, sent_emb).cpu().numpy()
    p_doc = 1 - util.cos_sim(sent_emb, doc_emb).cpu().numpy().flatten()
    C = np.ones(len(sents))
    for _ in range(MAX_ITER):
        Wn = W / (W.sum(axis=1, keepdims=True) + 1e-9)
        Cn = p_doc * (1 - DAMPING) + DAMPING * np.dot(Wn, C)
        if np.allclose(C, Cn, atol=THRESHOLD):
            break
        C = Cn
    return [sents[i] for i in sorted(np.argsort(C)[-k:])]

# ---------- 生成摘要（使用 GPT 替代硅基 API） ----------
def query_gpt(prompt):
    openai.api_key = "YOUR_OPENAI_API_KEY"
    openai.base_url = 'https://4.0.wokaai.com/v1/'

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

def summarize_case(text, model):
    extracted = extup_summary(text, model)
    prompt = f""""""
    return query_gpt(prompt)  

# ---------- 主函数 ----------
def main():
    model_path = ""
    data_path = ""
    output_path = ""

    model = SentenceTransformer(model_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 生成摘要并统计
    for idx, item in enumerate(tqdm(data, desc="Generating summaries", ncols=100, unit="sample")):
        src = item.get("病人自述", "")
        pred = summarize_case(src, model)
        item["predict_extup"] = pred


        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            mem_max = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            mem_alloc = mem_reserved = mem_max = 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    final_mem_max = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0



if __name__ == "__main__":
    main()