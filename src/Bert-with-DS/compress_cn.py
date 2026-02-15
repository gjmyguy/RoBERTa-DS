# -*- coding: utf-8 -*-
import torch
import numpy as np
from transformers import BertTokenizerFast, AutoModelForTokenClassification
from train_bert_dsf_cn import BertWithDSF

MODEL_PATH = "RoBERTa_DS_Cn"
RATIO = 0.7

device = "cuda" if torch.cuda.is_available() else "cpu"


PRETRAINED_MODEL = "chinese-roberta-wwm-ext-large"
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL,local_files_only=True)
model = BertWithDSF(PRETRAINED_MODEL, num_labels=2)

model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location="cpu"))
model.to(device)
model.eval()

def compress_batch(texts, ratio=RATIO):

    if not texts:
        return []

    # 批量编码
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    results = []
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits  # (batch, seq_len, 2)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]

    for i, text in enumerate(texts):
        length = (encodings["attention_mask"][i] == 1).sum().item()
        tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][i][:length])
        scores = probs[i, :length].cpu().numpy()

        keep_k = max(1, int(length * ratio))


        selected = np.argsort(scores)[-keep_k:]
        selected = sorted(selected)

        compressed_tokens = [tokens[j] for j in selected]
        compressed_text = tokenizer.convert_tokens_to_string(compressed_tokens)

        results.append(compressed_text)

    return results


# ========== 示例 ==========
if __name__ == "__main__":
    texts = []
    results = compress_batch(texts, ratio=0.75)
    for r in results:
        print(r)
