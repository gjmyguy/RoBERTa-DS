# predict_presumm.py
import json, math, torch, time
from safetensors.torch import load_file
from transformers import BertTokenizer
from train_presumm_en import ExtSummarizer, split_sentences


def summarize(model, tokenizer, text, ratio=0.7, device="cuda", max_len=512):
    sents = split_sentences(text)
    text_join = ""
    cls_positions = []
    for sent in sents:
        cls_positions.append(len(text_join.split()))
        text_join += "[CLS] " + sent + " [SEP] "

    enc = tokenizer(
        text_join, return_tensors="pt",
        truncation=True, padding=True,
        max_length=max_len
    ).to(device)
    cls_pos = torch.tensor([cls_positions]).to(device)

    with torch.no_grad():
        outputs = model(enc["input_ids"], enc["attention_mask"], cls_pos)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    if isinstance(probs, float):  # Single sentence case
        probs = [probs]

    N = len(probs)
    K = max(1, math.floor(N * ratio))
    topk = sorted(range(N), key=lambda i: probs[i], reverse=True)[:K]
    selected = [sents[i] for i in sorted(topk)]
    return "".join(selected)


def run_predict(test_file, model_dir, out_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("")  # Use the English tokenizer

    # === Load safetensors model ===
    model = ExtSummarizer(pretrained="d")  # Updated for the English model
    state_dict = load_file(f"{model_dir}/model.safetensors")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # === Prediction ===
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    start_time = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for idx, sample in enumerate(test_data):
        # Use "patient" field for English dataset
        pred = summarize(model, tokenizer, sample["patient"], ratio=0.71, device=device)
        sample["predict_presumm"] = pred

        # Time statistics
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)

      

    total_time = time.time() - start_time

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Predictions saved to {out_file}")
    print(f"⏱️ Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📊 Average time per sample: {total_time/len(test_data):.2f} seconds")

    if device == "cuda":
        final_mem_max = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"🖥️ Peak memory usage during inference: {final_mem_max:.1f} MB")
