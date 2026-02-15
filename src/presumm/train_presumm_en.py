# train_trainer.py
import re, json, torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertTokenizer, BertModel, Trainer, TrainingArguments
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


def split_sentences(text):
    sents = re.split(r"[.!?;]", text)  # Adjusted for English sentence splitting
    return [s.strip() for s in sents if s.strip()]

def build_labels(src_sents, tgt_sents):
    labels = []
    for s in src_sents:
        keep = 0
        for t in tgt_sents:
            overlap = len(set(s) & set(t)) / (len(set(s)) + 1e-6)
            if overlap > 0.7:
                keep = 1
                break
        labels.append(keep)
    return labels

class ExtSumCollator:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])

        # pad cls_positions 和 labels
        cls_positions = [b["cls_positions"] for b in batch]
        labels = [b["labels"] for b in batch]

        cls_positions = pad_sequence(cls_positions, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cls_positions": cls_positions,
            "labels": labels,
        }

class ExtSumDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_sents = split_sentences(sample["patient"])
        tgt_sents = split_sentences(sample["compressed_text"])
        labels = build_labels(src_sents, tgt_sents)

        # 拼接成 [CLS] sent [SEP] ...
        text = ""
        cls_positions = []
        for sent in src_sents:
            cls_positions.append(len(text.split()))
            text += "[CLS] " + sent + " [SEP] "

        encoding = self.tokenizer(
            text, max_length=self.max_len,
            truncation=True, padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "cls_positions": torch.tensor(cls_positions),
            "labels": torch.tensor(labels, dtype=torch.float)
        }


#========= 模型 =========
class ExtSummarizer(nn.Module):
    def __init__(self, pretrained="", hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.classifier = nn.Linear(hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, cls_positions, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, L, H]

        sent_vecs = []
        for i, cls_pos in enumerate(cls_positions):
            vecs = hidden[i, cls_pos, :]  # [Ns, H]
            sent_vecs.append(vecs)
        sent_vecs = torch.stack(sent_vecs)  # [B, Ns, H]

        logits = self.classifier(sent_vecs).squeeze(-1)  # [B, Ns]

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


# ========= 评估指标 =========
def compute_metrics(p):
    preds = (p.predictions > 0).astype(int)  # logits -> binary
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), preds.flatten(), average="binary")
    acc = accuracy_score(labels.flatten(), preds.flatten())
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ========= 训练入口 =========
def train_model(train_file, output_dir, epochs=10, batch_size=16, lr=4e-5):
    tokenizer = BertTokenizer.from_pretrained("")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    dataset = ExtSumDataset(train_data, tokenizer)
    data_collator = ExtSumCollator(tokenizer)
    model = ExtSummarizer(pretrained="")

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Model saved to {output_dir}")

