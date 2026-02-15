# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import swanlab
TORCHVIEW_AVAILABLE = True

# ==== 配置 ====
MODEL_NAME = ""
TRAIN_PATH = ""
OUTPUT_DIR = "RoBERTa_DSF_Cn"
NUM_LABELS = 2
BATCH_SIZE = 8
EPOCHS = 10
LR = 4e-5
MAX_LEN = 512


# ==== 自定义数据集 ====
class TokenClsDataset(Dataset):
    def __init__(self, data_pt, max_len=512):
        self.input_ids = data_pt["input_ids"]
        self.labels = data_pt["labels"]
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


data_pt = torch.load(TRAIN_PATH, weights_only=False)
dataset = TokenClsDataset(data_pt)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


class DeepSubmodular(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x * mask).sum() / (mask.sum() + eps)

    def forward(self, hidden, probs, labels=None, soft=False):


        norm_emb = F.normalize(hidden, dim=-1)
        sim_matrix = norm_emb @ norm_emb.T  # (n, n)

        if labels is not None:
            mask = (labels == 1).float()
        else:
            mask = probs if soft else (probs > 0.5).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        coverage = (mask.unsqueeze(0) * sim_matrix).max(dim=1)[0].mean()

        sel = mask.bool()
        n_sel = int(sel.sum().item())
        if n_sel <= 1:
            diversity = torch.tensor(0.0, device=hidden.device)
        else:
            sim_sel = sim_matrix[sel][:, sel]  # (k, k)
            off_diag_sum = sim_sel.sum() - sim_sel.diag().sum()
            off_diag_cnt = n_sel * (n_sel - 1)
            avg_sim = off_diag_sum / (off_diag_cnt + 1e-8)
            diversity = 1.0 - avg_sim

        importance = self._masked_mean(probs, mask)

        reg = mask.mean()

        features = torch.stack([coverage, diversity, importance, reg]).unsqueeze(0)  # (1, 4)

        score = self.mlp(features)        
        score = torch.tanh(score)         
        return score.squeeze()


class BertWithDSF(nn.Module):

    def __init__(self, model_name, num_labels=2, total_steps=4000):
        super().__init__()
        self.bert_cls = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.dsf = DeepSubmodular(self.bert_cls.config.hidden_size)
        self.global_step = 0
        self.total_steps = max(1, int(total_steps))

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        if self.training and labels is not None:
            self.global_step += 1

        kwargs.pop("num_items_in_batch", None)

        outputs = self.bert_cls(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        logits = outputs.logits
        hidden = outputs.hidden_states[-1]  # (B, L, H)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]  # (B, L)

        # === CE loss ===
        ce_loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        # === DS loss ===
        dsf_loss = torch.tensor(0.0, device=hidden.device)
        if labels is not None:
            for i in range(hidden.size(0)):
                valid_mask = labels[i] != -100
                if valid_mask.sum() == 0:
                    continue
                h_i = hidden[i][valid_mask]
                p_i = probs[i][valid_mask]
                l_i = labels[i][valid_mask]
                dsf_loss = dsf_loss + self.dsf(h_i, p_i, l_i)
            dsf_loss = dsf_loss / hidden.size(0)
        dsf_loss = torch.log1p(torch.abs(dsf_loss))

        lambda_dsf = None
        loss = None
        if labels is not None:
            progress = min(1.0, float(self.global_step) / float(self.total_steps))  
            ce = ce_loss.detach()
            ds = dsf_loss.detach()
            lambda_dsf = progress * (ce / (ce + ds + 1e-6))
            loss = ce_loss - lambda_dsf * dsf_loss

            return TokenClassifierOutput(loss=loss, logits=logits)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids

    valid_preds, valid_labels = [], []
    for pred, label in zip(preds, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                valid_preds.append(p_)
                valid_labels.append(l_)

    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average="binary"
    )
    acc = accuracy_score(valid_labels, valid_preds)

    swanlab.log({
        "eval/accuracy": float(acc),
        "eval/precision": float(precision),
        "eval/recall": float(recall),
        "eval/f1": float(f1),
    })
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ==== 初始化 ====
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
total_steps = (len(train_dataset) // BATCH_SIZE) * EPOCHS
model = BertWithDSF(MODEL_NAME, num_labels=NUM_LABELS, total_steps=total_steps)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # evaluation_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    metric_for_best_model="f1",
    warmup_ratio=0.1,
    fp16=True,
    save_safetensors=False
)


# ==== 主程序 ====
if __name__ == "__main__":

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
