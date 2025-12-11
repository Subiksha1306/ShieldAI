import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv("data/text/train.csv")

# Encode labels
df["label"] = df["label"].map({"safe":0,"threat":1})

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

dataset = Dataset(df)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="models/text/bert_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=5
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("models/text/bert_model")
tokenizer.save_pretrained("models/text/bert_model")

print("BERT training complete!")
