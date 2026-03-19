import pandas as pd
from datasets import Dataset

# load csv
df = pd.read_csv("data/training_data.csv")

dataset = Dataset.from_pandas(df)

print(dataset)
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
def preprocess(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True)
    outputs = tokenizer(example["output"], padding="max_length", truncation=True)

    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=10,
    per_device_train_batch_size=2
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
model.save_pretrained("question_model")
tokenizer.save_pretrained("question_model")

print("Model saved!")

