from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
import json
import numpy as np
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser(description="model configuration")

# Add arguments
parser.add_argument("--output_dir", type=str, required=True, help="output directory")
parser.add_argument(
    "--pretrained_model",
    type=str,
    required=False,
    default="gpt2",
    help="Pretrained model name",
)
parser.add_argument("output_dir", type=str, required=True, help="output directory")
parser.add_argument(
    "--num_train_epochs", type=int, required=False, default=5, help="num train epochs"
)
parser.add_argument(
    "--seed", type=int, required=False, default=42, help="num train epochs"
)
parser.add_argument(
    "--lr", type=float, required=False, default=3e-4, help="learning rate"
)
parser.add_argument(
    "--batch_size", type=int, required=False, default=32, help="batch size"
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Load the dataset (for demonstration, we'll use the IMDb dataset)
# Load the dataset
data_files = {
    "train": "datasets/Amazon/MoviesAndTV/train.jsonl",
    "test": "datasets/Amazon/MoviesAndTV/test.jsonl",
    # "train": "datasets/TripAdvisor/train.jsonl",
    # "test": "datasets/TripAdvisor/test.jsonl",
}

with open("user_profiles/amazon_profiles.json") as f:
    profiles_data = json.load(f)

profiles = {}

for i in profiles_data:
    user_id = i["user_id"]
    user_profile = i["profile"]
    profiles[user_id] = user_profile


dataset = load_dataset("json", data_files=data_files)

model_name = "gpt2"

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# convert input to prompt
def convert_to_prompt(example):
    user_id = example["user"]
    example["profile"] = profiles[user_id]
    example[
        "prompt"
    ] = f"User Profile:{example['profile']} Based on my user profile, from a scale of 1 to 5 (1 being the lowest and 5 being the highest), i would give \"{example['title']}\" a rating of"
    return example


dataset = dataset.map(convert_to_prompt)


# Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["prompt"], truncation=True, padding="max_length", max_length=300
    )
    # Scale the labels from [1,5] to [0,1]
    min_val, max_val = 1, 5
    scaled_labels = [
        (label - min_val) / (max_val - min_val) for label in examples["label"]
    ]

    tokenized_output["label"] = scaled_labels
    return tokenized_output


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, device_map="auto"
)
model.config.pad_token_id = model.config.eos_token_id


def compute_metrics(eval_pred):
    scaled_predictions, scaled_labels = eval_pred

    # Assuming this is a regression problem, we'll take the first value from the output logits
    scaled_predictions = scaled_predictions[:, 0]

    # Inverse scaling
    def inverse_scale(values, min_val=1, max_val=5):
        return [s * (max_val - min_val) + min_val for s in values]

    original_predictions = inverse_scale(scaled_predictions)
    original_labels = inverse_scale(scaled_labels)

    # Compute the metrics
    rmse = np.sqrt(
        ((np.array(original_predictions) - np.array(original_labels)) ** 2).mean()
    )
    mae = np.abs(np.array(original_predictions) - np.array(original_labels)).mean()

    return {"rmse": rmse, "mae": mae}

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluations with no improvement after which training will be stopped.
    early_stopping_threshold=0.0  # Minimum improvement to qualify as an improvement.
)

# Define training arguments and set up Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    logging_dir=f"./{args.output_dir}/logs",
    logging_steps=1000,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    learning_rate=args.lr,
    num_train_epochs=args.num_train_epochs,
    output_dir=f"./{args.output_dir}/results",
    remove_unused_columns=True,  # Important!
    seed=args.seed,
    lr_scheduler_type="linear",
    load_best_model_at_end = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# Train the model
trainer.train()
trainer.save_model(f"./{args.output_dir}")
