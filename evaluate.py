from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, Dataset
import json
import numpy as np
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser(description="model configuration")

# Add arguments
parser.add_argument(
    "--pretrained_model",
    type=str,
    required=False,
    default="gpt2",
    help="Pretrained model name",
)
parser.add_argument(
    "--sampling_file", type=str, required=False, help="sampling"
)
parser.add_argument(
    "--profiles", type=str, required=True, help="profiles"
)
parser.add_argument(
    "--output", type=str, required=False, help="output"
)
parser.add_argument(
    "--add_profile", type=str, required=False, help="add to profile"
)
parser.add_argument(
    "--seed", type=int, required=False, default=42, help="seed"
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load the dataset
data_files = {
    # "test": "datasets/TripAdvisor/test.jsonl",
    "test": "datasets/Amazon/MoviesAndTV/test.jsonl"
}

with open(args.profiles) as f:
    profiles_data = json.load(f)

profiles = {}

for i in profiles_data:
    user_id = i["user_id"]
    user_profile = i["profile"]
    profiles[user_id] = user_profile


dataset = load_dataset("json", data_files=data_files)
model_name = "out/llama-out"
tokenizer = AutoTokenizer.from_pretrained('gpt2', device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# convert input to prompt
def convert_to_prompt(example):
    user_id = example["user"]
    example["profile"] = profiles[user_id]
    example[
        "prompt"
    ] = f"User Profile: {example['profile']} Based on my user profile, from a scale of 1 to 5 (1 being the lowest and 5 being the highest), i would give \"{example['title']}\" a rating of"
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


def compute_scaled_metrics(eval_pred):
    # print([str(id) for id in eval_pred.users])
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

    users = trainer.eval_dataset['user']
    items = trainer.eval_dataset['item']
    output = [{'user': user, 'item': item, 'predicted_rating': pred, 'true_rating': actual} for (user,item,pred,actual) in zip(users,items, original_predictions, original_labels) ]
    if args.output:
        with open(args.output, 'w') as outfile:
            for entry in output:
                json.dump(entry, outfile)
                outfile.write('\n')
    return {"rmse": rmse, "mae": mae}

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_scaled_metrics,
)
print('saving to:', args.output)
# Evaluate the model
results = trainer.evaluate()
print(results)
