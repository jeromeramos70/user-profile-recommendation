# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import json
from tqdm import tqdm
from accelerate import Accelerator
import random
import numpy as np
import torch


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


device_map = {"": Accelerator().local_process_index}
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token


def create_prompt(reviews):
    nl = "\n"
    return f"""Summarize in a single paragraph using the first person my general movie and tv preferences based on my reviews. Do not mention the word reviews.
Reviews:
{nl.join(reviews)}
Summary:"""


def create_trip_advisor_prompt(reviews):
    nl = "\n"
    return f"""Summarize in a single paragraph summary, using the first person, describing my general trip and hotel preferences based on my reviews. Do not mention the word reviews in the summary.
Reviews:
{nl.join(reviews)}
Summary:"""


with open("./user_profiles/trip_advisor_k_best_features.json") as f:
    data = json.load(f)

profiles = []
for user in tqdm(data, total=len(data)):
    user_id = user["user_id"]
    profile_data = {"user_id": user_id}
    # seperate into positive and negative
    reviews = []
    for feature in user["k_best_data"][:5]:
        max_len = 5 if len(feature["reviews"]) > 5 else len(feature["reviews"])
        for review in feature["reviews"][:max_len]:
            reviews.append(f"- {review}")

    # create prompt
    prompt = create_trip_advisor_prompt(reviews)

    # generate summary
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"].to("cuda")
    mask = inputs["attention_mask"]

    gen_tokens = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=200,
        temperature=0.7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1] :])[0]
    summary = gen_text.strip().replace("\n", "").replace("</s>", "")
    profile_data["profile"] = summary
    profiles.append(profile_data)
with open(f"user_profiles/trip_advisor_profiles_mistral.json", "w") as f:
    json.dump(profiles, f, indent=4)
