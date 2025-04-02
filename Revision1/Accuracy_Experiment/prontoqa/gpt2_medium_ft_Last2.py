# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:10:53 2025

@author: xiaoyenche
"""

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

num_proc = 1

#%% Load dataset

from datasets import load_dataset, DatasetDict
dataset = load_dataset("text", data_files={"train": ["1hop_ProofsOnly_3shot_nodistractor.txt", "1hop_AndIntro_3shot_nodistractor.txt", "1hop_AndElim_3shot_nodistractor.txt", "1hop_OrIntro_3shot_nodistractor.txt", "1hop_OrElim_3shot_nodistractor.txt", "1hop_ProofByContra_3shot_nodistractor.txt"]}, sample_by="paragraph")

# Split the dataset into train and validation sets (90% train, 10% test)
train_test_split = dataset["train"].train_test_split(test_size=0.1)

# Access the splits
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Check the split sizes
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Combine the splits into a single DatasetDict
combined_dataset = DatasetDict({
    "train": train_dataset,
    "validation": test_dataset
})

# Check the combined dataset
print(combined_dataset)

#%% Load LM

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2-medium"
print("\n" + "Model name: " + model_name + "\n")

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Set unfreeze layers
if model_name in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]:
    tounfreeze= set(["layers.23", "layers.22"])
elif model_name in ["bigscience/bloomz-1b1"]:
    tounfreeze= set(["transformer.h.23", "transformer.h.22"])
elif model_name in ["meta-llama/Llama-3.2-1B-Instruct"]:
    tounfreeze= set(["model.layers.15", "model.layers.14"])
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
elif model_name in ["tiiuae/Falcon3-1B-Base"]:
    tounfreeze= set(["model.layers.17", "model.layers.16"])
elif model_name in ["Qwen/Qwen2.5-1.5B-Instruct"]:
    tounfreeze= set(["model.layers.27", "model.layers.26"])
elif model_name in ["gpt2"]:
    tounfreeze= set(["transformer.h.11", "transformer.h.10"])
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
elif model_name in ["gpt2-medium"]:
    tounfreeze= set(["transformer.h.23", "transformer.h.22"])
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
elif model_name in ["gpt2-large"]:
    tounfreeze= set(["transformer.h.35", "transformer.h.34"])
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
elif model_name in ["gpt2-xl"]:
    tounfreeze= set(["transformer.h.47", "transformer.h.46"])
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
else:
    raise ValueError("Invalid model name.")
    
for m in model.modules():
    for name, params in m.named_parameters():
        params.requires_grad = False
for m in model.modules():
    for name, params in m.named_parameters():
        if any(prefix for prefix in tounfreeze if name.startswith(prefix)):
            params.requires_grad = True
        print(name, params.requires_grad)
# https://github.com/huggingface/peft/issues/137
model.enable_input_require_grads()

#%% Preprocess
"""
Training
"""


#%% Preprocess

def recover_nextline_function(examples):
    for k in examples.keys():
        examples[k] = examples[k].replace("#", "\n")
    return examples

recovered_combined_dataset = combined_dataset.map(recover_nextline_function)
print(combined_dataset["train"][0])
print(recovered_combined_dataset["train"][0])

# # Tokenize and calculate token lengths
# token_lengths = [len(tokenizer.encode(ex, add_special_tokens=True)) for ex in recovered_combined_dataset["train"]]

token_lengths = [
    len(tokenizer.encode(ex["text"], add_special_tokens=True))
    for ex in recovered_combined_dataset["train"]
]

# Find maximum number of tokens
max_num_tokens = max(token_lengths)

# Print result
print("Maximum number of tokens:", max_num_tokens)

if tokenizer.model_max_length < max_num_tokens:
    seq_length = tokenizer.model_max_length
else:
    seq_length = max_num_tokens

def tokenize_function(examples):
    # result = train_tokenizer(next(iter(examples)), max_length=256, padding='max_length', truncation=True)
    result = tokenizer(examples["text"], max_length=seq_length, padding='max_length', truncation=True) #  characters
    # print(result)
    result["labels"] = result["input_ids"].copy()
    return result
# tokenized_datasets = combined_dataset.map(tokenize_function, batched=False, remove_columns=["text"])# , num_proc=num_proc
lm_datasets = recovered_combined_dataset.map(tokenize_function, batch_size=1, remove_columns=["text"])# , num_proc=num_proc
print("lm_datasets")
print(lm_datasets)
print("len(lm_datasets)")
print(len(lm_datasets))

#%% Training setting

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    f"{model_name}-finetuned-Last2-prontoqa",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    num_train_epochs=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end = True,
    save_total_limit=2, # Save the best and the last checkpoint
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()

#%% After training finishes, you can access the best model.
best_model = trainer.model

# Access the evaluation results (validation loss and other metrics).
eval_results = trainer.evaluate()

# Print out validation loss and other metrics
print(f"Validation Loss of the best model: {eval_results['eval_loss']}")