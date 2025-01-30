# -*- coding: utf-8 -*-
"""
Modified script to use Hugging Face Accelerate for multi-GPU training.
"""

import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# Set environment variables to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%% Load dataset

dataset = load_dataset(
    "text",
    data_files={
        "train": [
            "1hop_ProofsOnly_3shot_nodistractor.txt",
            "1hop_AndIntro_3shot_nodistractor.txt",
            "1hop_AndElim_3shot_nodistractor.txt",
            "1hop_OrIntro_3shot_nodistractor.txt",
            "1hop_OrElim_3shot_nodistractor.txt",
            "1hop_ProofByContra_3shot_nodistractor.txt",
        ]
    },
    sample_by="paragraph",
)

# Split the dataset
train_test_split = dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

combined_dataset = DatasetDict(
    {"train": train_dataset, "validation": test_dataset}
)

#%% Load tokenizer and model

model_name = "gpt2-large"
print("\nModel name: " + model_name + "\n")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#%% Preprocess dataset

def recover_nextline_function(examples):
    for k in examples.keys():
        examples[k] = examples[k].replace("#", "\n")
    return examples


recovered_combined_dataset = combined_dataset.map(recover_nextline_function)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)


tokenized_datasets = recovered_combined_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Convert to PyTorch Datasets
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

#%% Initialize Accelerator

accelerator = Accelerator()  # Automatically detects and sets up GPUs/TPUs
device = accelerator.device

# Move model to device and wrap DataLoader
model = accelerator.prepare(model)
train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

#%% Optimizer and Scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
num_training_steps = len(train_dataloader) * 3  # Assume 3 epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

#%% Training Loop

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
        outputs = model(**batch)  # Forward pass
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader):.4f}")

#%% Save the model

if accelerator.is_main_process:  # Only save on the main process
    model.save_pretrained(f"{model_name}-finetuned-prontoqa")
    tokenizer.save_pretrained(f"{model_name}-finetuned-prontoqa")
    print("\nModel saved successfully.")
