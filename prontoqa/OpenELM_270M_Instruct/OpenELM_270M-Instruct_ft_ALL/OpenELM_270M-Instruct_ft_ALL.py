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

model_name = "apple/OpenELM-270M-Instruct"
print("\n" + "Model name: " + model_name + "\n")

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="cuda", token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, token=access_token)

#%% Preprocess

def recover_nextline_function(examples):
    for k in examples.keys():
        examples[k] = examples[k].replace("#", "\n")
    return examples

recovered_combined_dataset = combined_dataset.map(recover_nextline_function)
print(combined_dataset["train"][0])
print(recovered_combined_dataset["train"][0])

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = recovered_combined_dataset.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=["text"])

# block_size = tokenizer.model_max_length
block_size = 1024

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
)

#%% Training setting

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    f"{model_name}-finetuned-prontoqa",
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