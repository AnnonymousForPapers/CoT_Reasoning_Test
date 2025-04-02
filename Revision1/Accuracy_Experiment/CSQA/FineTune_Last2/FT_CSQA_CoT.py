import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

num_proc = 1

#%% Load LM
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    # Measure model loading time
    start_time = time.time()
    print(f"\nModel name: {model_name}\n")
    access_token = ""
    
    # access_token
    if model_name in ["google/gemma-2-9b-it", "google/gemma-2-2b-it"]:
        if not access_token:
            raise ValueError("Access token is required")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=access_token
        )
    elif model_name in ["google/gemma-3-1b-pt", "google/gemma-3-1b-it"]:
        if not access_token:
            raise ValueError("Access token is required")
        # # Ref: https://github.com/google-deepmind/gemma/issues/169
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_flash_sdp(False)
        # torch.backends.cuda.enable_math_sdp(True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=access_token
        )
    elif model_name in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        if not access_token:
            raise ValueError("Access token is required")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=access_token
        )
    # need access_token and trust_remote_code
    elif model_name in ["apple/OpenELM-3B", "apple/OpenELM-3B-Instruct", "apple/OpenELM-1_1B-Instruct", "apple/OpenELM-1_1B", "apple/OpenELM-270M", "apple/OpenELM-450M",
                        "apple/OpenELM-270M-Instruct", "apple/OpenELM-450M-Instruct"]:
        if not access_token:
            raise ValueError("Access token is required")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="cuda:0",  token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0", # 'auto' somehow doesn't work for models from apple
            trust_remote_code=True,
            token=access_token
        )
    elif model_name in ["ibm-granite/granite-3.1-2b-instruct", "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "tiiuae/Falcon3-1B-Instruct", "tensoropera/Fox-1-1.6B-Instruct-v0.1", "stabilityai/stablelm-2-zephyr-1_6b", 
                        "ibm-granite/granite-3.1-2b-base", "Qwen/Qwen2.5-1.5B", "HuggingFaceTB/SmolLM2-1.7B", "tiiuae/Falcon3-1B-Base", "tensoropera/Fox-1-1.6B", "stabilityai/stablelm-2-1_6b", "allenai/OLMo-1B-hf", "gpt2-xl", "EleutherAI/pythia-1.4b", "facebook/opt-1.3b", "JackFram/llama-160m", "pints-ai/1.5-Pints-16K-v0.1", "TinyLlama/TinyLlama_v1.1", "bigscience/bloomz-1b7",
                        "ibm-granite/granite-3.1-1b-a400m-base", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "gpt2", "gpt2-medium", "gpt2-large", "EleutherAI/pythia-1b", "EleutherAI/pythia-410m", "EleutherAI/pythia-160m", "EleutherAI/pythia-70m", "EleutherAI/pythia-14m", "facebook/opt-125m", "facebook/opt-350m", "JackFram/llama-68m", "pints-ai/1.5-Pints-2K-v0.1", "bigscience/bloomz-560m", "bigscience/bloomz-1b1",
                        "ibm-granite/granite-3.1-1b-a400m-instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, output_attentions=True, device_map="cuda:0")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")
    else:
        raise ValueError("Invalid model name.")
    if model_name in ["meta-llama/Llama-3.2-1B-Instruct"]:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model_name in ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B"]:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model_name in ["apple/OpenELM-3B", "apple/OpenELM-3B-Instruct", "apple/OpenELM-1_1B-Instruct", "apple/OpenELM-1_1B", "apple/OpenELM-270M", "apple/OpenELM-450M",
                        "apple/OpenELM-270M-Instruct", "apple/OpenELM-450M-Instruct"]:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model_name in ["TinyLlama/TinyLlama_v1.1", "stabilityai/stablelm-2-1_6b"]:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    end_time = time.time()
    print(f"Model loading time: {end_time - start_time:.2f} seconds")
    return tokenizer, model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }

#%% Load CSQA-CoT data
import pandas as pd

# Measure dataset loading time
start_time = time.time()

train_df = pd.read_csv("train_cot_correct.csv")
val_df = pd.read_csv("val_cot_correct.csv")

print("Train dataset size:", len(train_df))
print("Validation dataset size:", len(val_df))

# View first few entries
print(train_df.head())
# Output example: 

end_time = time.time()
print(f"Dataset loading time: {end_time - start_time:.2f} seconds")

#%% Create dictionay to map response to answer key
# Example dictionary with keys "(a)" to "(z)" and values "A" to "Z"
alphabet_dict = {f"{chr(97 + i)}": chr(65 + i) for i in range(26)}

# Access the value for the key "(a)"
result = alphabet_dict["a"]

#%% Define Chain-of-Thought Prompting
def create_cot_prompt(question, choices, cot_response):
    prompt = f"Q: {question} Answer Choices: "
    for i, choice in enumerate(choices):
        # prompt += f"Choice {chr(65 + i)}: {choice}\n"
        prompt += f"({chr(97 + i)}) {choice}\n"
    prompt += cot_response # change here T -> t
    return prompt
csqa_data = []
# Iterate through training data
for idx, row in train_df.iterrows():
    question = row["question"]
    choices = eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
    cot_response = row["cot_response"]
    predicted = row["predicted"]
    answer_key = row["answer_key"]
    csqa_data.append(create_cot_prompt(question, choices, cot_response))

    print(f"Train Example {idx + 1}")
    print("Q:", question)
    print("Choices:", choices)
    print("CoT Reasoning:", cot_response)
    print("Predicted Answer:", predicted)
    print("Ground Truth:", answer_key)
    print("=" * 50)
    
csqa_val_data = []
# Iterate through training data
for idx, row in val_df.iterrows():
    question = row["question"]
    choices = eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
    cot_response = row["cot_response"]
    predicted = row["predicted"]
    answer_key = row["answer_key"]
    csqa_val_data.append(create_cot_prompt(question, choices, cot_response))

    print(f"Train Example {idx + 1}")
    print("Q:", question)
    print("Choices:", choices)
    print("CoT Reasoning:", cot_response)
    print("Predicted Answer:", predicted)
    print("Ground Truth:", answer_key)
    print("=" * 50)
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load a model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size per device during training (default: 1)")
    
    args = parser.parse_args()
    
    train_model_name = args.model_name
    train_batch_size = args.train_batch_size
    print(f"Training batch size: {train_batch_size}")
    train_tokenizer, train_model = load_model(train_model_name)
    
    # Set unfreeze layers
    if train_model_name in ["gpt2"]:
        tounfreeze= set(["transformer.h.11", "transformer.h.10"])
        train_tokenizer.pad_token = train_tokenizer.eos_token
        train_model.generation_config.pad_token_id = train_tokenizer.pad_token_id
    elif train_model_name in ["gpt2-medium"]:
        tounfreeze= set(["transformer.h.23", "transformer.h.22"])
        train_tokenizer.pad_token = train_tokenizer.eos_token
        train_model.generation_config.pad_token_id = train_tokenizer.pad_token_id
    elif train_model_name in ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"]:
        tounfreeze= set(["model.layers.29", "model.layers.28"])
        train_tokenizer.pad_token = train_tokenizer.eos_token
        train_model.generation_config.pad_token_id = train_tokenizer.pad_token_id
    elif train_model_name in ["apple/OpenELM-270M", "apple/OpenELM-270M-Instruct"]:
        tounfreeze= set(["transformer.layers.15", "transformer.layers.14"])
        train_tokenizer.pad_token = train_tokenizer.eos_token
        train_model.generation_config.pad_token_id = train_tokenizer.pad_token_id
    else:
        raise ValueError("Invalid model name.")  
        
    for m in train_model.modules():
        for name, params in m.named_parameters():
            params.requires_grad = False
    for m in train_model.modules():
        for name, params in m.named_parameters():
            if any(prefix for prefix in tounfreeze if name.startswith(prefix)):
                params.requires_grad = True
            print(name, params.requires_grad)
    # https://github.com/huggingface/peft/issues/137
    train_model.enable_input_require_grads()
    
    #%% Load dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": csqa_data})
    val_dataset = Dataset.from_dict({"text": csqa_val_data})

    #%% Preprocess
    """
    Training
    """
    # Tokenize and calculate token lengths
    token_lengths = [len(train_tokenizer.encode(ex, add_special_tokens=True)) for ex in csqa_data]
    
    # Find maximum number of tokens
    max_num_tokens = max(token_lengths)
    
    # Print result
    print("Maximum number of tokens:", max_num_tokens)

    def tokenize_function(examples):
        # result = train_tokenizer(next(iter(examples)), max_length=256, padding='max_length', truncation=True)
        result = train_tokenizer(examples["text"], max_length=max_num_tokens, padding='max_length', truncation=True) # 348 characters
        # print(result)
        result["labels"] = result["input_ids"].copy()
        return result
    # tokenized_datasets = combined_dataset.map(tokenize_function, batched=False, remove_columns=["text"])# , num_proc=num_proc
    lm_datasets = dataset.map(tokenize_function, batch_size=1, remove_columns=["text"])# , num_proc=num_proc
    val_lm_datasets = val_dataset.map(tokenize_function, batch_size=1, remove_columns=["text"])# , num_proc=num_proc
    print("lm_datasets")
    print(lm_datasets)
    print("len(lm_datasets)")
    print(len(lm_datasets))
    print("val_lm_datasets")
    print(val_lm_datasets)
    print("len(val_lm_datasets)")
    print(len(val_lm_datasets))

    #%% Training setting
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        f"{train_model_name}-Batch{train_batch_size}-CSQA",
        logging_strategy = "epoch",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        num_train_epochs=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end = True,
        save_total_limit=2, # Save the best and the last checkpoint
    )

    trainer = Trainer(
        model=train_model,
        args=training_args,
        train_dataset=lm_datasets,
        eval_dataset=val_lm_datasets,
        tokenizer=train_tokenizer
    )
    
    # Pre-training loss + metrics on training dataset
    # Set model to evaluation mode
    trainer.model.eval()
    
    # Get the dataloader from the Trainer
    train_dataloader = trainer.get_train_dataloader()
    
    total_loss = 0.0
    num_batches = 0
    
    # Iterate over dataloader batches
    for batch in train_dataloader:
        # Move batch tensors to the correct device
        batch = {k: v.to(trainer.model.device) for k, v in batch.items()}
        
        # Compute loss for this batch (return_outputs=False returns only the loss)
        with torch.no_grad():
            loss = trainer.compute_loss(trainer.model, batch)
        
        total_loss += loss.item()
        num_batches += 1
    
        print(f"Batch {num_batches} - Loss: {loss.item()}")
    
    # Compute the average loss over the whole training set
    average_train_loss = total_loss / num_batches
    print(f"\n✅ Average Training Loss: {average_train_loss}", flush=True)
    
    trainer.model.train()
    
    # Count parameters
    param_counts = count_parameters(train_model)
    print(param_counts, flush=True)

    trainer.train()
    
    #%% Show training loss
    train_loss = []
    for elem in trainer.state.log_history:
        if 'loss' in elem.keys():
            train_loss.append(elem['loss'])
    print("training_loss: ")
    print(train_loss)
    
    #%% Prepare the CSQA Dataset
    from datasets import load_dataset

    # Measure dataset loading time
    start_time = time.time()

    # Load CSQA dataset
    csqa = load_dataset("commonsense_qa")

    end_time = time.time()
    print(f"Dataset loading time: {end_time - start_time:.2f} seconds")

    # Sample data entry
    sample_entry = csqa["train"][0]
    print(sample_entry)
    # Output example: {'id': '075e483d21c29a511267ef62bedc0461', 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?', 'question_concept': 'punishing', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']}, 'answerKey': 'A'}

    
    #%% Define Chain-of-Thought Prompting
    """
    Testing
    """
    
    if train_model_name in ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"]:    
        train_model.generation_config.max_new_tokens = None  # Ensure it's unset
    if train_model_name in ["google/gemma-3-1b-pt", "google/gemma-3-1b-it"]:
        # Ref: https://github.com/google-deepmind/gemma/issues/169
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    def create_cot_prompt(question, choices):
        # Examplars = "Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: (a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).\nQ: What home entertainment equipment requires cable? Answer Choices: (a) radio shack\n(b) substation\n(c) television\n(d) cabinet\nA: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).\nQ: The fox walked from the city into the forest, what was it looking for? Answer Choices: (a) pretty flowers\n(b) hen house\n(c) natural habitat\n(d) storybook\nA: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b).\nQ: Sammy wanted to go to where the people were. Where might he go? Answer Choices: (a) populated areas\n(b) race track\n(c) desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).\nQ: Where do you put your grapes just before checking out? Answer Choices: (a) mouth\n(b) grocery cart\n(c) super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).\nQ: Google Maps and other highway and street GPS services have replaced what? Answer Choices: (a) united states\n(b) mexico\n(c) countryside\n(d) atlas\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).\nQ: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c).\n"
        # prompt = Examplars + f"Q: {question}\n"
        # prompt = Examplars + f"Q: {question} Answer Choices: "
        prompt = f"Q: {question} Answer Choices: "
        for i, choice in enumerate(choices):
            # prompt += f"Choice {chr(65 + i)}: {choice}\n"
            prompt += f"({chr(97 + i)}) {choice}\n"
        return prompt

    #%% Predicting the Answer

    def predict_answer(question, choices, label, no_answer_count):
        prompt = create_cot_prompt(question, choices)
        print("\n\n")
        print("#" + str(label) + " prompt")
        print(prompt)
        print("\n\n")
        inputs = train_tokenizer(prompt, return_tensors="pt").to(train_model.device)# .to("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(inputs.device)

        # Generate response with chain-of-thought if desired
        # outputs = model.generate(
        #     **inputs,
        #     max_length=inputs["input_ids"].shape[-1] + 30,  # Adjust to include reasoning steps
        #     do_sample=True,
        #     temperature=0.7  # Adjust temperature for creativity in reasoning
        # )
        outputs = train_model.generate(**inputs, max_length=inputs["input_ids"].shape[-1] + 100, use_cache = True, do_sample=False, num_beams=1, top_p=1, repetition_penalty=0.0001,)
        response = train_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("#" + str(label) + " response")
        print(response)
        print("\n\n")
        
        stop = prompt
        # slicing off the prompt 
        if stop in response:
            response_sliced = response.split(stop)[1]
            # print("\n-----True 1-----\n")
        else:
            response_sliced = response
            # print("\n-----False 1-----\n")
            
        print("#" + str(label) + " sliced response")
        print(response_sliced)
        print("\n\n")
        
        # Parse response for the answer (this can vary based on CoT structure)
        stop = "the answer is ("
        # slicing off the prompt 
        if stop in response_sliced:
            chosen_answer_incomplete = response_sliced.split(stop)[1].strip()
            stop = ")"
            if stop in chosen_answer_incomplete:
                chosen_answer = chosen_answer_incomplete.split(stop)[0].strip()
            else:
                chosen_answer = chosen_answer_incomplete
            # print("\n-----True 1-----\n")
        else:
            chosen_answer = "none"
            # print("\n-----False 1-----\n")
        
        chosen_answer = chosen_answer.lower()

        # Check if the string is not in the dictionary
        if chosen_answer not in alphabet_dict:
            no_answer_count += 1
            return chosen_answer, no_answer_count
        else:
            return alphabet_dict[chosen_answer], no_answer_count
    
    #%% Evaluate the Model on the CSQA Validation Set
    label = 1
    correct = 0
    no_answer_count = 0
    total = len(csqa["validation"])

    print(f"Total number of validation data: {total}")

    # Track total validation time
    start_time = time.time()

    for entry in csqa["validation"]:
        question = entry["question"]
        choices = entry["choices"]["text"]
        answer_key = entry["answerKey"]
        
        # Predict answer
        predicted_answer, no_answer_count = predict_answer(question, choices, label, no_answer_count)
        
        # Parse response for best answer (this can vary based on CoT structure)
        # chosen_answer_incomplete = response_sliced.split("the answer is ")[1].strip()
        # chosen_answer = chosen_answer_incomplete.split(".")[0].strip()
        print("#" + str(label) + " chosen answer")
        print(predicted_answer)
        print("\n")
        print("correct answer")
        print(answer_key)
        print("\n")
        
        # Check correctness
        correct += (predicted_answer == answer_key)
        
        print(f"correct count: {correct}")
        print(f"total count: {label}")
        accuracy = correct / label
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        
        print(f"no answer count: {no_answer_count}")
        wrong_answer_count = label - no_answer_count - correct
        print(f"wrong answer count: {wrong_answer_count}")
        
        label += 1
        
    end_time = time.time()
    print(f"Total validation time: {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()

