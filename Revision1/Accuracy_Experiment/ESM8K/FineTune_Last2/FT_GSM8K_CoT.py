# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:36:04 2025

@author: xiaoyenche
"""

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
        # Ref: https://github.com/google-deepmind/gemma/issues/169
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, output_attentions=True, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
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

#%% Prepare the CSQA Dataset
from datasets import load_dataset

# Measure dataset loading time
start_time = time.time()

# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")

end_time = time.time()
print(f"Dataset loading time: {end_time - start_time:.2f} seconds")

# Sample data entry
sample_entry = gsm8k["test"][0]
print(sample_entry)
# Output example: {'question': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", 'answer': 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18'}

print(f"train data length: {len(gsm8k['train'])}")

#%% Create dictionay to map response to answer key
# Example dictionary with keys "(a)" to "(z)" and values "A" to "Z"
alphabet_dict = {f"{chr(97 + i)}": chr(65 + i) for i in range(26)}

# Access the value for the key "(a)"
result = alphabet_dict["a"]

#%% Define Chain-of-Thought Prompting
def create_cot_prompt(question, splited_answer_text, answer_number):
    # Examplars = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    # prompt = Examplars + f"\n\nQ: {question}\nA: {splited_answer_text}" +  "The answer is " + str(answer_number) + "."
    prompt = f"Q: {question}\nA: {splited_answer_text}" +  "The answer is " + str(answer_number) + "."
    return prompt
gsm8k_data = []
cnt = 0
for entry in gsm8k["train"]:
    question = entry["question"]
    answer_text = entry["answer"]
    stop = "\n#### "
    splited_answer_text = answer_text.split(stop)[0].replace("\n", " ")
    answer_text = answer_text.replace(',', '')  # 2,125 → 2125
    answer_number = int(answer_text.split(stop)[1].strip())
    gsm8k_data.append(create_cot_prompt(question, splited_answer_text, answer_number))
    cnt += 1
    if cnt == 900:
        break # Take first 900 data
        
gsm8k_val_data = []
cnt = 0
for entry in gsm8k["train"]:
    cnt += 1
    if cnt <= 900:
        continue # Take after first 1000 data
    question = entry["question"]
    answer_text = entry["answer"]
    stop = "\n#### "
    splited_answer_text = answer_text.split(stop)[0].replace("\n", " ")
    answer_text = answer_text.replace(',', '')  # 2,125 → 2125
    answer_number = int(answer_text.split(stop)[1].strip())
    gsm8k_val_data.append(create_cot_prompt(question, splited_answer_text, answer_number))
    if cnt == 1000:
        break # Take first 1000 data
# print(gsm8k_data[0])

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
    dataset = Dataset.from_dict({"text": gsm8k_data})
    val_dataset = Dataset.from_dict({"text": gsm8k_val_data})

    #%% Preprocess
    """
    Training
    """
    # Tokenize and calculate token lengths
    token_lengths = [len(train_tokenizer.encode(ex, add_special_tokens=True)) for ex in gsm8k_data]
    
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
        f"{train_model_name}-Batch{train_batch_size}-GSM8K",
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
    
    #%% Define Chain-of-Thought Prompting
    """
    Testing
    """
    if train_model_name in ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"]:    
        train_model.generation_config.max_new_tokens = None  # Ensure it's unset
    def create_cot_prompt(question):
        # Examplars = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
        # prompt = Examplars + f"\n\nQ: {question}\nA: "
        prompt = f"Q: {question}\nA: "
        return prompt

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    #%% Predicting the Answer

    def predict_answer(question, answer_text, answer_number, no_answer_count):
        prompt = create_cot_prompt(question)
        print("\n\n")
        print("#" + str(label) + " prompt")
        print(prompt)
        print("\n\n")
        try:
            inputs = train_tokenizer(prompt, return_tensors="pt").to(train_model.device)# .to("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(inputs.device)

        # Generate response with chain-of-thought if desired
        # outputs = model.generate(
        #     **inputs,
        #     max_length=inputs["input_ids"].shape[-1] + 30,  # Adjust to include reasoning steps
        #     do_sample=True,
        #     temperature=0.7  # Adjust temperature for creativity in reasoning
        # )
            outputs = train_model.generate(**inputs, max_length=inputs["input_ids"].shape[-1] + 256, use_cache = True, do_sample=False, num_beams=1, top_p=1, repetition_penalty=0.0001,)
        except RuntimeError as e:
            print("Error during generation:", e)
            return "", no_answer_count + 1
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
        stop = "The answer is "
        # slicing off the prompt 
        if stop in response_sliced:
            chosen_answer_incomplete = response_sliced.split(stop)[1].strip()
            stop = "."
            if stop in chosen_answer_incomplete:
                chosen_answer = chosen_answer_incomplete.split(stop)[0].strip()
            else:
                chosen_answer = chosen_answer_incomplete
            # print("\n-----True 1-----\n")
        else:
            chosen_answer = "none"
            # print("\n-----False 1-----\n")
            
        chosen_answer = chosen_answer.replace(',', '')  # 2,125 → 2125

        # Check if the string is not in the dictionary
        if not is_int(chosen_answer):
            no_answer_count += 1
            return chosen_answer, no_answer_count
        else:
            return int(chosen_answer), no_answer_count

    #%% Evaluate the Model on the CSQA Validation Set
    label = 1
    correct = 0
    no_answer_count = 0
    total = len(gsm8k["test"])

    print(f"Total number of validation data: {total}")

    # Track total validation time
    start_time = time.time()

    for entry in gsm8k["test"]:
        question = entry["question"]
        answer_text = entry["answer"]
        stop = "\n#### "
        answer_text = answer_text.replace(',', '')  # 2,125 → 2125
        answer_number = int(answer_text.split(stop)[1].strip())
        # Predict answer
        predicted_answer, no_answer_count = predict_answer(question, answer_text, answer_number, no_answer_count)
        
        # Parse response for best answer (this can vary based on CoT structure)
        # chosen_answer_incomplete = response_sliced.split("the answer is ")[1].strip()
        # chosen_answer = chosen_answer_incomplete.split(".")[0].strip()
        print("#" + str(label) + " chosen answer")
        print(predicted_answer)
        print("\n")
        print("correct answer")
        print(answer_number)
        print("\n")
        
        # Check correctness
        correct += (predicted_answer == answer_number)
        
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

