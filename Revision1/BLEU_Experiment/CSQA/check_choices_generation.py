import time
import pandas as pd
import numpy as np
import torch

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
def create_cot_prompt(question, choices):
    prompt_1 = f"Q: {question} Answer Choices: "
    prompt_2 = ""
    for i, choice in enumerate(choices):
        if i < 3:
            prompt_1 += f"({chr(97 + i)}) {choice}\n"
        else:
            prompt_2 += f"({chr(97 + i)}) {choice}\n"
    return prompt_1, prompt_2
csqa_q = []
csqa_c = []
# Iterate through training data
for idx, row in train_df.iterrows():
    question = row["question"]
    choices = eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
    cot_response = row["cot_response"]
    predicted = row["predicted"]
    answer_key = row["answer_key"]
    q,c = create_cot_prompt(question, choices)
    csqa_q.append(q)
    csqa_c.append(c)

    print(f"Train Example {idx + 1}")
    print("Q:", question)
    print("Choices:", choices)
    print("CoT Reasoning:", cot_response)
    print("Predicted Answer:", predicted)
    print("Ground Truth:", answer_key)
    print("=" * 50)

def analyze_generated_text(model, tokenizer):
    #%% ✅ Compute metrics function
    import evaluate
    # ✅ Load metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    def compute_metrics():
        bleu_list = []
        cnt = 0
        skip = 0
        skips = []
        for prompt_text, label_text in zip(csqa_q, csqa_c):
            cnt += 1
            
            # First half: input to model, second half: ground truth label
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
            
            # Prepare input tensor for generation
            input_tensor = input_ids.unsqueeze(0).to(model.device)
            model.generation_config.max_new_tokens = None  # Ensure it's unset
            generated_tokens = model.generate(
                input_tensor,
                max_length=input_tensor.shape[-1] + 100,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                top_p=1,
                repetition_penalty=0.0001,
            ).detach().cpu()
    
            decoded_pred = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            stop = prompt_text
            # slicing off the prompt 
            if stop in decoded_pred:
                response_sliced = decoded_pred.split(stop)[1]
                # print("\n-----True 1-----\n")
            else:
                response_sliced = decoded_pred
                # print("\n-----False 1-----\n")
    
            print(f"\n Input text: {prompt_text}\n")
            print(f"\n Label text: {label_text}\n")
            print(f"\n Generated text: {response_sliced}\n")
            
            if response_sliced.strip() == "":
                print("Only whitespace")
                bleu_list.append(0)
                print(f"\n BLEU-4 score: {0}\n")
            else:
                # Compute BLEU
                bleu_result = bleu_metric.compute(predictions=[response_sliced], references=[[label_text]])
                bleu_list.append(bleu_result["bleu"])
                print(f"\n BLEU-4 score: {bleu_result['bleu']}\n")
        print(f"Number of skips: {skip}") 
        print(skips)
        return bleu_list

    #%% Evaluate the Model on the GSM8K prompt
    # Track total validation time
    start_time = time.time()
        
    # Predict answer
    bleu_score = compute_metrics()
    print("\n")
    print("BLEU-4 score")
    print(np.mean(bleu_score))
    print("\n")
        
    end_time = time.time()
    print(f"Total validation time: {end_time - start_time:.2f} seconds")