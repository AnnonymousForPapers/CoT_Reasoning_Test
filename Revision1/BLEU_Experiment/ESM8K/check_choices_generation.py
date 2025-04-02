import time
import pandas as pd
import numpy as np
import torch

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
        for i, row in enumerate(gsm8k_data):
            cnt += 1
            
            # Find the index of the first "Q:"
            Q_text = row.split("\n")[0]
            prompt_text = Q_text.split(". ")[0] + ". "
            try:
                label_text = Q_text.split(prompt_text)[1]
            except IndexError:
                print(f"Skip Q #{cnt}")
                print(Q_text)
                skip += 1
                skips.append(cnt)
                continue  # or pass, depending on context
            except Exception as e:
                print(f"Skipping due to error: {e}")
                continue  # or pass
            
            # First half: input to model, second half: ground truth label
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
            
            # Prepare input tensor for generation
            input_tensor = input_ids.unsqueeze(0).to(model.device)
            model.generation_config.max_new_tokens = None  # Ensure it's unset
            generated_tokens = model.generate(
                input_tensor,
                max_length=input_tensor.shape[-1] + 256,
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
    print(bleu_score)
    print("\n")
        
    end_time = time.time()
    print(f"Total validation time: {end_time - start_time:.2f} seconds")