import time
import pandas as pd
import numpy as np
import torch

def analyze_generated_text(model, tokenizer):
    #%% ✅ Compute metrics function
    import evaluate
    # ✅ Load metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    ood_sample = False
    
    def compute_metrics():
        
        df = pd.read_csv("logic_proving_dataset.csv")
        bleu_list_each = []
        bleu_list = []
        cnt = 0
        for i, row in df.iterrows():
            cnt += 1
            input_text = row['input']
            
            if ood_sample == True:
                # Find the index of the last "Q:"
                last_q_index = input_text.rfind("Q: ")
                Q_text = input_text[last_q_index:]
                prompt_text = Q_text.split(". ")[0] + ". "
                label_text = Q_text.split(prompt_text)[1]
                label_text = label_text.split("\n")[0]
            else:
                # Find the index of the first "Q:"
                Q_text = input_text.split("\n")[0]
                prompt_text = Q_text.split(". ")[0] + ". "
                label_text = Q_text.split(prompt_text)[1]

            # First half: input to model, second half: ground truth label
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
            # label_ids = tokenizer.encode(label_text, return_tensors="pt")[0]
    
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
                bleu_list_each.append(0)
                print(f"\n BLEU-4 score: {0}\n")
            else:
                # Compute BLEU
                bleu_result = bleu_metric.compute(predictions=[response_sliced], references=[[label_text]])
                bleu_list_each.append(bleu_result["bleu"])
                print(f"\n BLEU-4 score: {bleu_result['bleu']}\n")
            if not cnt % 100:
                bleu_list.append(np.mean(bleu_list_each))
                bleu_list_each.clear()
                
        return bleu_list

    #%% Evaluate the Model on the CSQA prompt
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