
import pandas as pd
import numpy as np
import torch

deduction_rules = ["ModusPonens", "AndIntro", "AndElim", "OrIntro", "OrElim", "ProofByContra", "Composed"]

def analyze_each_output_tokens(csv_path, model, tokenizer):
    df = pd.read_csv(csv_path)
    
    saved_stats = []
    correct_token_counts = []
    total_token_counts = []
    Q_avg = []

    for i, row in df.iterrows():
        input_text = row['input']
        label_text = row['label']
        label_tokens = tokenizer.tokenize(label_text)
        print("current tokens")
        print(label_tokens)

        correct_count = 0
        total_count = 0
        

        for j in range(len(label_tokens)):
            partial_label = tokenizer.convert_tokens_to_string(label_tokens[:j])
            prompt = input_text + " " + partial_label if partial_label else input_text

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens = 1, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
            generated_token = tokenizer.batch_decode(outputs)[0][-1]
            current_label_token = label_tokens[j]
            if generated_token == current_label_token:
                correct_count += 1
            total_count += 1

        correct_token_counts.append(correct_count)
        total_token_counts.append(total_count)
        Q_avg.append(correct_count/total_count)
        
        print(f"Data {i+1} processed.")
        print(f"Correct token counts: {correct_token_counts}")

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"Data {i+1} processed.")
            print(f"{deduction_rules[i//100]}")
            avg_Q_avg = np.mean(Q_avg)
            print(f"correct_token_counts: {correct_token_counts}")
            print(f"total_token_counts: {total_token_counts}")
            print(f"Q_avg: {Q_avg}")
            print(f"avg_Q_avg: {avg_Q_avg:.4f}")
            stat = {
                "data_processed": i + 1,
                "correct_token_counts": correct_token_counts,
                "total_token_counts": total_token_counts,
                "Q_avg": Q_avg,
                "avg_Q_avg": avg_Q_avg
            }
            saved_stats.append(stat)

            correct_token_counts.clear()
            total_token_counts.clear()
            Q_avg.clear()

    print("\nFinal saved statistics every 100 data:")
    for stat in saved_stats:
        print(stat)
        
    