
import pandas as pd
import numpy as np
import torch

deduction_rules = ["ModusPonens", "AndIntro", "AndElim", "OrIntro", "OrElim", "ProofByContra", "Composed"]

def analyze_label_attention_scores(csv_path, model, tokenizer):
    df = pd.read_csv(csv_path)
    
    Show_atten = True

    label_scores = []
    running_avg = []
    running_std = []
    saved_stats = []

    for i, row in df.iterrows():
        input_text = row['input']
        label_text = row['label']
        label_tokens = tokenizer.tokenize(label_text)
        print("current tokens")
        print(label_tokens)

        token_scores = []

        for j in range(len(label_tokens)):
            partial_label = tokenizer.convert_tokens_to_string(label_tokens[:j])
            prompt = input_text + " " + partial_label if partial_label else input_text

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            attention_matrix = outputs.attentions[-1][0]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            current_label_token = label_tokens[j]
            print("Input token")
            print(tokens)
            print("current_label_token")
            print(current_label_token)
            scores_per_head = []
            
            if Show_atten:
                # Extract attention matrix
                layer_num = len(outputs.attentions)
                print("Number of layers: ")
                print(layer_num)
                head_num = outputs.attentions[-1].shape[1]
                print("Number of heads: ")
                print(head_num)
                Show_atten = False

            for head_attention in attention_matrix:
                attn = head_attention.cpu().numpy()
                Gt = np.sum(attn, axis=0)
                Gt[0] = 0
                B = np.dot(attn, np.diag(Gt))
                B_col_sum = np.sum(B, axis=0)
                B2 = B / B_col_sum
                B2_0 = np.nan_to_num(B2, nan=0)
                Pt = np.sum(B2_0, axis=1)
                St = Gt + Pt
                values = (St - np.min(St)) / (np.max(St) - np.min(St))

                matching_token_scores = [values[k] for k, tok in enumerate(tokens) if tok == current_label_token]
                scores_per_head.append(max(matching_token_scores) if matching_token_scores else 0.0)

            avg_token_score = np.mean(scores_per_head)
            token_scores.append(avg_token_score)

        label_avg = np.mean(token_scores)
        label_scores.append(label_avg)
        running_avg.append(np.mean(label_scores))
        running_std.append(np.std(label_scores))
        
        print(f"Data {i+1} processed.")
        print(f"Sum of scores: {np.sum(label_scores):.4f}")
        print(f"Number of labels: {len(label_scores)}")
        print(f"Average for current label: {label_avg:.4f}")
        print(f"Standard deviation so far: {np.std(label_scores):.4f}")
        print(f"Running average: {running_avg[-1]:.4f}, Running std: {running_std[-1]:.4f}")
        print("------\n")

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"Data {i+1} processed.")
            print(f"{deduction_rules[i//100]}")
            stat = {
                "data_processed": i + 1,
                "sum_scores": float(np.sum(label_scores)),
                "num_labels": len(label_scores),
                "avg_current": float(label_avg),
                "std_current": float(np.std(label_scores)),
                "running_avg": float(running_avg[-1]),
                "running_std": float(running_std[-1])
            }
            saved_stats.append(stat)
            
            label_avg = 0
            label_scores.clear()
            running_avg.clear()
            running_std.clear()

        

    print("\nFinal saved statistics every 100 data:")
    for stat in saved_stats:
        print(stat)
        
    