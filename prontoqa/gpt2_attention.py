from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
print("\n" + "Model name: " + model_name + "\n")

tokenizer = AutoTokenizer.from_pretrained(model_name, output_attentions=True, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


def predict(prompt, rule, trial):
    
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
    
    tokenized_context = tokenizer(prompt)
    ctx_input_ids = tokenized_context['input_ids']
    print("\nContext length:" + str(len(ctx_input_ids)))
    print('\n')
    
    # inputs = tokenizer(
    # [
    #      prompt
    # ], return_tensors = "pt").to("cuda")
    
    # outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
    outputs = model.generate(input_ids, max_new_tokens = 256, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
    out_1 = tokenizer.batch_decode(outputs)[0]
    # if i == 1:
    #     print("\n\n-----1st LLM output start-----\n")
    #     print(action_1)
    #     print("\n-----1st LLM output end-----\n\n")
    
    # max_context_length = len(input_ids)
    print("\n\n-----LLM output start-----\n")
    print(out_1)
    print("\n-----LLM output end-----\n\n")
    # print("\n-----NLP output-----\n")
    # 1
    # initializing stop string
    out_1 = out_1.replace(r"\n", "\n")
    stop = prompt # + " "
    # slicing off after length computation    
    if stop in out_1:
        out_2 = out_1.split(stop)[1]
        # print("\n-----True 1-----\n")
    else:
        out_2 = out_1
        # print("\n-----False 1-----\n")
    # 2
    # initializing stop string
    stop="Q:"
    # slicing off after length computation       
    if stop in out_2:
        out_3 = out_2.split(stop)[0]
        # print("\n-----True 2-----\n")
    else:
        out_3 = out_2
        # print("\n-----False 2-----\n")
    generated_string = out_3
    
	# input_ids = tokenizer.encode(prompt.replace('\n', ' \\n '), return_tensors="pt").cuda()
	# res = model.generate(input_ids, max_new_tokens=256, do_sample=False)
	# generated_string = tokenizer.batch_decode(res, skip_special_tokens=True)
    # if len(generated_string) != 1:
    #     print("WARNING: len(generated_string) is not 1.")
    # return generated_string[0]
    
    #%% Plot attention heat map
    plot_type = 'heat'
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract attention weights
    attention = outputs.attentions  # A tuple of attention layers
    # print("Attention")
    # print(attention)

    # attention = outputs.attentions
    head = 0
    attention_matrix = attention[-1][0, head].cpu().detach().numpy()
    
    import os
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'images/' + rule + '/')
    sample_file_name = model_name + '_' + rule + '_' + plot_type + '_T' + str(trial) + '_H' + str(head) + '.png'
    os.makedirs(results_dir, exist_ok=True)
    
    # Increase figure size to make axis labels more readable
    plt.figure(figsize=(25, 25))  # Adjust the size as needed
    sns.heatmap(attention_matrix, xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), cmap="viridis")
    plt.title("Attention Weights")
    plt.savefig(os.path.join(results_dir, sample_file_name), bbox_inches='tight')
    
    #%% Plot attention scatter map
    plot_type = 'scatter'
    # import matplotlib.pyplot as plt
    # import os
    import numpy as np
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # Extract attention weights
    attention = outputs.attentions  # A tuple of attention layers
    
    # Extract attention matrix
    layer_num = len(attention)
    print("Number of layers: ")
    print(layer_num)
    head_num = attention[-1].shape[1]
    print("Number of heads: ")
    print(head_num)
    for head in range(head_num):
        # attention_matrix = attention[-1][0, head].cpu().detach().numpy()
        attention_matrix = attention[-1][0][head].cpu().detach().numpy()
        
        # Scale the matrix so the mean is 1
        # amin = np.min(attention_matrix)
        # normalized_attention_matrix = (attention_matrix / amin) * 1
        
        # Define the tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Define the directory to save the image
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'images/' + rule + '/')
        sample_file_name = model_name + '_' + rule + '_' + plot_type + '_T' + str(trial) + '_H' + str(head) + '.png'
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up the plot
        plt.figure(figsize=(200, 5))
        ax = plt.gca()
        
        # Define vertical position for the tokens
        y_position = 2
        
        # Create a scatter plot for tokens
        for i, token in enumerate(tokens):
            ax.text(i, 0, token, ha='center', va='center', fontsize=12)
            ax.text(i, y_position, token, ha='center', va='center', fontsize=12)
        
        # Create a graph to visualize attention
        for i in range(len(tokens)):
            # for j in range(len(tokens)):
            for j in range(i + 1):
                attention_strength = attention_matrix[i, j]
                ax.plot([i, j], [0, y_position], 'k-', lw=attention_strength)  # Line with thickness scaled by attention strength
                # ax.plot([i, j], [0, y_position], 'k-', lw=1)  # Line with thickness scaled by attention strength
        
        # Remove axis for a cleaner view
        ax.axis('off')
        
        # Add title
        plt.title("Attention Visualization", fontsize=16)
        
        # Save the image
        plt.savefig(os.path.join(results_dir, sample_file_name), bbox_inches='tight')
        
        # Close the plot to avoid memory issues
        plt.close()

    
    return generated_string, None
