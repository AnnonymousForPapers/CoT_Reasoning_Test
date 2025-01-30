from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

rule = 'ModusPonens'
trial = 1

model_name = "google/gemma-2-9b-it"
print("\n" + "Model name: " + model_name + "\n")

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16, # I need to change bfloat16 to float16 otherwise get TypeError: Got unsupported ScalarType BFloat16
    token=access_token
)

# Your tokenized list
tokens = ['Q', ':', '▁Every', '▁imp', 'us', '▁is', '▁not', '▁floral', '.', '▁Alex', '▁is', '▁an', '▁imp', 'us', '.', '▁Prove', ':', '▁Alex', '▁is', '▁not', '▁floral', '.', '\n', 'A', ':', '▁Alex', '▁is', '▁an', '▁imp', 'us', '.', '▁Every', '▁imp', 'us', '▁is', '▁not', '▁floral', '.', '▁Alex', '▁is', '▁not', '▁floral', '.', '\n\n', 'Q', ':', '▁Every', '▁j', 'om', 'pus', '▁is', '▁not', '▁loud', '.', '▁Rex', '▁is', '▁a', '▁j', 'om', 'pus', '.', '▁Prove', ':', '▁Rex', '▁is', '▁not', '▁loud', '.', '\n', 'A', ':', '▁Rex', '▁is', '▁a', '▁j', 'om', 'pus', '.', '▁Every', '▁j', 'om', 'pus', '▁is', '▁not', '▁loud', '.', '▁Rex', '▁is', '▁not', '▁loud', '.', '\n\n', 'Q', ':', '▁Each', '▁t', 'ump', 'us', '▁is', '▁not', '▁liquid', '.', '▁Rex', '▁is', '▁a', '▁t', 'ump', 'us', '.', '▁Prove', ':', '▁Rex', '▁is', '▁not', '▁liquid', '.', '\n', 'A', ':', '▁Rex', '▁is', '▁a', '▁t', 'ump', 'us', '.', '▁Each', '▁t', 'ump', 'us', '▁is', '▁not', '▁liquid', '.', '▁Rex', '▁is', '▁not', '▁liquid', '.', '\n\n', 'Q', ':', '▁Ster', 'p', 'uses', '▁are', '▁transparent', '.', '▁Wren', '▁is', '▁a', '▁ster', 'pus', '.', '▁Prove', ':', '▁Wren', '▁is', '▁transparent', '.', '\n', 'A', ':', '▁Wren', '▁is', '▁a', '▁ster', 'pus', '.', '▁Ster', 'p', 'uses', '▁are', '▁transparent', '.', '▁Wren', '▁is']
# tokens == tokens_2 True

# Convert tokens back to string
decoded_text = tokenizer.convert_tokens_to_string(tokens)

# Print the decoded text
print(decoded_text)

inputs = tokenizer(decoded_text, return_tensors="pt").to("cuda")

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

import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
#%% Plot attention heat map
plot_type = 'heat'

# Extract attention weights
attention = outputs.attentions  # A tuple of attention layers
# print("Attention")
# print(attention)

# attention = outputs.attentions
head = 0
attention_matrix = attention[-1][0, head].cpu().detach().numpy()

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'images/' + rule + '/')
sample_file_name = 'gemma-2-9b-it_' + '_' + rule + '_' + plot_type + '_T' + str(trial) + '_H' + str(head) + '.png'
os.makedirs(results_dir, exist_ok=True)

# Increase figure size to make axis labels more readable
plt.figure(figsize=(25, 25))  # Adjust the size as needed
sns.heatmap(attention_matrix, xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), cmap="viridis")
plt.title("Attention Weights")
plt.savefig(os.path.join(results_dir, sample_file_name), bbox_inches='tight')

#%% Scatter plot
plot_type = 'scatter'
head_num = 1 # head_num = 1 if just want to print the first head, otherwise comment this line
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
    sample_file_name = 'gemma-2-9b-it_' + '_' + rule + '_' + plot_type + '_T' + str(trial) + '_H' + str(head) + '.png'
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(5, 50))
    ax = plt.gca()
    
    # Define vertical position for the tokens
    x_position = 2
    scatter_pad = 1 # no use
    
    # Create a graph to visualize attention
    for i in range(len(tokens)):
        # for j in range(len(tokens)):
        for j in range(i + 1):
            attention_strength = attention_matrix[i, j]
            ax.plot([0, x_position], [scatter_pad*(len(tokens)-i), scatter_pad*(len(tokens)-j)], 'k-', lw=attention_strength)  # Line with thickness scaled by attention strength
            # ax.plot([i, j], [0, y_position], 'k-', lw=1)  # Line with thickness scaled by attention strength
    
    # Create a scatter plot for tokens
    for i, token in enumerate(tokens):
        ax.text(0, scatter_pad*(len(tokens)-i), token, ha='center', va='top', fontsize=12)
        ax.text(x_position, scatter_pad*(len(tokens)-i), token, ha='center', va='top', fontsize=12)
    
    # Remove axis for a cleaner view
    ax.axis('off')
    
    # Add title
    # plt.title("Attention Visualization", fontsize=16)
    
    # Save the image
    plt.savefig(os.path.join(results_dir, sample_file_name))#, bbox_inches='tight')
    
    # Close the plot to avoid memory issues
    plt.close()

    #%% SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2
    # The part below is just for debugging
    # a = [[1, 0, 0], [0.8, 0.2, 0], [0.7, 0.2, 0.1]]
    # attention_matrix = np.array(a)
    
    # Global attention score (Column sum)
    Gt = np.sum(attention_matrix,0)
    Gt[0] = 0 # set the score of the first token to be zero
    
    # Proportional Attention Score 
    B = np.dot(attention_matrix, np.diag(Gt))
    B_col_sum = np.sum(B,0) # (Column sum)
    B2 = B/B_col_sum
    B2_0 = np.nan_to_num(B2, nan=0) # Set any division by zero (where NaN appears) to 0
    Pt = np.sum(B2_0,1) # (Row sum)
    
    # Token-level scores
    St = Gt + Pt
    
    #%%
    # Normalize the vector to the range [0, 1] (Min-max normalization)
    values = (St - np.min(St)) / (np.max(St) - np.min(St))

    # Define image properties
    img_width = 400  # Width of the image
    padding = 1
    line_height = 15# 11
    font_size = 25# 11 

    # Calculate number of lines needed based on occurrences of '\n'
    num_lines = tokens.count('\n') + tokens.count('\n\n')*2 + 2
    img_height = num_lines * line_height + padding * 2

    # Create an image canvas
    image = Image.new("RGB", (img_width + 80, img_height), "white")  # Adjust space for smaller color bar
    draw = ImageDraw.Draw(image)

    # Load a font (use default if not found)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Position tracking
    x_pos, y_pos = padding, padding

    for token, value in zip(tokens, values):
        # Handle new line on encountering \n
        if token == '\n':
            x_pos = padding  # Reset X position
            y_pos += line_height  # Move to the next line
            continue
        if token == '\n\n':
            x_pos = padding  # Reset X position
            y_pos += line_height*2  # Move to the next line
            continue

        # Replace ▁ with a space
        token = token.replace('▁', ' ')

        # Convert value (0 to 1) to a shade of blue (RGB format)
        color = (int(255 * (1 - value)), int(255 * (1 - value)), 255)

        # Calculate text size
        token = token.replace("\n", r"\n")
        text_width = draw.textlength(token, font=font)
        token = token.replace(r"\n", "\n")
        text_height = 10
        
        # Move to the next line only if there's a \n
        if x_pos + text_width + padding > img_width - 0:
            x_pos = padding  # Reset X position
            y_pos += line_height  # Move to the next line

        # Draw background rectangle with blue shades
        draw.rectangle([x_pos+4, y_pos, x_pos + text_width + padding+3, y_pos + text_height], fill=color)
        
        # Decide text color based on background brightness
        brightness = np.mean(color)
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        # Draw text on top of colored background
        draw.text((x_pos + 5, y_pos), token, fill=text_color, font=font)
        
        # Update X position for the next token
        x_pos += text_width + padding

    # Set the desired height for the color bar (e.g., 70% of image height)
    color_bar_height = int(img_height * 0.7)  # 70% of the image height
    color_bar_top = int((img_height - color_bar_height) / 2)  # Position it at the middle

    # Create a smaller color bar to visualize blue intensity scale (0 at bottom to 1 at top)
    bar_width = 10
    color_bar_x = img_width + 20  # Adjust position of the bar
    for i in range(101):
        # Calculate shade based on the blue intensity
        shade = int(255 * (i / 100))
        bar_color = (shade, shade, 255)
        
        # Draw each rectangle for the color bar
        draw.rectangle([color_bar_x, color_bar_top + i * (color_bar_height / 100), 
                        color_bar_x + bar_width, color_bar_top + (i + 1) * (color_bar_height / 100)], 
                       fill=bar_color)

    # Draw a black border around the color bar
    draw.rectangle([color_bar_x - 2, color_bar_top, color_bar_x + bar_width + 2, color_bar_top + color_bar_height], 
                   outline="black", width=3)

    # Add color bar labels (0 at bottom and 1 at top)
    draw.text((color_bar_x + bar_width + 10, color_bar_top + color_bar_height - 10), "0", fill="black", font=font)
    draw.text((color_bar_x + bar_width + 10, color_bar_top - 10), "1", fill="black", font=font)

    # Save and display the image
    image.show()
    image.save('gemma-2-9b-it_' + "tokens_blue_gradient_with_black_border.png")