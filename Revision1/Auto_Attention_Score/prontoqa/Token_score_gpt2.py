import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from analyze_attention_scores_with_model_input import analyze_label_attention_scores

# Load dataset
df = pd.read_csv("logic_proving_dataset.csv")

model_name = "gpt2"
print("\n" + "Model name: " + model_name + "\n")

tokenizer = AutoTokenizer.from_pretrained(model_name, output_attentions=True, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

analyze_label_attention_scores("logic_proving_dataset.csv", model, tokenizer)