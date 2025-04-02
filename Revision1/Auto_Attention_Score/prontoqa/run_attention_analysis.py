import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from analyze_attention_scores_with_model_input import analyze_label_attention_scores

def main():
    parser = argparse.ArgumentParser(description="Analyze attention scores for logic proving dataset.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Hugging Face model name (e.g., 'gpt2')")
    parser.add_argument("--dataset_path", type=str, default="logic_proving_dataset.csv", help="Path to CSV dataset")
    args = parser.parse_args()

    print("\nModel name: " + args.model_name + "\n")
    model_name = args.model_name

    access_token = ""
    
    # access_token
    if model_name in ["google/gemma-2-9b-it", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-1B-Instruct",
                      "google/gemma-2-2b", "meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        if not access_token:
            raise ValueError("Access token is required")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
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
            token=access_token
        )
    # need access_token and trust_remote_code
    elif model_name in ["apple/OpenELM-3B", "apple/OpenELM-3B-Instruct", "apple/OpenELM-1_1B-Instruct", "apple/OpenELM-1_1B", "apple/OpenELM-270M", "apple/OpenELM-450M",
                        "apple/OpenELM-270M-Instruct", "apple/OpenELM-450M-Instruct"]:
        if not access_token:
            raise ValueError("Access token is required")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", output_attentions=True, device_map="cuda:0",  token=access_token)
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

    analyze_label_attention_scores(args.dataset_path, model, tokenizer)

if __name__ == "__main__":
    main()
