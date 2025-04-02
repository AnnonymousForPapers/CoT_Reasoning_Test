import time

#%% Load the Gemma-9B Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Measure model loading time
start_time = time.time()

model_name = "google/gemma-2-9b-it"
print("\n" + "Model name: " + model_name + "\n")

access_token = ""

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=access_token
)

end_time = time.time()
print(f"Model loading time: {end_time - start_time:.2f} seconds")

#%% Prepare the CSQA Dataset
from datasets import load_dataset

# Measure dataset loading time
start_time = time.time()

# Load CSQA dataset
csqa = load_dataset("commonsense_qa")

end_time = time.time()
print(f"Dataset loading time: {end_time - start_time:.2f} seconds")

# Sample data entry
sample_entry = csqa["train"][0]
print(sample_entry)
# Output example: {'id': '075e483d21c29a511267ef62bedc0461', 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?', 'question_concept': 'punishing', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']}, 'answerKey': 'A'}

#%% Create dictionay to map response to answer key
# Example dictionary with keys "(a)" to "(z)" and values "A" to "Z"
alphabet_dict = {f"{chr(97 + i)}": chr(65 + i) for i in range(26)}

# Access the value for the key "(a)"
result = alphabet_dict["a"]

#%% Define Chain-of-Thought Prompting
def create_cot_prompt(question, choices):
    Examplars = "Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: (a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).\nQ: What home entertainment equipment requires cable? Answer Choices: (a) radio shack\n(b) substation\n(c) television\n(d) cabinet\nA: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).\nQ: The fox walked from the city into the forest, what was it looking for? Answer Choices: (a) pretty flowers\n(b) hen house\n(c) natural habitat\n(d) storybook\nA: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b).\nQ: Sammy wanted to go to where the people were. Where might he go? Answer Choices: (a) populated areas\n(b) race track\n(c) desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).\nQ: Where do you put your grapes just before checking out? Answer Choices: (a) mouth\n(b) grocery cart\n(c) super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).\nQ: Google Maps and other highway and street GPS services have replaced what? Answer Choices: (a) united states\n(b) mexico\n(c) countryside\n(d) atlas\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).\nQ: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c).\n"
    # prompt = Examplars + f"Q: {question}\n"
    prompt = Examplars + f"Q: {question} Answer Choices: "
    for i, choice in enumerate(choices):
        # prompt += f"Choice {chr(65 + i)}: {choice}\n"
        prompt += f"({chr(97 + i)}) {choice}\n"
    return prompt

#%% Predicting the Answer Using Gemma-9B

def predict_answer(question, choices, label=0, no_answer_count=0):
    prompt = create_cot_prompt(question, choices)
    print("\n\n")
    print("#" + str(label) + " prompt")
    print(prompt)
    print("\n\n")
    inputs = tokenizer(prompt, return_tensors="pt")# .to("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(inputs.device)

    # Generate response with chain-of-thought if desired
    # outputs = model.generate(
    #     **inputs,
    #     max_length=inputs["input_ids"].shape[-1] + 30,  # Adjust to include reasoning steps
    #     do_sample=True,
    #     temperature=0.7  # Adjust temperature for creativity in reasoning
    # )
    outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[-1] + 100, use_cache = True, do_sample=False, num_beams=1, top_p=1, repetition_penalty=0.0001,)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    stop = "the answer is ("
    # slicing off the prompt 
    if stop in response_sliced:
        chosen_answer_incomplete = response_sliced.split(stop)[1].strip()
        stop = ")"
        if stop in chosen_answer_incomplete:
            chosen_answer = chosen_answer_incomplete.split(stop)[0].strip()
        else:
            chosen_answer = chosen_answer_incomplete
        # print("\n-----True 1-----\n")
    else:
        chosen_answer = "none"
        # print("\n-----False 1-----\n")
    
    chosen_answer_lower = chosen_answer.lower()

    # Check if the string is not in the dictionary
    if chosen_answer_lower not in alphabet_dict:
        no_answer_count += 1
        return chosen_answer_lower, no_answer_count, ""
    else:
        stop = "the answer is ("
        complete_cot = response_sliced.split(stop)[0] + "the answer is (" + chosen_answer_lower + ")."
        return alphabet_dict[chosen_answer], no_answer_count, complete_cot

#%% Evaluate the Model on the CSQA training Set
import csv

# New tracking variables
label = 1
correct = 0
no_answer_count = 0

correct_examples = []

for entry in csqa["train"]:
    question = entry["question"]
    choices = entry["choices"]["text"]
    answer_key = entry["answerKey"]

    predicted_answer, no_answer_count, cot_response = predict_answer(question, choices, label, no_answer_count)

    if predicted_answer == answer_key:
        prompt = create_cot_prompt(question, choices)
        
        correct_examples.append({
            "question": question,
            "choices": choices,
            "answer_key": answer_key,
            # "prompt": prompt,
            "cot_response": cot_response,
            "predicted": predicted_answer
        })
        
        print(correct_examples[-1])

        correct += 1
        if correct >= 1000:
            break

    label += 1

# Split and save CSVs
def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "choices", "answer_key", "cot_response", "predicted"])
        writer.writeheader()
        for row in data:
            row["choices"] = str(row["choices"])  # list -> string
            writer.writerow(row)

train_data = correct_examples[:900]
val_data = correct_examples[900:1000]

save_to_csv(train_data, "train_cot_correct.csv")
save_to_csv(val_data, "val_cot_correct.csv")

print(f"Saved {len(train_data)} training examples and {len(val_data)} validation examples.")