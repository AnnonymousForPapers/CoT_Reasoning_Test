import time
import csv

#%% Define Initial Prompting
def create_cot_prompt(question, shots):
    # Examplars = [
    # "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
    # , "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
    # , "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
    # , "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
    # , "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
    # , "Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
    # , "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
    # , "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    # ]
    # # prompt = Examplars + f"Q: {question}\n"
    # prompt = ""
    # for i in range(0,shots,1):
    #     prompt += Examplars[i] + "\n\n"
    # prompt += f"Q: {question}\nA: "
    # return prompt
    return f"{question}"

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def test_GSM8K_func(model, tokenizer, shots, file_name):
    # Create and open a CSV file for writing labeled responses
    csv_output_path = file_name + ".csv"
    with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["question", "ground_truth", "model_name", "response"])  # header row
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

    #%% Predicting the Answer

    def predict_answer(question, answer_text, answer_number, no_answer_count):
        prompt = create_cot_prompt(question, shots)
        print("\n\n")
        print("#" + str(label) + " prompt")
        print(prompt)
        print("\n\n")
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)# .to("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(inputs.device)

        # Generate response with chain-of-thought if desired
        # outputs = model.generate(
        #     **inputs,
        #     max_length=inputs["input_ids"].shape[-1] + 30,  # Adjust to include reasoning steps
        #     do_sample=True,
        #     temperature=0.7  # Adjust temperature for creativity in reasoning
        # )
            # outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[-1] + 256, use_cache = True, do_sample=False, num_beams=1, top_p=1, repetition_penalty=0.0001,)
            outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[-1] + 400, use_cache = True, do_sample=False)
        except RuntimeError as e:
            print("Error during generation:", e)
            return "", no_answer_count + 1
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
        found_text = str(answer_number)
        # slicing off the prompt 
        if found_text in response_sliced:
            chosen_answer = found_text
        else:
            chosen_answer = "none"
            # print("\n-----False 1-----\n")

        # Check if the string is not int
        if not is_int(chosen_answer):
            no_answer_count += 1
            # return chosen_answer, no_answer_count
            return chosen_answer, no_answer_count, response_sliced
        else:
            # return int(chosen_answer), no_answer_count
            return int(chosen_answer), no_answer_count, response_sliced


    #%% Evaluate the Model on the CSQA Validation Set
    label = 1
    correct = 0
    no_answer_count = 0
    total = len(gsm8k["test"])

    print(f"Total number of validation data: {total}")

    # Track total validation time
    start_time = time.time()

    for entry in gsm8k["test"]:
        question = entry["question"]
        answer_text = entry["answer"]
        stop = "\n#### "
        answer_text = answer_text.replace(',', '')  # 2,125 → 2125
        answer_number = int(answer_text.split(stop)[1].strip())
        # Predict answer
        # predicted_answer, no_answer_count = predict_answer(question, answer_text, answer_number, no_answer_count)
        predicted_answer, no_answer_count, response_sliced = predict_answer(question, answer_text, answer_number, no_answer_count)

        # Parse response for best answer (this can vary based on CoT structure)
        # chosen_answer_incomplete = response_sliced.split("the answer is ")[1].strip()
        # chosen_answer = chosen_answer_incomplete.split(".")[0].strip()
        print("#" + str(label) + " chosen answer")
        print(predicted_answer)
        print("\n")
        print("correct answer")
        print(answer_number)
        print("\n")
        
        # Check correctness
        correct += (predicted_answer == answer_number)
        
        # Append to CSV for manual evaluation
        with open(csv_output_path, mode="a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                question.strip(),
                str(answer_number),
                model.config._name_or_path if hasattr(model, "config") else "UnknownModel",
                response_sliced.strip().replace('\n', ' ')
            ])

        
        print(f"correct count: {correct}")
        print(f"total count: {label}")
        accuracy = correct / label
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        
        print(f"no answer count: {no_answer_count}")
        wrong_answer_count = label - no_answer_count - correct
        print(f"wrong answer count: {wrong_answer_count}")
        
        label += 1
        
    end_time = time.time()
    print(f"Total validation time: {end_time - start_time:.2f} seconds")