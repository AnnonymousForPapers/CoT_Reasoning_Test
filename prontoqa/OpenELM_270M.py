from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "apple/OpenELM-270M"
print("\n" + "Model name: " + model_name + "\n")

access_token = ""

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="cuda", token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, token=access_token)


def predict(prompt):
    
    inputs = tokenizer(
    [
         prompt
    ], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True, do_sample=False, top_p=1, repetition_penalty=0.0001,)
    out_1 = tokenizer.batch_decode(outputs)[0]
    # if i == 1:
    #     print("\n\n-----1st LLM output start-----\n")
    #     print(action_1)
    #     print("\n-----1st LLM output end-----\n\n")
    tokenized_context = tokenizer(prompt)
    input_ids = tokenized_context['input_ids']
    print("\nContext length:" + str(len(input_ids)))
    print('\n')
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
    
    return generated_string, None
