# Testing the commensence question answering ability of 23 different language models
The codes related to the CommonsenseQA (CSQA) dataset.
## Contents
Each python file is named in a format of "CoT_" + {model_name} + ".py". We used [the exemplar](#Exemplar_from_the_CoT_prompting_paper) from the [Chain-of-Thought (CoT) prompting paper](#CoT_prompting) to prompt the language models (LMs). We extract the chosen answer from the output of the LM and compare it to the answer key from the CSQA dataset. We used the validation subset in the CSQA dataset with 1221 questions.

In order to run the python file with OpenELM and gemma2, we need to enter the Hugging Face access token on the right side of "access_token = ". We can get the Hugging Face access token by following the instruction in [their tutorial](https://huggingface.co/docs/hub/security-tokens).
### Outputs
By running each python file, the output of each question will show: 
1. The input text for the language model (LM)
2. The response of the LM
3. The sliced response of the language model
4. The chosen answer
5. The correct answer
6. The correct count
7. The total count
8. The Validation Accuracy
9. The no answer count
10. The wrong answer count

At the end of the output, you will see the "Total validation time" in second to report the computational time.
### Language models
We consider 23 different language models included:
1. gpt2
2. gpt2-medium
3. gpt2-large
4. gpt-xl
5. SmolLM1-135M
6. SmolLM1-135M-Instruct
7. SmolLM1-360M
8. SmolLM1-360M-Instruct
9. SmolLM1-1.7B
10. SmolLM1-1.7B-Instruct
11. OpenELM-270M
12. OpenELM-270M-Instruct
13. OpenELM-360M
14. OpenELM-360M-Instruct
15. OpenELM-1_1B
16. OpenELM-1_1B-Instruct
17. OpenELM-3B
18. OpenELM-3B-Instruct
19. TinyLlama_v1_1
20. stablelm-2-1_6b
21. stablelm-2-zephyr-1_6b
22. gemma-2-2b-it
23. gemma-2-9b-it

Please report any issues if you encounter any problems.

### Exemplar from the CoT prompting paper
```
Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: (a) shirt pocket
(b) calligrapher's hand
(c) inkwell
(d) desk drawer
(e) blotter
A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).
Q: What home entertainment equipment requires cable? Answer Choices: (a) radio shack
(b) substation
(c) television
(d) cabinet
A: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).
Q: The fox walked from the city into the forest, what was it looking for? Answer Choices: (a) pretty flowers
(b) hen house
(c) natural habitat
(d) storybook
A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b).
Q: Sammy wanted to go to where the people were. Where might he go? Answer Choices: (a) populated areas
(b) race track
(c) desert
(d) apartment
(e) roadblock
A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).
Q: Where do you put your grapes just before checking out? Answer Choices: (a) mouth
(b) grocery cart
(c)super market
(d) fruit basket
(e) fruit market
A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).
Q: Google Maps and other highway and street GPS services have replaced what? Answer Choices: (a) united states
(b) mexico
(c) countryside
(d) atlas
A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).
Q: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (a) harder
(b) anguish
(c) bitterness
(d) tears
(e) sadness
A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c).
```

## References
### CoT_prompting
```
@inproceedings{NEURIPS2022_9d560961,
 author = {Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and ichter, brian and Xia, Fei and Chi, Ed and Le, Quoc V and Zhou, Denny},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {24824--24837},
 publisher = {Curran Associates, Inc.},
 title = {Chain-of-Thought Prompting Elicits Reasoning in Large Language Models},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
