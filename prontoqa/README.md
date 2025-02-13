# Testing the deductive reasoning ability on the PrOntoQA-OOD dataset
The codes related to the accuracy of the 23 considered language models (LMs) using the Proof and Ontology-Generated Question-Answering-Out-Of-Demonstration (PrOntoQA-OOD) dataset are in the [Testing the deductive reasoning ability of 23 different language models section](#Testing-the-deductive-reasoning-ability-of-23-different-language-models).

The codes related to the finetuning and the accuracy of the finetuned models are in the [Fine-Tuning Improves Sub-Threshold Models section](#Fine-Tuning-Improves-Sub-Threshold-Models).

The codes related to the attention map analysis are in the [Attention_analysis folder](Attention_analysis/README.md).

## Testing the deductive reasoning ability of 23 different language models
Each python file is named in a format of {model_name} + ".py". We used eight premise, conclusion, and gold Chain-of-Thought (CoT) tuples to prompt the models from SmolLM2, OpenELM, TinyLlama, Stable LM 2, and Gemma 2, while we used three premise, conclusion, and gold CoT tuples to prompt the models from GPT2, due to the limit on their context length of 1024. We extracted the generated CoT from the output of the LM and compare it to the gold CoT from the PrOntoQA-OOD dataset. The exemplars used in each question are different and are generated by the PrOntoQA-OOD dataset. Every LMs shares the same exemplars and test proof pairs. We tested the LMs on 100 proofs.

In order to run the python file with OpenELM and gemma2, we need to enter the Hugging Face access token on the right side of "access_token = ". We can get the Hugging Face access token by following the instruction in [their tutorial](https://huggingface.co/docs/hub/security-tokens).

Please report any issues if you encounter any problems.
### Commands
To get the responses from each model with a specific deductive rule, we used
```
python run_experiment.py --model-name {model_name} --distractors none --test-distractors none --num-trials 100 --few-shot-examples 8 --proofs-only --max-hops 1 --deduction-rule {deductive_rule}
```
Please use one of the model name in the [Language models section](#Language_models) to replace {model_name} and one of the name of the deductive rules in the [Deductive rules section](#Deductive_rules) to replace {deductive_rule}. They are also shown in the SH file with the name in a format of "run_experiment_" + {model_name} + ".sh".

To get the statistics of each of the experiment, please use the following command if the completed running the experiment "run_experiment.py":
```
python analyze_results.py <log_file>
```
The <log_file> is the one that generated from "run_experiment.py" once the code is completed running.
### Outputs
By running each python file, "run_experiment.py", the output of each question will show: 
1. The input text for the language model
2. Number of tokens of the input text (On the right of "Context length:")
3. The response of the LM
4. The sliced response of the language model (On the right of the "Predicted answer")
5. Expected answer
6. The number of the task (On the right of "n: ")

At the end of the output, you will see the "Total validation time" in second to report the computational time. We saved the output from each model as an OUT file in a folder with the name the same as the model names listed in the next section. The OUT file is named in a format of "({model_name_abbr})R-run_experiment_{model_name}.sh_{task_number}", where {model_name_abbr} is the abbreviation shown in the [Deductive rules section](#Deductive_rules), {model_name} is the same as the one in the corresponding command, and {task_number} is the task number assigned by the slurm from the High-performance computing (HPC) I used.

By running each python file, "analyze_results.py", the output will include several rows of statistics. We used the "Proportion of proofs that contain the correct proof" to get the success rate of each task in the paper. We saved the output file as an OUT file in a folder with the name the same as the corresponding model names listed in the next section. The OUT file is named in a format of "({model_name_abbr})R-run_analyze_{model_name}.sh_{task_number}", where {model_name_abbr} is the abbreviation shown in the [Deductive rules section](#Deductive_rules), {model_name} is the same as the one in the corresponding command, and {task_number} is the task number assigned by the slurm from the High-performance computing (HPC) I used.

### Figures in the paper
We have recorded all the results manually in the "Plot_comparison_prontoqa.py" file. Please run
```
python Plot_comparison_prontoqa.py
```
to get the figures we used in the paper.

### Language_models
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

### Deductive_rules
1. ModusPonens
2. AndIntro
3. AndElim
4. OrIntro
5. OrElim
6. ProofByContra

The names of the six deduction rules above correspond to the implication elimination (also known as modus ponens) (MP), conjunction introduction (AI), conjunction elimination (AE), disjunction introduction (OI), disjunction elimination (also called proof by cases) (OE), and proof by contradiction (PBC), respectively.

### Exemplars
We show the exemplars used in the first question in our task as an intance:
```
Q: Each lempus is wooden. Max is a lempus. Prove: Max is wooden.
A: Max is a lempus. Each lempus is wooden. Max is wooden.

Q: Gorpuses are not muffled. Rex is a gorpus. Prove: Rex is not muffled.
A: Rex is a gorpus. Gorpuses are not muffled. Rex is not muffled.

Q: Shumpuses are fruity. Polly is a shumpus. Prove: Polly is fruity.
A: Polly is a shumpus. Shumpuses are fruity. Polly is fruity.

Q: Vumpuses are not hot. Wren is a vumpus. Prove: Wren is not hot.
A: Wren is a vumpus. Vumpuses are not hot. Wren is not hot.

Q: Wumpuses are not cold. Rex is a wumpus. Prove: Rex is not cold.
A: Rex is a wumpus. Wumpuses are not cold. Rex is not cold.

Q: Each numpus is melodic. Rex is a numpus. Prove: Rex is melodic.
A: Rex is a numpus. Each numpus is melodic. Rex is melodic.

Q: Lorpuses are discordant. Max is a lorpus. Prove: Max is discordant.
A: Max is a lorpus. Lorpuses are discordant. Max is discordant.

Q: Each shumpus is transparent. Max is a shumpus. Prove: Max is transparent.
A: Max is a shumpus. Each shumpus is transparent. Max is transparent.
```
We append the test proof and the conclusion of the test proof below the corresponding exemplars following the same format as shown in the instance of the exemplars.

## Fine-Tuning Improves Sub-Threshold Models
The folder related to the fine-tuning of the 6 language models,gpt2, gpt2_medium, OpenELM_270M, OpenELM_270M_Instruct, SmolLM2_135M, and SmolLM2_135M_Instruct are in the folder with their name on it. We name the folder related to the fine-tuning of the 6 language models in a format of {model_name} + "_ft_ALL.py", where the {model_name} is one of the names listed in the first line in this paragraph. We can fine-tune the model with 100 epochs using the following command:
```
python {model_name}_ft_ALL.py
```
After the model is finished training, we will get two folders with "checkpoint-{best_number}" and "checkpoint-1700". Please copy the path of the "checkpoint-{best_number}" folder, it may look like "{path_you_saved_this_repository}/CoT_Reasoning_Test/prontoqa/gpt2-finetuned-prontoqa/checkpoint-{best_number}", and then paste it at the begginging of the "{model_name}_load_best_ft_ALL.py" file with "model = AutoModelForCausalLM.from_pretrained" by the following line
```
model = AutoModelForCausalLM.from_pretrained({paste_here}, device_map="auto")
```
, where {paste_here} is the place you should paste the copied path of the folde.
Next, we run the fine-tuned model on three premise, conclusion, and gold CoT tuples from the PrOntoQA-OOD dataset as
```
python run_experiment.py --model-name {model_name}_best_ft_ALL --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --disable-examples True --proofs-only --max-hops 1 --deduction-rule {deductive_rule}
```
Please use one of the model name in the [Language models section](#Language_models) to replace {model_name} and one of the name of the deductive rules in the [Deductive rules section](#Deductive_rules) to replace {deductive_rule}. They are also shown in the SH file with the name in a format of "run_experiment_" + {model_name} + "_best_ft_ALL.sh".
To get the statistics of the result, please run
```
python analyze_results.py <log_file>
```
The <log_file> is the one that generated from "run_experiment.py" once the code is completed running.

The explaination of the output of "run_experiment.py" and "analyze_results.py" is the same as in the [Outputs section](#Outputs). We saved the output generated by "run_experiment.py" and  "analyze_results.py" in "R-run_experiment_{model_name}\_load_best_ft_ALL.sh_{task_number}.out" and "R-run_analyze_{model_name}\_best_ft_ALL.sh_{task_number}.out", respectively, where {task_number} is the task number assigned by the slurm from the High-performance computing (HPC) I used.

### Figures in the paper
We have recorded all the results manually in the "Plot_comparison_prontoqa_finetuned.py" file. Please run
```
python Plot_comparison_prontoqa_finetuned.py
```
to get the figures we used in the paper.

## References
**PrOntoQA-OOD**
```
@inproceedings{NEURIPS2023_09425891,
 author = {Saparov, Abulhair and Pang, Richard Yuanzhe and Padmakumar, Vishakh and Joshi, Nitish and Kazemi, Mehran and Kim, Najoung and He, He},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {3083--3105},
 publisher = {Curran Associates, Inc.},
 title = {Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/09425891e393e64b0535194a81ba15b7-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```
### GPT 2
```
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya and others},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}
```
### SmolLM2
```
@misc{allal2024SmolLM2,
      title={SmolLM2 - with great data, comes great performance}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Lewis Tunstall and Agustín Piqueres and Andres Marafioti and Cyril Zakka and Leandro von Werra and Thomas Wolf},
      year={2024}
}
```
### OpenELM
```
@inproceedings{
mehta2024openelm,
title={Open{ELM}: An Efficient Language Model Family with Open Training and Inference Framework},
author={Sachin Mehta and Mohammad Hossein Sekhavat and Qingqing Cao and Maxwell Horton and Yanzi Jin and Chenfan Sun and Seyed Iman Mirzadeh and Mahyar Najibi and Dmitry Belenko and Peter Zatloukal and Mohammad Rastegari},
booktitle={Workshop on Efficient Systems for Foundation Models II @ ICML2024},
year={2024},
url={https://openreview.net/forum?id=XNMbTkxroF}
}
```
### TinyLlama
```
@misc{zhang2024tinyllama,
      title={TinyLlama: An Open-Source Small Language Model}, 
      author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
      year={2024},
      eprint={2401.02385},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.02385}, 
}
```
### Stable LM 2 
```
@misc{bellagente2024stable,
      title={Stable LM 2 1.6B Technical Report}, 
      author={Marco Bellagente and Jonathan Tow and Dakota Mahan and Duy Phung and Maksym Zhuravinskyi and Reshinth Adithyan and James Baicoianu and Ben Brooks and Nathan Cooper and Ashish Datta and Meng Lee and Emad Mostaque and Michael Pieler and Nikhil Pinnaparju and Paulo Rocha and Harry Saini and Hannah Teufel and Niccolo Zanichelli and Carlos Riquelme},
      year={2024},
      eprint={2402.17834},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.17834}, 
}
```
### Gemma 2
```
@misc{team2024gemma,
      title={Gemma 2: Improving Open Language Models at a Practical Size}, 
      author={Gemma Team and Morgane Riviere and Shreya Pathak and Pier Giuseppe Sessa and Cassidy Hardin and Surya Bhupatiraju and Léonard Hussenot and Thomas Mesnard and Bobak Shahriari and Alexandre Ramé and Johan Ferret and Peter Liu and Pouya Tafti and Abe Friesen and Michelle Casbon and Sabela Ramos and Ravin Kumar and Charline Le Lan and Sammy Jerome and Anton Tsitsulin and Nino Vieillard and Piotr Stanczyk and Sertan Girgin and Nikola Momchev and Matt Hoffman and Shantanu Thakoor and Jean-Bastien Grill and Behnam Neyshabur and Olivier Bachem and Alanna Walton and Aliaksei Severyn and Alicia Parrish and Aliya Ahmad and Allen Hutchison and Alvin Abdagic and Amanda Carl and Amy Shen and Andy Brock and Andy Coenen and Anthony Laforge and Antonia Paterson and Ben Bastian and Bilal Piot and Bo Wu and Brandon Royal and Charlie Chen and Chintu Kumar and Chris Perry and Chris Welty and Christopher A. Choquette-Choo and Danila Sinopalnikov and David Weinberger and Dimple Vijaykumar and Dominika Rogozińska and Dustin Herbison and Elisa Bandy and Emma Wang and Eric Noland and Erica Moreira and Evan Senter and Evgenii Eltyshev and Francesco Visin and Gabriel Rasskin and Gary Wei and Glenn Cameron and Gus Martins and Hadi Hashemi and Hanna Klimczak-Plucińska and Harleen Batra and Harsh Dhand and Ivan Nardini and Jacinda Mein and Jack Zhou and James Svensson and Jeff Stanway and Jetha Chan and Jin Peng Zhou and Joana Carrasqueira and Joana Iljazi and Jocelyn Becker and Joe Fernandez and Joost van Amersfoort and Josh Gordon and Josh Lipschultz and Josh Newlan and Ju-yeong Ji and Kareem Mohamed and Kartikeya Badola and Kat Black and Katie Millican and Keelin McDonell and Kelvin Nguyen and Kiranbir Sodhia and Kish Greene and Lars Lowe Sjoesund and Lauren Usui and Laurent Sifre and Lena Heuermann and Leticia Lago and Lilly McNealus and Livio Baldini Soares and Logan Kilpatrick and Lucas Dixon and Luciano Martins and Machel Reid and Manvinder Singh and Mark Iverson and Martin Görner and Mat Velloso and Mateo Wirth and Matt Davidow and Matt Miller and Matthew Rahtz and Matthew Watson and Meg Risdal and Mehran Kazemi and Michael Moynihan and Ming Zhang and Minsuk Kahng and Minwoo Park and Mofi Rahman and Mohit Khatwani and Natalie Dao and Nenshad Bardoliwalla and Nesh Devanathan and Neta Dumai and Nilay Chauhan and Oscar Wahltinez and Pankil Botarda and Parker Barnes and Paul Barham and Paul Michel and Pengchong Jin and Petko Georgiev and Phil Culliton and Pradeep Kuppala and Ramona Comanescu and Ramona Merhej and Reena Jana and Reza Ardeshir Rokni and Rishabh Agarwal and Ryan Mullins and Samaneh Saadat and Sara Mc Carthy and Sarah Cogan and Sarah Perrin and Sébastien M. R. Arnold and Sebastian Krause and Shengyang Dai and Shruti Garg and Shruti Sheth and Sue Ronstrom and Susan Chan and Timothy Jordan and Ting Yu and Tom Eccles and Tom Hennigan and Tomas Kocisky and Tulsee Doshi and Vihan Jain and Vikas Yadav and Vilobh Meshram and Vishal Dharmadhikari and Warren Barkley and Wei Wei and Wenming Ye and Woohyun Han and Woosuk Kwon and Xiang Xu and Zhe Shen and Zhitao Gong and Zichuan Wei and Victor Cotruta and Phoebe Kirk and Anand Rao and Minh Giang and Ludovic Peran and Tris Warkentin and Eli Collins and Joelle Barral and Zoubin Ghahramani and Raia Hadsell and D. Sculley and Jeanine Banks and Anca Dragan and Slav Petrov and Oriol Vinyals and Jeff Dean and Demis Hassabis and Koray Kavukcuoglu and Clement Farabet and Elena Buchatskaya and Sebastian Borgeaud and Noah Fiedel and Armand Joulin and Kathleen Kenealy and Robert Dadashi and Alek Andreev},
      year={2024},
      eprint={2408.00118},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.00118}, 
}
```
