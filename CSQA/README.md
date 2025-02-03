# Testing the commensence question answering ability of 23 different language models
The codes related to the CommonsenseQA (CSQA) dataset.
## Contents
Each python file is named in a format of "CoT_" + {model_name} + ".py". We used [the exemplar](#Exemplar) from the [Chain-of-Thought (CoT) prompting paper](#CoT_prompting) to prompt the language models (LMs). We extract the chosen answer from the output of the LM and compare it to the answer key from the CSQA dataset. We used the validation subset in the CSQA dataset with 1221 questions.

In order to run the python file with OpenELM and gemma2, we need to enter the Hugging Face access token on the right side of "access_token = ". We can get the Hugging Face access token by following the instruction in [their tutorial](https://huggingface.co/docs/hub/security-tokens).

Please report any issues if you encounter any problems.
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

### Exemplar
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
We append the test question and the answer keys of the test question below the exemplar following the same format as shown in the exemplar.
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
