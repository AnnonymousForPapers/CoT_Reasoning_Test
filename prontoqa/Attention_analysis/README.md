# Attention Map Analysis
The codes related to generate the attention map, attention map with lines connection, and the [visualization of the token-level scores](#Token-level-scores-computation) of the gpt2 and gemma2-9b-it models given the first prompt of the implication elimination task from the PrOntoQA-OOD dataset. The prompt is shown below:
```
Q: Every impus is not floral. Alex is an impus. Prove: Alex is not floral.
A: Alex is an impus. Every impus is not floral. Alex is not floral.

Q: Every jompus is not loud. Rex is a jompus. Prove: Rex is not loud.
A: Rex is a jompus. Every jompus is not loud. Rex is not loud.

Q: Each tumpus is not liquid. Rex is a tumpus. Prove: Rex is not liquid.
A: Rex is a tumpus. Each tumpus is not liquid. Rex is not liquid.

Q: Sterpuses are transparent. Wren is a sterpus. Prove: Wren is transparent.
A: Wren is a sterpus. Sterpuses are transparent. Wren is
```

To get the attention map from gpt2, please run
```
python gpt2_MP1_wrong_out_generated_tokens_plot.py
```
The output figures will be stored in the "Attention_analysis\images\gpt2" folder. 

In order to run the python file with gemma2, we need to enter the Hugging Face access token on the right side of "access_token = ". We can get the Hugging Face access token by following the instruction in [their tutorial](https://huggingface.co/docs/hub/security-tokens).
To get the attention map from gemma2-9b-it, please run
```
python gemma2_9b_it_MP1_correct_out_generated_tokens_plot.py
```
The output figures will be stored in the "Attention_analysis\images\gemma2-9b-it" folder. 

## Reference
### Token-level scores computation
```
@inproceedings{kang-shin-2023-samrank,
    title = "{SAMR}ank: Unsupervised Keyphrase Extraction using Self-Attention Map in {BERT} and {GPT}-2",
    author = "Kang, Byungha  and
      Shin, Youhyun",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.630/",
    doi = "10.18653/v1/2023.emnlp-main.630",
    pages = "10188--10201"
}
```
