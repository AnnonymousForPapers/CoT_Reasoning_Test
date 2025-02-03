# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:04:38 2025

@author: xiaoyenche
"""

import numpy as np
import matplotlib.pyplot as plt

model_names = ["gpt2","gpt2-finetuned-best","gpt2-finetuned-last","SmolLM2-135M","SmolLM2-135M-finetuned-best","SmolLM2-135M-finetuned-last","SmolLM2-135M-Instruct","SmolLM2-135M-Instruct-finetuned-best","SmolLM2-135M-Instruct-finetuned-last","OpenELM-270M","OpenELM-270M-finetuned-best","OpenELM-270M-finetuned-last","OpenELM-270M-Instruct","OpenELM-270M-Instruct-finetuned-best","OpenELM-270M-Instruct-finetuned-last","gpt2-medium","gpt2-medium-finetuned-best","gpt2-medium-finetuned-last"]
# Total 100 questions

steps_accuracy = {
    'Implication elimination': (63.37,100,100,80,95.83,91.06,71.88,98,94.75,76.92,92.69,91.13,88.97,77.82,90.07,77.24,100,100),
    'Conjunction introduction': (84,100,100,93.31,98.53,97.04,91.24,98.63,96.84,93.55,99.62,98.20,90.32,87.10,93.77,91.67,100,100),
    'Conjunction elimination': (99.41,100,100,98.48,100,98.45,98.98,100,97.97,100,100,100,100,99.50,99.23,100,100,100),
    'Disjunction introduction': (100,100,100,95.80,100,98.5,96,99.50,94.71,86.09,99.52,99,83.63,98.52,97.53,100,100,100),
    'Disjunction elimination': (75.56,49.17,48.37,86.95,59.88,50.44,80.23,43.92,45.30,87.10,95.15,98.51,86.07,71.70,73.41,94.87,34.71,37.42),
    'Proof by contradiction': (50.16,56.07,46.52,45.40,38.80,45.78,44.78,37.92,26.44,49.72,72.08,80.40,51.61,32.93,35.06,47.38,28.17,39.87),
    }

example_accuracy = {
    'Implication elimination': (51,100,100,69,94,90,70,96,91,76,93,79,66,91,83,97,100,100),
    'Conjunction introduction': (72,100,100,87,100,94,89,100,80,83,100,88,73,100,98,81,100,100),
    'Conjunction elimination': (95,100,100,91,100,99,90,100,96,90,100,100,94,99,100,98,100,100),
    'Disjunction introduction': (100,100,100,99,100,96,98,99,88,90,99,95,95,97,95,100,100,100),
    'Disjunction elimination': (0,0,0,19,0,0,13,0,0,41,66,34,30,4,8,0,0,0),
    'Proof by contradiction': (0,0,0,3,0,9,3,42,9,3,46,8,11,12,16,0,0,0),
    }

mask = []
for name in model_names:
    if "last" in name:
        mask.append(0)
    else:
        mask.append(1)

# Select elements where the mask is 1
model_names = [model_names[i] for i in range(len(mask)) if mask[i] == 1]

steps_accuracy = {
    key: tuple(value[i] for i in range(len(mask)) if mask[i] == 1)
    for key, value in steps_accuracy.items()
}

example_accuracy = {
    key: tuple(value[i] for i in range(len(mask)) if mask[i] == 1)
    for key, value in example_accuracy.items()
}

print(model_names)  # Output: [1, 6]

x = np.arange(len(model_names))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

# Define custom colors for each attribute
colors = {
    'Implication elimination': '#0173b2',
    'Conjunction introduction': '#de8f05',
    'Conjunction elimination': '#029e73',
    'Disjunction introduction': '#d55e00',
    'Disjunction elimination': '#cc78bc',
    'Proof by contradiction': '#ca9161',
}

fig, ax = plt.subplots(layout='constrained', figsize=(19, 10))

for attribute, measurement in example_accuracy.items():
    offset = width * multiplier
    rects = ax.bar(x + offset , measurement, width, label=attribute, color=colors[attribute])
    ax.bar_label(rects, padding=6, rotation=45, weight='bold')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Validation accuracy (%)', fontsize=16)
ax.set_title('Comparison of Percentage of Correct Proofs for Various LMs and their finetuned models on the ProntoQA Dataset', fontsize=20)
ax.set_xticks(x + 5/2*width
              , model_names, rotation=90, fontsize=16)
ax.legend(ncol=6, fontsize=12, loc='upper left')
ax.set_ylim(0, 119)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0)
plt.show()

# #%%

# validation_accuracy = [0.08,20.8,17.94,2.38,0.66,0.33,22.44,24,5.41,8.11,22.03,10.73,14.66,20.48,19.41,43.57,61.18,
#                        54.87,59.95,68.88,6.39,0.08,75.27,]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

# import matplotlib.pyplot as plt

# colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

# markers = ["o","v","s","P","*","D"]

# hatches = ['/', '|', '-', '+', 'x', 'o']

# model_parameters = [
#     124000000,  # gpt2 "o" '#0173b2'
#     135000000,  # SmolLM2-135M "v" '#0173b2'
#     135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
#     270000000,  # OpenELM-270M "s" '#0173b2'
#     270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
#     355000000,  # gpt2-medium "o" '#de8f05'
#     360000000,  # SmolLM2-360M "v" '#029e73'
#     360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
#     450000000,  # OpenELM-450M "s" '#029e73'
#     450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
#     774000000,  # gpt2-large "o" '#029e73'
#     1100000000,  # OpenELM-1_1B "s" '#cc78bc'
#     1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
#     1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
#     1500000000,  # gpt2-xl "o" '#d55e00'
#     1600000000,  # stablelm-2-1_6b "*" '#0173b2'
#     1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
#     1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
#     1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
#     2000000000,  # gemma-2-2b-it "D" '#0173b2'
#     3000000000,  # OpenELM-3B "s" '#fbafe4'
#     3000000000,  # OpenELM-3B-Instruct "s" '#949494'
#     9000000000,  # gemma-2-9b-it "D" '#de8f05'
# ]

# # Define custom X-axis labels and ticks to better align with the model parameter magnitudes
# ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
# labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

# developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
# hatch_list = []
# color_list = []
# models_info = {}
# gpt_cnt = 0
# SmolLM2_cnt = 0
# OpenELM_cnt = 0
# TinyLlama_cnt = 0
# stablelm_cnt = 0
# gemma_2_cnt = 0
# for index, name in enumerate(model_names):
#     if "gpt2" in name:
#         color = colors[gpt_cnt]
#         marker = markers[0]
#         hatch_list.append(hatches[0])
#         color_list.append(colors[0])
#         gpt_cnt += 1
#     elif "SmolLM2" in name:
#         color = colors[SmolLM2_cnt]
#         marker = markers[1]
#         hatch_list.append(hatches[1])
#         color_list.append(colors[1])
#         SmolLM2_cnt += 1
#     elif "OpenELM" in name:
#         color = colors[OpenELM_cnt]
#         marker = markers[2]
#         hatch_list.append(hatches[2])
#         color_list.append(colors[2])
#         OpenELM_cnt += 1
#     elif "TinyLlama" in name:
#         color = colors[TinyLlama_cnt]
#         marker = markers[3]
#         hatch_list.append(hatches[3])
#         color_list.append(colors[3])
#         TinyLlama_cnt += 1
#     elif "stablelm" in name:
#         color = colors[stablelm_cnt]
#         marker = markers[4]
#         hatch_list.append(hatches[4])
#         color_list.append(colors[4])
#         stablelm_cnt += 1
#     elif "gemma-2" in name:
#         color = colors[gemma_2_cnt]
#         marker = markers[5]
#         hatch_list.append(hatches[5])
#         color_list.append(colors[5])
#         gemma_2_cnt += 1
#     else:
#         raise Exception("Name not recognized")
#     # color_list.append(color)
#     models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index], "correct_cnt": correct_cnt[index], "no_answer_cnt": no_answers[index], "wrong_cnt": wrong_cnt[index], "time": time[index]}

# # Create a figure and a set of subplots
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# # Scatter Plot for Validation Accuracy vs Number of Model Parameters
# for i, name in enumerate(model_names):
#     ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
# ax.set_xlabel("Number of Model Parameters", fontsize=16)
# ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for\n Various LMs on the CSQA Dataset", fontsize=20)
# ax.set_xticks(ticks, labels)
# ax.set_xscale('log')
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
# ax.legend(ncol=2, fontsize=11, loc='upper left')
# plt.tight_layout()

# #%% Bar plots
# # Create legend handles manually
# import matplotlib.patches as mpatches
# handles = []
# for i, name in enumerate(developer_names):
#     patch = mpatches.Patch(color=colors[i], label=name)
#     handles.append(patch)


# # Create a figure and a set of subplots
# # Plotting the bar plot
# plt.figure(figsize=(12, 6))
# # plt.bar(model_names, correct_cnt, color = 'w', edgecolor='k', hatch = hatch_list)
# plt.bar(model_names, correct_cnt, color = color_list)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("Model Names", fontsize=16)
# plt.ylabel("Number of Correct Answers", fontsize=16)
# plt.title("Number of Correct Answers for Each Model", fontsize=20)
# plt.legend(handles=handles, fontsize=12)
# plt.tight_layout()
# plt.show()

# # Create a figure and a set of subplots
# # Plotting the bar plot
# plt.figure(figsize=(12, 6))
# plt.bar(model_names, no_answers, color = color_list)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("Model Names", fontsize=16)
# plt.ylabel("Number of No Answers", fontsize=16)
# plt.title("Number of No Answers for Each Model", fontsize=20)
# plt.legend(handles=handles, fontsize=12)
# plt.tight_layout()
# plt.show()

# # Create a figure and a set of subplots
# # Plotting the bar plot
# plt.figure(figsize=(12, 6))
# plt.bar(model_names, wrong_cnt, color = color_list)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("Model Names", fontsize=16)
# plt.ylabel("Number of Incorrect Answers", fontsize=16)
# plt.title("Number of Incorrect Answers for Each Model", fontsize=20)
# plt.legend(handles=handles, fontsize=12)
# plt.tight_layout()
# plt.show()

# # Create a figure and a set of subplots
# # Plotting the bar plot
# plt.figure(figsize=(12, 6))
# plt.bar(model_names, time, color = color_list)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("Model Names", fontsize=16)
# plt.ylabel("Computational Time (sec)", fontsize=16)
# plt.title("Computational Time for Each Model", fontsize=20)
# plt.legend(handles=handles, fontsize=12)
# plt.tight_layout()
# plt.show()