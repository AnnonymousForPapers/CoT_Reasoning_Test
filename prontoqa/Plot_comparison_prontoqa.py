import numpy as np
import matplotlib.pyplot as plt

model_names = ["gpt2","SmolLM2-135M","SmolLM2-135M-Instruct","OpenELM-270M","OpenELM-270M-Instruct","gpt2-medium","SmolLM2-360M","SmolLM2-360M-Instruct","OpenELM-450M","OpenELM-450M-Instruct","gpt2-large","OpenELM-1_1B","OpenELM-1_1B-Instruct","TinyLlama_v1_1","gpt2-xl","stablelm-2-1_6b","stablelm-2-zephyr-1_6b","SmolLM2-1.7B","SmolLM2-1.7B-Instruct","gemma-2-2b-it","OpenELM-3B","OpenELM-3B-Instruct","gemma-2-9b-it"]
# Total 100 questions

steps_accuracy = {
    'Implication elimination': (63.37,84.28,85.62,97.64,93.56,77.24,99.67,100,98.32,86.02,91.64,100,90.70,99.67,82.42,71.29,72.62,100,100,100,99.67,97.33,100),
    'Conjunction introduction': (84,94.93,93.64,98.28,95.77,91.67,100,98.99,93.97,85.77,96.07,99.33,99.66,99.32,93.39,93.35,71.67,100,99.67,100,100,99.67,100),
    'Conjunction elimination': (99.41,100,100,100,100,100,100,100,100,99.49,100,100,98.99,100,100,65.28,99.15,100,100,100,100,100,100),
    'Disjunction introduction': (100,91.62,96.48,94.79,95.83,100,100,100,100,97.06,100,100,99.5,100,100,81.95,92.24,100,100,100,100,100,100),
    'Disjunction elimination': (75.56,92.94,81.80,89.06,88.82,94.87,97.25,91.88,94.77,91.06,96,98.62,96.22,93.20,97.55,99.50,74.16,99.88,99.5,100,99.88,99.75,100),
    'Proof by contradiction': (50.16,43.56,43.75,54.86,51.83,47.38,64.17,63.52,50.27,51.28,41.53,66.59,60.16,53.57,45.17,69.90,68.31,81.52,82.27,84.15,81.30,73.49,84.62),
    }

example_accuracy = {
    'Implication elimination': (51,68,77,96,78,97,98,100,97,81,87,100,72,99,81,90,99,100,100,100,99,92,100),
    'Conjunction introduction': (72,88,91,87,82,81,100,100,96,93,79,100,100,99,95,100,100,100,100,100,100,100,100),
    'Conjunction elimination': (95,95,96,98,98,98,100,100,100,96,100,100,96,100,100,100,100,100,100,100,100,100,100),
    'Disjunction introduction': (100,97,98,96,99,100,100,100,100,97,100,100,99,100,100,100,100,100,100,100,100,100,100),
    'Disjunction elimination': (0,37,19,40,39,0,72,53,50,26,0,90,53,63,49,95,69,98,98,100,98,98,100),
    'Proof by contradiction': (0,5,4,1,2,0,5,8,0,10,0,22,6,1,0,52,22,86,97,98,83,66,100),
    }

computational_time = {
    'Implication elimination': ('03:01','09:53','11:24','06:37','06:24','05:08','10:41','12:02','08:28','08:53','07:51','13:15','12:10','09:57','10:41','09:01','09:34','09:36','11:15','20:33','19:26','17:07','33:34'),
    'Conjunction introduction': ('02:57','10:05','11:17','06:02','06:04','05:08','10:19','11:59','07:42','07:27','07:46','12:53','12:17','07:34','10:12','08:42','09:18','09:32','09:45','20:21','19:02','16:49','33:26'),
    'Conjunction elimination': ('03:14','09:48','11:10','06:59','07:01','05:14','11:36','11:33','08:37','07:37','07:45','12:39','11:46','07:58','09:47','08:27','08:49','09:44','09:37','18:00','18:39','16:28','28:51'),
    'Disjunction introduction': ('03:00','11:15','11:12','07:17','06:50','05:34','11:34','11:32','07:48','08:45','08:03','10:40','10:13','07:44','10:10','08:46','08:31','09:41','10:03','17:32','16:24','14:18','28:21'),
    'Disjunction elimination': ('02:59','11:04','11:21','06:13','06:14','05:17','11:47','11:38','07:42','07:32','07:38','10:05','10:03','08:41','10:01','08:38','08:42','09:43','09:36','20:06','16:10','14:20','33:00'),
    'Proof by contradiction': ('02:59','11:18','11:24','07:04','07:05','05:16','11:50','11:39','08:28','08:33','07:39','10:23','10:08','08:49','10:16','08:54','08:52','10:04','09:53','20:29','16:55','14:44','33:3'),
    }

# Define custom colors for each attribute
colors = {
    'Implication elimination': '#0173b2',
    'Conjunction introduction': '#de8f05',
    'Conjunction elimination': '#029e73',
    'Disjunction introduction': '#d55e00',
    'Disjunction elimination': '#cc78bc',
    'Proof by contradiction': '#ca9161',
}

x = np.arange(len(model_names))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(19, 8))

for attribute, measurement in example_accuracy.items():
    offset = width * multiplier
    rects = ax.bar(x + offset , measurement, width, label=attribute, color=colors[attribute])
    ax.bar_label(rects, padding=6, rotation=90, weight='bold')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Validation accuracy (%)', fontsize=16)
ax.set_title('Comparison of Percentage of Correct Proofs for Various LMs on the ProntoQA Dataset', fontsize=20)
ax.set_xticks(x + 5/2*width
              , model_names, rotation=90, fontsize=16)
ax.legend(ncol=6, fontsize=12, loc='upper left')
ax.set_ylim(0, 119)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0)
plt.show()

#%% Computational time
def convert_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def convert_dict_to_seconds(computational_time):
    seconds_dict = {}
    for key, times in computational_time.items():
        seconds_dict[key] = [convert_to_seconds(time) for time in times]
    return seconds_dict

seconds_computational_time = convert_dict_to_seconds(computational_time)

# Print the result
for key, value in seconds_computational_time.items():
    print(f"{key}: {value}")
    
x = np.arange(len(model_names))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(19, 8))

for attribute, measurement in seconds_computational_time.items():
    offset = width * multiplier
    rects = ax.bar(x + offset , measurement, width, label=attribute, color=colors[attribute])
    ax.bar_label(rects, padding=6, rotation=90, weight='bold', fontsize=9)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Computational time (second)', fontsize=16)
ax.set_title('Computational time for Various LMs on the ProntoQA Dataset', fontsize=20)
ax.set_xticks(x + 5/2*width
              , model_names, rotation=90, fontsize=16)
ax.legend(ncol=6, fontsize=12, loc='upper left')
ax.set_ylim(0, 2249)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0)
plt.show()

#%%Implication elimination

rules = 'Implication elimination'

validation_accuracy = example_accuracy[rules]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

import matplotlib.pyplot as plt

colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

markers = ["o","v","s","P","*","D"]

hatches = ['/', '|', '-', '+', 'x', 'o']

model_parameters = [
    124000000,  # gpt2 "o" '#0173b2'
    135000000,  # SmolLM2-135M "v" '#0173b2'
    135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
    270000000,  # OpenELM-270M "s" '#0173b2'
    270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
    355000000,  # gpt2-medium "o" '#de8f05'
    360000000,  # SmolLM2-360M "v" '#029e73'
    360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
    450000000,  # OpenELM-450M "s" '#029e73'
    450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
    774000000,  # gpt2-large "o" '#029e73'
    1100000000,  # OpenELM-1_1B "s" '#cc78bc'
    1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
    1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
    1500000000,  # gpt2-xl "o" '#d55e00'
    1600000000,  # stablelm-2-1_6b "*" '#0173b2'
    1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
    1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
    1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
    2000000000,  # gemma-2-2b-it "D" '#0173b2'
    3000000000,  # OpenELM-3B "s" '#fbafe4'
    3000000000,  # OpenELM-3B-Instruct "s" '#949494'
    9000000000,  # gemma-2-9b-it "D" '#de8f05'
]

# Define custom X-axis labels and ticks to better align with the model parameter magnitudes
ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
hatch_list = []
color_list = []
models_info = {}
gpt_cnt = 0
SmolLM2_cnt = 0
OpenELM_cnt = 0
TinyLlama_cnt = 0
stablelm_cnt = 0
gemma_2_cnt = 0
for index, name in enumerate(model_names):
    if "gpt2" in name:
        color = colors[gpt_cnt]
        marker = markers[0]
        hatch_list.append(hatches[0])
        color_list.append(colors[0])
        gpt_cnt += 1
    elif "SmolLM2" in name:
        color = colors[SmolLM2_cnt]
        marker = markers[1]
        hatch_list.append(hatches[1])
        color_list.append(colors[1])
        SmolLM2_cnt += 1
    elif "OpenELM" in name:
        color = colors[OpenELM_cnt]
        marker = markers[2]
        hatch_list.append(hatches[2])
        color_list.append(colors[2])
        OpenELM_cnt += 1
    elif "TinyLlama" in name:
        color = colors[TinyLlama_cnt]
        marker = markers[3]
        hatch_list.append(hatches[3])
        color_list.append(colors[3])
        TinyLlama_cnt += 1
    elif "stablelm" in name:
        color = colors[stablelm_cnt]
        marker = markers[4]
        hatch_list.append(hatches[4])
        color_list.append(colors[4])
        stablelm_cnt += 1
    elif "gemma-2" in name:
        color = colors[gemma_2_cnt]
        marker = markers[5]
        hatch_list.append(hatches[5])
        color_list.append(colors[5])
        gemma_2_cnt += 1
    else:
        raise Exception("Name not recognized")
    # color_list.append(color)
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for Various LMs\n on the PrOntoQA-OOD Dataset with " + rules, fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='best')
plt.tight_layout()

#%%Conjunction introduction

rules = 'Conjunction introduction'

validation_accuracy = example_accuracy[rules]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

import matplotlib.pyplot as plt

colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

markers = ["o","v","s","P","*","D"]

hatches = ['/', '|', '-', '+', 'x', 'o']

model_parameters = [
    124000000,  # gpt2 "o" '#0173b2'
    135000000,  # SmolLM2-135M "v" '#0173b2'
    135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
    270000000,  # OpenELM-270M "s" '#0173b2'
    270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
    355000000,  # gpt2-medium "o" '#de8f05'
    360000000,  # SmolLM2-360M "v" '#029e73'
    360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
    450000000,  # OpenELM-450M "s" '#029e73'
    450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
    774000000,  # gpt2-large "o" '#029e73'
    1100000000,  # OpenELM-1_1B "s" '#cc78bc'
    1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
    1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
    1500000000,  # gpt2-xl "o" '#d55e00'
    1600000000,  # stablelm-2-1_6b "*" '#0173b2'
    1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
    1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
    1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
    2000000000,  # gemma-2-2b-it "D" '#0173b2'
    3000000000,  # OpenELM-3B "s" '#fbafe4'
    3000000000,  # OpenELM-3B-Instruct "s" '#949494'
    9000000000,  # gemma-2-9b-it "D" '#de8f05'
]

# Define custom X-axis labels and ticks to better align with the model parameter magnitudes
ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
hatch_list = []
color_list = []
models_info = {}
gpt_cnt = 0
SmolLM2_cnt = 0
OpenELM_cnt = 0
TinyLlama_cnt = 0
stablelm_cnt = 0
gemma_2_cnt = 0
for index, name in enumerate(model_names):
    if "gpt2" in name:
        color = colors[gpt_cnt]
        marker = markers[0]
        hatch_list.append(hatches[0])
        color_list.append(colors[0])
        gpt_cnt += 1
    elif "SmolLM2" in name:
        color = colors[SmolLM2_cnt]
        marker = markers[1]
        hatch_list.append(hatches[1])
        color_list.append(colors[1])
        SmolLM2_cnt += 1
    elif "OpenELM" in name:
        color = colors[OpenELM_cnt]
        marker = markers[2]
        hatch_list.append(hatches[2])
        color_list.append(colors[2])
        OpenELM_cnt += 1
    elif "TinyLlama" in name:
        color = colors[TinyLlama_cnt]
        marker = markers[3]
        hatch_list.append(hatches[3])
        color_list.append(colors[3])
        TinyLlama_cnt += 1
    elif "stablelm" in name:
        color = colors[stablelm_cnt]
        marker = markers[4]
        hatch_list.append(hatches[4])
        color_list.append(colors[4])
        stablelm_cnt += 1
    elif "gemma-2" in name:
        color = colors[gemma_2_cnt]
        marker = markers[5]
        hatch_list.append(hatches[5])
        color_list.append(colors[5])
        gemma_2_cnt += 1
    else:
        raise Exception("Name not recognized")
    # color_list.append(color)
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for Various LMs\n on the PrOntoQA-OOD Dataset with " + rules, fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='best')
plt.tight_layout()

#%%Conjunction elimination

rules = 'Conjunction elimination'

validation_accuracy = example_accuracy[rules]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

import matplotlib.pyplot as plt

colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

markers = ["o","v","s","P","*","D"]

hatches = ['/', '|', '-', '+', 'x', 'o']

model_parameters = [
    124000000,  # gpt2 "o" '#0173b2'
    135000000,  # SmolLM2-135M "v" '#0173b2'
    135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
    270000000,  # OpenELM-270M "s" '#0173b2'
    270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
    355000000,  # gpt2-medium "o" '#de8f05'
    360000000,  # SmolLM2-360M "v" '#029e73'
    360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
    450000000,  # OpenELM-450M "s" '#029e73'
    450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
    774000000,  # gpt2-large "o" '#029e73'
    1100000000,  # OpenELM-1_1B "s" '#cc78bc'
    1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
    1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
    1500000000,  # gpt2-xl "o" '#d55e00'
    1600000000,  # stablelm-2-1_6b "*" '#0173b2'
    1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
    1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
    1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
    2000000000,  # gemma-2-2b-it "D" '#0173b2'
    3000000000,  # OpenELM-3B "s" '#fbafe4'
    3000000000,  # OpenELM-3B-Instruct "s" '#949494'
    9000000000,  # gemma-2-9b-it "D" '#de8f05'
]

# Define custom X-axis labels and ticks to better align with the model parameter magnitudes
ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
hatch_list = []
color_list = []
models_info = {}
gpt_cnt = 0
SmolLM2_cnt = 0
OpenELM_cnt = 0
TinyLlama_cnt = 0
stablelm_cnt = 0
gemma_2_cnt = 0
for index, name in enumerate(model_names):
    if "gpt2" in name:
        color = colors[gpt_cnt]
        marker = markers[0]
        hatch_list.append(hatches[0])
        color_list.append(colors[0])
        gpt_cnt += 1
    elif "SmolLM2" in name:
        color = colors[SmolLM2_cnt]
        marker = markers[1]
        hatch_list.append(hatches[1])
        color_list.append(colors[1])
        SmolLM2_cnt += 1
    elif "OpenELM" in name:
        color = colors[OpenELM_cnt]
        marker = markers[2]
        hatch_list.append(hatches[2])
        color_list.append(colors[2])
        OpenELM_cnt += 1
    elif "TinyLlama" in name:
        color = colors[TinyLlama_cnt]
        marker = markers[3]
        hatch_list.append(hatches[3])
        color_list.append(colors[3])
        TinyLlama_cnt += 1
    elif "stablelm" in name:
        color = colors[stablelm_cnt]
        marker = markers[4]
        hatch_list.append(hatches[4])
        color_list.append(colors[4])
        stablelm_cnt += 1
    elif "gemma-2" in name:
        color = colors[gemma_2_cnt]
        marker = markers[5]
        hatch_list.append(hatches[5])
        color_list.append(colors[5])
        gemma_2_cnt += 1
    else:
        raise Exception("Name not recognized")
    # color_list.append(color)
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for Various LMs\n on the PrOntoQA-OOD Dataset with " + rules, fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='best')
plt.tight_layout()

#%%Disjunction introduction

rules = 'Disjunction introduction'

validation_accuracy = example_accuracy[rules]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

import matplotlib.pyplot as plt

colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

markers = ["o","v","s","P","*","D"]

hatches = ['/', '|', '-', '+', 'x', 'o']

model_parameters = [
    124000000,  # gpt2 "o" '#0173b2'
    135000000,  # SmolLM2-135M "v" '#0173b2'
    135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
    270000000,  # OpenELM-270M "s" '#0173b2'
    270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
    355000000,  # gpt2-medium "o" '#de8f05'
    360000000,  # SmolLM2-360M "v" '#029e73'
    360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
    450000000,  # OpenELM-450M "s" '#029e73'
    450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
    774000000,  # gpt2-large "o" '#029e73'
    1100000000,  # OpenELM-1_1B "s" '#cc78bc'
    1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
    1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
    1500000000,  # gpt2-xl "o" '#d55e00'
    1600000000,  # stablelm-2-1_6b "*" '#0173b2'
    1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
    1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
    1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
    2000000000,  # gemma-2-2b-it "D" '#0173b2'
    3000000000,  # OpenELM-3B "s" '#fbafe4'
    3000000000,  # OpenELM-3B-Instruct "s" '#949494'
    9000000000,  # gemma-2-9b-it "D" '#de8f05'
]

# Define custom X-axis labels and ticks to better align with the model parameter magnitudes
ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
hatch_list = []
color_list = []
models_info = {}
gpt_cnt = 0
SmolLM2_cnt = 0
OpenELM_cnt = 0
TinyLlama_cnt = 0
stablelm_cnt = 0
gemma_2_cnt = 0
for index, name in enumerate(model_names):
    if "gpt2" in name:
        color = colors[gpt_cnt]
        marker = markers[0]
        hatch_list.append(hatches[0])
        color_list.append(colors[0])
        gpt_cnt += 1
    elif "SmolLM2" in name:
        color = colors[SmolLM2_cnt]
        marker = markers[1]
        hatch_list.append(hatches[1])
        color_list.append(colors[1])
        SmolLM2_cnt += 1
    elif "OpenELM" in name:
        color = colors[OpenELM_cnt]
        marker = markers[2]
        hatch_list.append(hatches[2])
        color_list.append(colors[2])
        OpenELM_cnt += 1
    elif "TinyLlama" in name:
        color = colors[TinyLlama_cnt]
        marker = markers[3]
        hatch_list.append(hatches[3])
        color_list.append(colors[3])
        TinyLlama_cnt += 1
    elif "stablelm" in name:
        color = colors[stablelm_cnt]
        marker = markers[4]
        hatch_list.append(hatches[4])
        color_list.append(colors[4])
        stablelm_cnt += 1
    elif "gemma-2" in name:
        color = colors[gemma_2_cnt]
        marker = markers[5]
        hatch_list.append(hatches[5])
        color_list.append(colors[5])
        gemma_2_cnt += 1
    else:
        raise Exception("Name not recognized")
    # color_list.append(color)
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for Various LMs\n on the PrOntoQA-OOD Dataset with " + rules, fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='best')
plt.tight_layout()

#%%Disjunction elimination

rules = 'Disjunction elimination'

validation_accuracy = example_accuracy[rules]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

import matplotlib.pyplot as plt

colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

markers = ["o","v","s","P","*","D"]

hatches = ['/', '|', '-', '+', 'x', 'o']

model_parameters = [
    124000000,  # gpt2 "o" '#0173b2'
    135000000,  # SmolLM2-135M "v" '#0173b2'
    135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
    270000000,  # OpenELM-270M "s" '#0173b2'
    270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
    355000000,  # gpt2-medium "o" '#de8f05'
    360000000,  # SmolLM2-360M "v" '#029e73'
    360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
    450000000,  # OpenELM-450M "s" '#029e73'
    450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
    774000000,  # gpt2-large "o" '#029e73'
    1100000000,  # OpenELM-1_1B "s" '#cc78bc'
    1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
    1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
    1500000000,  # gpt2-xl "o" '#d55e00'
    1600000000,  # stablelm-2-1_6b "*" '#0173b2'
    1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
    1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
    1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
    2000000000,  # gemma-2-2b-it "D" '#0173b2'
    3000000000,  # OpenELM-3B "s" '#fbafe4'
    3000000000,  # OpenELM-3B-Instruct "s" '#949494'
    9000000000,  # gemma-2-9b-it "D" '#de8f05'
]

# Define custom X-axis labels and ticks to better align with the model parameter magnitudes
ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
hatch_list = []
color_list = []
models_info = {}
gpt_cnt = 0
SmolLM2_cnt = 0
OpenELM_cnt = 0
TinyLlama_cnt = 0
stablelm_cnt = 0
gemma_2_cnt = 0
for index, name in enumerate(model_names):
    if "gpt2" in name:
        color = colors[gpt_cnt]
        marker = markers[0]
        hatch_list.append(hatches[0])
        color_list.append(colors[0])
        gpt_cnt += 1
    elif "SmolLM2" in name:
        color = colors[SmolLM2_cnt]
        marker = markers[1]
        hatch_list.append(hatches[1])
        color_list.append(colors[1])
        SmolLM2_cnt += 1
    elif "OpenELM" in name:
        color = colors[OpenELM_cnt]
        marker = markers[2]
        hatch_list.append(hatches[2])
        color_list.append(colors[2])
        OpenELM_cnt += 1
    elif "TinyLlama" in name:
        color = colors[TinyLlama_cnt]
        marker = markers[3]
        hatch_list.append(hatches[3])
        color_list.append(colors[3])
        TinyLlama_cnt += 1
    elif "stablelm" in name:
        color = colors[stablelm_cnt]
        marker = markers[4]
        hatch_list.append(hatches[4])
        color_list.append(colors[4])
        stablelm_cnt += 1
    elif "gemma-2" in name:
        color = colors[gemma_2_cnt]
        marker = markers[5]
        hatch_list.append(hatches[5])
        color_list.append(colors[5])
        gemma_2_cnt += 1
    else:
        raise Exception("Name not recognized")
    # color_list.append(color)
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for Various LMs\n on the PrOntoQA-OOD Dataset with " + rules, fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='best')
plt.tight_layout()

#%%Proof by contradiction

rules = 'Proof by contradiction'

validation_accuracy = example_accuracy[rules]

# correct_cnt = [1,254,219,29,8,4,274,293,66,99,269,131,179,250,237,532,747,670,732,841,78,1,919,]

# no_answers = [1214,13,116,1054,1156,1208,1,7,866,759,15,600,392,1,5,290,15,4,10,12,803,1208,80,]

# wrong_cnt = [6,954,886,138,57,9,946,921,289,363,937,490,650,970,979,399,459,547,479,368,340,12,222,]

# time = [773.57,2520.44,2671.66,2029.69,1988.01,1344.36,3201.6,2949.32,2095.46,2486.64,2038.50,3019.21,3051.17,2825.40,
#         2818.53,2639.30,2709.49,2051.36,2056.59,5755.59,4063.31,3908.77,9076.33]

import matplotlib.pyplot as plt

colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9','#800000']

markers = ["o","v","s","P","*","D"]

hatches = ['/', '|', '-', '+', 'x', 'o']

model_parameters = [
    124000000,  # gpt2 "o" '#0173b2'
    135000000,  # SmolLM2-135M "v" '#0173b2'
    135000000,  # SmolLM2-135M-Instruct "v" '#de8f05'
    270000000,  # OpenELM-270M "s" '#0173b2'
    270000000,  # OpenELM-270M-Instruct "s" '#de8f05'
    355000000,  # gpt2-medium "o" '#de8f05'
    360000000,  # SmolLM2-360M "v" '#029e73'
    360000000,  # SmolLM2-360M-Instruct "v" '#d55e00'
    450000000,  # OpenELM-450M "s" '#029e73'
    450000000,  # OpenELM-450M-Instruct "s" '#d55e00'
    774000000,  # gpt2-large "o" '#029e73'
    1100000000,  # OpenELM-1_1B "s" '#cc78bc'
    1100000000,  # OpenELM-1_1B-Instruct "s" '#ca9161'
    1100000000,  # TinyLlama_v1_1 (assumed smaller size) "P" '#0173b2'
    1500000000,  # gpt2-xl "o" '#d55e00'
    1600000000,  # stablelm-2-1_6b "*" '#0173b2'
    1600000000,  # stablelm-2-zephyr-1_6b "*" '#de8f05'
    1700000000,  # SmolLM2-1.7B "v" '#cc78bc'
    1700000000,  # SmolLM2-1.7B-Instruct "v" '#ca9161'
    2000000000,  # gemma-2-2b-it "D" '#0173b2'
    3000000000,  # OpenELM-3B "s" '#fbafe4'
    3000000000,  # OpenELM-3B-Instruct "s" '#949494'
    9000000000,  # gemma-2-9b-it "D" '#de8f05'
]

# Define custom X-axis labels and ticks to better align with the model parameter magnitudes
ticks = [124000000, 135000000, 270000000, 355000000, 360000000, 450000000, 774000000, 1100000000, 1500000000, 1600000000, 1700000000, 2000000000, 3000000000, 9000000000]  # Custom ticks to align with the models' parameters
labels = ["124M", "135M", "270M", "355M", "360M", "450M", "774M", "1.1B", "1.5B", "1.6B", "1.7B", "2B", "3B", "9B"]

developer_names = ["gpt2","SmolLM2","OpenELM","TinyLlama","stablelm","gemma-2"]
hatch_list = []
color_list = []
models_info = {}
gpt_cnt = 0
SmolLM2_cnt = 0
OpenELM_cnt = 0
TinyLlama_cnt = 0
stablelm_cnt = 0
gemma_2_cnt = 0
for index, name in enumerate(model_names):
    if "gpt2" in name:
        color = colors[gpt_cnt]
        marker = markers[0]
        hatch_list.append(hatches[0])
        color_list.append(colors[0])
        gpt_cnt += 1
    elif "SmolLM2" in name:
        color = colors[SmolLM2_cnt]
        marker = markers[1]
        hatch_list.append(hatches[1])
        color_list.append(colors[1])
        SmolLM2_cnt += 1
    elif "OpenELM" in name:
        color = colors[OpenELM_cnt]
        marker = markers[2]
        hatch_list.append(hatches[2])
        color_list.append(colors[2])
        OpenELM_cnt += 1
    elif "TinyLlama" in name:
        color = colors[TinyLlama_cnt]
        marker = markers[3]
        hatch_list.append(hatches[3])
        color_list.append(colors[3])
        TinyLlama_cnt += 1
    elif "stablelm" in name:
        color = colors[stablelm_cnt]
        marker = markers[4]
        hatch_list.append(hatches[4])
        color_list.append(colors[4])
        stablelm_cnt += 1
    elif "gemma-2" in name:
        color = colors[gemma_2_cnt]
        marker = markers[5]
        hatch_list.append(hatches[5])
        color_list.append(colors[5])
        gemma_2_cnt += 1
    else:
        raise Exception("Name not recognized")
    # color_list.append(color)
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
# ax.set_title("Comparison of Validation Accuracy for Various LMs\n on the PrOntoQA-OOD Dataset with " + rules, fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='best')
plt.tight_layout()

#%% Bar plots
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