model_names = ["gpt2","SmolLM2-135M","SmolLM2-135M-Instruct","OpenELM-270M","OpenELM-270M-Instruct","gpt2-medium","SmolLM2-360M","SmolLM2-360M-Instruct","OpenELM-450M","OpenELM-450M-Instruct","gpt2-large","OpenELM-1_1B","OpenELM-1_1B-Instruct","TinyLlama_v1_1","gpt2-xl","stablelm-2-1_6b","stablelm-2-zephyr-1_6b","SmolLM2-1.7B","SmolLM2-1.7B-Instruct","gemma-2-2b-it","OpenELM-3B","OpenELM-3B-Instruct","gemma-2-9b-it"]

# Total 1221 questions

correct_cnt = [46,254,219,239,129,231,274,293,248,185,269,252,254,250,237,532,747,670,732,843,297,279,923]

validation_accuracy = [3.77,20.8,17.94,19.57,10.57,18.92,22.44,24,20.31,15.15,22.03,20.64,20.80,20.48,19.41,43.57,61.18,54.87,59.95,69.04,24.32,22.85,75.59]

no_answers = [1037,13,116,18,619,66,1,7,7,365,15,1,43,1,5,290,15,4,10,13,2,21,77]

wrong_cnt = [138,954,886,964,473,924,946,921,966,671,937,968,924,970,979,399,459,547,479,365,922,921,221]

time = [713.80,2556.08,2962.50,1985.76,2030.15,1542.13,3129.68,3219.39,2440.42,2526.58,2268.81,2865.06,3009.85,3329.80,2590.65,3116.96,2730.70,2013.32,2317.82,5291.48,3873.72,3813.59,8617.48]

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
    models_info[name] = {"color": color, "marker": marker, "num_parm": model_parameters[index], "accuracy": validation_accuracy[index], "correct_cnt": correct_cnt[index], "no_answer_cnt": no_answers[index], "wrong_cnt": wrong_cnt[index], "time": time[index]}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# Scatter Plot for Validation Accuracy vs Number of Model Parameters
for i, name in enumerate(model_names):
    ax.scatter(models_info[name]["num_parm"], models_info[name]["accuracy"], marker=models_info[name]["marker"], color=models_info[name]["color"], s=100, label=name, zorder=3)
ax.set_xlabel("Number of Model Parameters", fontsize=16)
ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
ax.set_title("Comparison of Validation Accuracy for\n Various LMs on the CSQA Dataset", fontsize=20)
ax.set_xticks(ticks, labels)
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(ncol=2, fontsize=11, loc='upper left')
plt.tight_layout()

#%% Bar plots
# Create legend handles manually
import matplotlib.patches as mpatches
handles = []
for i, name in enumerate(developer_names):
    patch = mpatches.Patch(color=colors[i], label=name)
    handles.append(patch)

# function to add value labels
def addlabels(x,y,r):
    for i in range(len(x)):
        plt.text(i, y[i]+10, y[i], ha = 'center', weight='bold', fontsize=12, rotation=r)

# Create a figure and a set of subplots
# Plotting the bar plot
plt.figure(figsize=(12, 6))
# plt.bar(model_names, correct_cnt, color = 'w', edgecolor='k', hatch = hatch_list)
plt.bar(model_names, correct_cnt, color = color_list)
addlabels(model_names, correct_cnt, 0)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Model Names", fontsize=16)
plt.ylabel("Number of Correct Answers", fontsize=16)
plt.title("Number of Correct Answers for Each Model", fontsize=20)
plt.legend(handles=handles, fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()

# Create a figure and a set of subplots
# Plotting the bar plot
plt.figure(figsize=(12, 6))
plt.bar(model_names, no_answers, color = color_list)
addlabels(model_names, no_answers, 0)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Model Names", fontsize=16)
plt.ylabel("Number of No Answers", fontsize=16)
plt.title("Number of No Answers for Each Model", fontsize=20)
plt.legend(handles=handles, fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()

# Create a figure and a set of subplots
# Plotting the bar plot
plt.figure(figsize=(12, 6))
plt.bar(model_names, wrong_cnt, color = color_list)
addlabels(model_names, wrong_cnt, 0)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Model Names", fontsize=16)
plt.ylabel("Number of Incorrect Answers", fontsize=16)
plt.title("Number of Incorrect Answers for Each Model", fontsize=20)
plt.legend(handles=handles, fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()

# Create a figure and a set of subplots
# Plotting the bar plot
plt.figure(figsize=(12, 6))
plt.bar(model_names, time, color = color_list)
addlabels(model_names, time, 40)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Model Names", fontsize=16)
plt.ylabel("Computational Time (sec)", fontsize=16)
plt.title("Computational Time for Each Model", fontsize=20)
plt.legend(handles=handles, fontsize=12)
plt.ylim(0, 11000)
plt.grid()
plt.tight_layout()
plt.show()
