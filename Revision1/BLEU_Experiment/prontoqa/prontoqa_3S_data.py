import re
import pandas as pd
from pathlib import Path

# Load the file
file_path = Path("R-run_experiment_dummy_3S_AM.sh_16067257.out")
text = file_path.read_text()

# Define the regex pattern to capture input and label
pattern = re.compile(
    r"(Q:.*?Prove:.*?A:.*?)(?:Expected answer: )(.*?)(?=\nQ:|\Z)", re.DOTALL
)

# Extract matches and store them in a list of dicts
data = []
for match in pattern.finditer(text):
    input_text = match.group(1).strip()
    input_text = input_text.split('\n\nPredicted answer:')[0]
    label_text = match.group(2).strip()
    # Split the text by nextline symbols
    label_text = " " + label_text.split('\nn')[0]
    data.append({"input": input_text, "label": label_text})

# Convert to DataFrame
df = pd.DataFrame(data)

# (Optional) Save to CSV
df.to_csv("logic_proving_dataset.csv", index=False)

# Print preview
print(df.head())
