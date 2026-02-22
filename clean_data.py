import pandas as pd 
from transformers import AutoTokenizer 
import matplotlib.pyplot as plt
import os
import re

file= "metadata/arxiv_metadata.csv"
print("loaded file")

df = pd.read_csv(file,index_col="entry_id",on_bad_lines='skip')
print(f"df shape: {df.shape}")

df.drop(columns=['Unnamed: 0'],inplace=True)

# Use raw strings (r'') to avoid double-escaping backslashes

import re

def clean_latex(text: str) -> str:
    if not isinstance(text, str):
        return text

    # 1. Remove common formatting commands
    text = re.sub(r'\\(textbf|textit|mathrm|mathcal)\{([^}]+)\}', r'\2', text)

    # 2. Generic fallback for unknown commands
    text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)

    # 3. Fix escaped percent
    text = re.sub(r'\\%', '%', text)

    # 4. Normalize inline math spacing
    text = re.sub(r'\$\s*([^$]+?)\s*\$', r'$\1$', text)

    # 5. Replace common math symbols
    replacements = {
        r'\\varepsilon': 'ε',
        r'\\times': '×'
    }
    for k, v in replacements.items():
        text = re.sub(k, v, text)

    # 6. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["summary"] = df["summary"].apply(clean_latex)

print(sum(df['summary'].str.contains(r'\\|\{', regex=True)))


df.to_csv(file)
print("file saved")





