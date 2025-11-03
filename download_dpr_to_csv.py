# download_dpr_to_csv.py
# Joshua Ogunseinde — saves the Definite Pronoun Resolution dataset locally
from datasets import load_dataset
import pandas as pd

print("📥 Downloading Definite Pronoun Resolution dataset...")
ds = load_dataset("community-datasets/definite_pronoun_resolution")

# Convert both splits to pandas DataFrames
train_df = pd.DataFrame(ds["train"])
test_df = pd.DataFrame(ds["test"])

# Save locally as CSV files
train_df.to_csv("dpr_train.csv", index=False, encoding="utf-8")
test_df.to_csv("dpr_test.csv", index=False, encoding="utf-8")

print("\n✅ Saved the dataset locally:")
print(" - dpr_train.csv")
print(" - dpr_test.csv")

# Optional: preview a few examples
print("\n🔹 Sample rows from training data:")
print(train_df.sample(5))

# ALL LIBRARIES USED
# pip install spacy==3.6.1
# pip install coreferee==1.4.1
# pip install pandas
# pip install numpy
# python -m spacy download en_core_web_sm
# python -m coreferee install en

# Create new virtual environment
# python -m venv pronoun_env

# Activate it (U can do this all in visual studio, thats what i did)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# pronoun_env\Scripts\activate
# U can also try this if this doesnt work
# D:\NLProject\Natural-Lang-Project\pronoun_env\Scripts\python.exe main.py
# 



