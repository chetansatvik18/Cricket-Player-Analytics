import pandas as pd
import os

print("Current working folder:")
print(os.getcwd())

# Correct path
data = pd.read_csv(file_path, low_memory=False)

data = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully")
print("\nColumns:\n", data.columns)
print("\nFirst 5 rows:\n", data.head())