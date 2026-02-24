import pandas as pd
import os

print("Current folder:", os.getcwd())

# Correct file path
file_path = os.path.join("Dataset", "IPL.csv")

# Load dataset
data = pd.read_csv(file_path, low_memory=False)

print("✅ Dataset Loaded")

# Calculate total runs per player
player_runs = data.groupby("batter")["runs_batter"].sum()

# Sort highest runs first
player_runs = player_runs.sort_values(ascending=False)

print("\n🏏 TOP 10 RUN SCORERS IN DATASET:\n")
print(player_runs.head(10))