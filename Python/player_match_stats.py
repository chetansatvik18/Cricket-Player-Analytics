import pandas as pd
import os

print("Creating match-wise player statistics...")

file_path = os.path.join("Dataset","IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

# total runs per player per match
match_stats = data.groupby(
    ["match_id","batter"]
).agg(
    runs=("runs_batter","sum"),
    balls=("ball","count")
).reset_index()

# strike rate per match
match_stats["strike_rate"] = (
    match_stats["runs"] / match_stats["balls"]
) * 100

print("\n✅ Match-wise stats created")
print(match_stats.head())