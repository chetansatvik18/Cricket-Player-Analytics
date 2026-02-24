import pandas as pd
import os

file_path = os.path.join("Dataset","IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

# match wise stats
match_stats = data.groupby(
    ["match_id","batter"]
).agg(
    runs=("runs_batter","sum"),
    balls=("ball","count")
).reset_index()

match_stats["strike_rate"] = (
    match_stats["runs"]/match_stats["balls"]
)*100

# sort matches newest first
match_stats = match_stats.sort_values("match_id", ascending=False)

# last 10 matches per player
recent = match_stats.groupby("batter").head(10)

# calculate recent metrics
recent_form = recent.groupby("batter").agg(
    avg_runs=("runs","mean"),
    avg_sr=("strike_rate","mean"),
    consistency=("runs","std")
)

print("\n🏏 RECENT PLAYER FORM\n")
print(recent_form.sort_values("avg_runs",ascending=False).head(10))