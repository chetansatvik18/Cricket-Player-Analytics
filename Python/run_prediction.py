import pandas as pd
import os
from sklearn.linear_model import LinearRegression

print("Starting Run Prediction Model...")

# ---------------- LOAD DATA ----------------
file_path = os.path.join("Dataset", "IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

print("✅ Dataset Loaded")

# ---------------- MATCH-WISE PERFORMANCE ----------------
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

# newest matches first
match_stats = match_stats.sort_values("match_id", ascending=False)

# last 10 matches per player
recent = match_stats.groupby("batter").head(10)

# recent performance features
player_stats = recent.groupby("batter").agg(
    avg_runs=("runs","mean"),
    avg_sr=("strike_rate","mean")
).fillna(0)

# ---------------- REGRESSION MODEL ----------------
X = player_stats[["avg_sr"]]
y = player_stats["avg_runs"]

model = LinearRegression()
model.fit(X, y)

# predict next match runs
player_stats["expected_runs_next_match"] = model.predict(X)

# ---------------- OUTPUT ----------------
print("\n🏏 EXPECTED RUNS (NEXT MATCH PREDICTION):\n")
print(
    player_stats[["avg_runs","avg_sr","expected_runs_next_match"]]
    .sort_values(by="expected_runs_next_match", ascending=False)
    .head(10)
)