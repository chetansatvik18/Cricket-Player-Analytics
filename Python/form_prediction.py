import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

print("Starting Player Form Prediction...")

# ---------------- LOAD DATA ----------------
file_path = os.path.join("Dataset", "IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

print("✅ Dataset Loaded")

# ---------------- PREPROCESSING ----------------
batting_data = data[data["runs_batter"].notna()]

# ---------------- MATCH-WISE PERFORMANCE ----------------
match_stats = data.groupby(
    ["match_id","batter"]
).agg(
    runs=("runs_batter","sum"),
    balls=("ball","count")
).reset_index()

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
    avg_sr=("strike_rate","mean"),
    consistency=("runs","std")
).fillna(0)

# Strike Rate
player_stats["strike_rate"] = (
    player_stats["total_runs"] / player_stats["balls_faced"]
) * 100

# Remove very small players
player_stats = player_stats[player_stats["balls_faced"] > 50]

# ---------------- AUTOMATIC CUTOFFS ----------------
high_runs = player_stats["total_runs"].quantile(0.75)
low_runs = player_stats["total_runs"].quantile(0.25)
avg_sr = player_stats["strike_rate"].mean()

print("\n📊 Automatic Thresholds:")
print("High Runs Cutoff:", round(high_runs,2))
print("Low Runs Cutoff :", round(low_runs,2))
print("Average Strike Rate:", round(avg_sr,2))

# ---------------- FORM CLASSIFICATION ----------------
def assign_form(row):
    if row["avg_runs"] >= 40 and row["avg_sr"] >= 130:
        return "GOOD"
    elif row["avg_runs"] >= 25:
        return "AVERAGE"
    else:
        return "POOR"

player_stats["form"] = player_stats.apply(assign_form, axis=1)

# ---------------- MACHINE LEARNING ----------------
X = player_stats[["avg_runs","avg_sr"]]
y = player_stats["form"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Predict again (demo output)
player_stats["predicted_form"] = model.predict(X)

# ---------------- OUTPUT ----------------
print("\n🏏 PLAYER FORM PREDICTION RESULT:\n")
print(player_stats[["total_runs","strike_rate","predicted_form"]]
      .sort_values(by="total_runs", ascending=False)
      .head(10))