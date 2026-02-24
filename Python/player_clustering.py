import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Starting Player Clustering (Recent Form Based)...")

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
    avg_sr=("strike_rate","mean"),
    consistency=("runs","std")
).fillna(0)

# ---------------- FEATURE SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    player_stats[["avg_runs","avg_sr","consistency"]]
)

# ---------------- KMEANS ----------------
kmeans = KMeans(n_clusters=3, random_state=42)
player_stats["cluster"] = kmeans.fit_predict(X_scaled)

# ---------------- OUTPUT ----------------
print("\n🏏 PLAYER CLUSTERS (RECENT PERFORMANCE):\n")
print(
    player_stats[["avg_runs","avg_sr","consistency","cluster"]]
    .sort_values(by="avg_runs", ascending=False)
    .head(15)
)