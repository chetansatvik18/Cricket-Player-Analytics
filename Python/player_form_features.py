import pandas as pd
import os

# Load dataset
file_path = os.path.join("Dataset", "IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

print("✅ Dataset Loaded")

# Keep only valid batting records
batting_data = data[data["runs_batter"].notna()]

# Group by player
player_stats = batting_data.groupby("batter").agg(
    total_runs=("runs_batter", "sum"),
    balls_faced=("ball", "count")
)

# Calculate strike rate
player_stats["strike_rate"] = (
    player_stats["total_runs"] / player_stats["balls_faced"]
) * 100

# Remove players with very few balls (noise)
player_stats = player_stats[player_stats["balls_faced"] > 50]

# Sort by runs
player_stats = player_stats.sort_values(by="total_runs", ascending=False)

print("\n🏏 PLAYER PERFORMANCE FEATURES:\n")
print(player_stats.head(10))