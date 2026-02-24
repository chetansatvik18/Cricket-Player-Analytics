import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\n🏏 CRICKET PLAYER ANALYSIS SYSTEM\n")

# ---------------- LOAD DATA ----------------
file_path = os.path.join("Dataset", "IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

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

# ---------------- LAST 10 MATCHES ----------------
recent = match_stats.groupby("batter").head(10)

player_stats = recent.groupby("batter").agg(
    avg_runs=("runs","mean"),
    avg_sr=("strike_rate","mean"),
    consistency=("runs","std")
).fillna(0)

# =====================================================
# FORM CLASSIFICATION
# =====================================================
def assign_form(row):
    if row["avg_runs"] >= 40 and row["avg_sr"] >= 130:
        return "GOOD"
    elif row["avg_runs"] >= 25:
        return "AVERAGE"
    else:
        return "POOR"

player_stats["form"] = player_stats.apply(assign_form, axis=1)

X_class = player_stats[["avg_runs","avg_sr"]]
y_class = player_stats["form"]

dt = DecisionTreeClassifier()
dt.fit(X_class, y_class)

# =====================================================
# NEXT MATCH RUN PREDICTION (WEIGHTED TREND)
# =====================================================
def predict_next_runs(last_scores):

    if len(last_scores) == 0:
        return 0

    if len(last_scores) < 3:
        return sum(last_scores) / len(last_scores)

    # recent matches weighted more
    weights = list(range(1, len(last_scores)+1))
    weighted_runs = sum(r*w for r, w in zip(last_scores, weights))

    return weighted_runs / sum(weights)

# =====================================================
# CLUSTERING (RECENT PERFORMANCE)
# =====================================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    player_stats[["avg_runs","avg_sr","consistency"]]
)

kmeans = KMeans(n_clusters=3, random_state=42)
player_stats["cluster"] = kmeans.fit_predict(X_scaled)

# =====================================================
# PLAYER SEARCH
# =====================================================
player_name = input("\nEnter Player Name: ")

if player_name in player_stats.index:

    print("\n✅ PLAYER ANALYSIS RESULT\n")

    # last 10 match runs
    last_matches = recent[recent["batter"] == player_name]["runs"]
    last_scores = list(last_matches)

    print("Last 10 Match Scores:")
    print(last_scores)

    avg_runs = player_stats.loc[player_name,"avg_runs"]
    avg_sr = player_stats.loc[player_name,"avg_sr"]
    form = player_stats.loc[player_name,"form"]
    cluster = player_stats.loc[player_name,"cluster"]

    predicted_runs = predict_next_runs(last_scores)

    print("\nAverage Runs:", round(avg_runs,2))
    print("Average Strike Rate:", round(avg_sr,2))
    print("Form:", form)
    print("Cluster:", cluster)
    print("Expected Runs Next Match:", round(predicted_runs,2))

else:
    print("❌ Player not found in dataset.")