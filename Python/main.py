import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

print("\n🏏 CRICKET PLAYER ANALYTICS SYSTEM STARTED\n")

# ---------------- LOAD DATA ----------------
file_path = os.path.join("Dataset", "IPL.csv")
data = pd.read_csv(file_path, low_memory=False)

print("✅ Dataset Loaded Successfully")

# ---------------- PREPROCESS ----------------
batting_data = data[data["runs_batter"].notna()]

player_stats = batting_data.groupby("batter").agg(
    total_runs=("runs_batter", "sum"),
    balls_faced=("ball", "count")
)

player_stats["strike_rate"] = (
    player_stats["total_runs"] / player_stats["balls_faced"]
) * 100

player_stats = player_stats[player_stats["balls_faced"] > 50]

# =====================================================
# 1️⃣ OLAP STYLE ANALYSIS (TOP PLAYERS)
# =====================================================
print("\n🏆 TOP 10 PLAYERS BY RUNS\n")
print(player_stats.sort_values(by="total_runs", ascending=False).head(10))

# =====================================================
# 2️⃣ FORM CLASSIFICATION (Decision Tree)
# =====================================================
high_runs = player_stats["total_runs"].quantile(0.75)
low_runs = player_stats["total_runs"].quantile(0.25)
avg_sr = player_stats["strike_rate"].mean()

def assign_form(row):
    if row["total_runs"] >= high_runs and row["strike_rate"] >= avg_sr:
        return "GOOD"
    elif row["total_runs"] <= low_runs:
        return "POOR"
    else:
        return "AVERAGE"

player_stats["form"] = player_stats.apply(assign_form, axis=1)

X_class = player_stats[["total_runs", "strike_rate"]]
y_class = player_stats["form"]

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_class, y_class)

player_stats["predicted_form"] = dt_model.predict(X_class)

print("\n📊 PLAYER FORM PREDICTION\n")
print(player_stats[["total_runs","strike_rate","predicted_form"]]
      .sort_values(by="total_runs", ascending=False)
      .head(10))

# =====================================================
# 3️⃣ REGRESSION (EXPECTED RUNS)
# =====================================================
X_reg = player_stats[["strike_rate", "balls_faced"]]
y_reg = player_stats["total_runs"]

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

player_stats["expected_runs"] = reg_model.predict(X_reg)

print("\n📈 EXPECTED RUNS PREDICTION\n")
print(player_stats[["strike_rate","expected_runs"]]
      .sort_values(by="expected_runs", ascending=False)
      .head(10))

# =====================================================
# 4️⃣ CLUSTERING (K-MEANS)
# =====================================================
kmeans = KMeans(n_clusters=3, random_state=42)
player_stats["cluster"] = kmeans.fit_predict(
    player_stats[["total_runs","strike_rate"]]
)

print("\n🔵 PLAYER CLUSTERS\n")
print(player_stats[["total_runs","strike_rate","cluster"]]
      .sort_values(by="total_runs", ascending=False)
      .head(10))

print("\n✅ SYSTEM EXECUTED SUCCESSFULLY")