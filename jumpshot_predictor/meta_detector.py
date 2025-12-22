# meta_detector.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt

# load in data
shots = pd.read_csv("shots.csv")
shots["make_pct"] = shots["makes"] / shots["attempts"]

# calculate green % from 2kLabs
shots["green_pct"] = shots["make_pct"] * (
    0.65 + (shots["player_3pt"] - 85) * 0.003  # 2KLab green window rate
)
shots["greens"] = (shots["green_pct"] * shots["attempts"]).round().astype(int)
print(f"Est. greens calculated (92 3PT → ~75% of makes)")

print(f"Loaded {len(shots)} shot sessions")

# train + baseline
cat_cols = ["base", "release1", "release2", "speed", "patch"]
num_cols = ["player_3pt"]
X = shots[cat_cols + num_cols]
y = shots["make_pct"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("xgb", XGBRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)
joblib.dump(model, "jumpshot_model.pkl")
print("Baseline model trained + saved (R2:", r2_score(y, model.predict(X)), ")")

# compute residuals
shots["pred_make"] = model.predict(X)
shots["residual"] = shots["make_pct"] - shots["pred_make"]

exploit_cols = ["base", "release1", "release2", "speed"]
exploits = shots.groupby(exploit_cols).agg(
    total_attempts=("attempts", "sum"),
    avg_residual=("residual", "mean"),
    n_sessions=("date", "count"),
    total_greens=("greens", "sum")
).reset_index()

exploits = exploits[exploits["total_attempts"] >= 50]  # reliable volume
exploits["advantage_pct"] = exploits["avg_residual"] * 100

print("\nEXPLOIT LEADERBOARD (top 5)")
print(exploits.nlargest(5, "avg_residual")[["base", "release1", "release2", "speed", "avg_residual", "total_attempts", "total_greens"]])

exploits.to_csv("exploit_leaderboard.csv", index=False)

# patch detection 
print("\nPATCH IMPACT (biggest drops)")
patch_changes = shots.groupby(["patch"] + exploit_cols).agg(
    mean_make_pct=("make_pct", "mean"),
    total_attempts=("attempts", "sum"),
    avg_residual=("residual", "mean"),
    total_greens=("greens", "sum")
).reset_index()
patch_changes.to_csv("patch_impact.csv", index=False)

print("Files saved: exploit_leaderboard.csv, patch_impact.csv, jumpshot_model.pkl")

# VISUALIZATION
top_exploits = exploits.nlargest(10, "avg_residual")
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_exploits)), top_exploits["avg_residual"] * 100)
plt.yticks(range(len(top_exploits)), 
           [f"{r['base']}/{r['release1']}/{r['release2']}" for _, r in top_exploits.iterrows()])
plt.xlabel("Advantage %")
plt.title("Top 10 Meta Jumpshots (Real Data)")
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("exploit_chart.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: exploit_chart.png")
