# meta_detector.py 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
import joblib
import matplotlib.pyplot as plt

# =========================
# load data
# =========================
shots = pd.read_csv("shots.csv")
shots["make_pct"] = shots["makes"] / shots["attempts"]

# estimated greens (UNCHANGED)
shots["green_pct"] = shots["make_pct"] * (
    0.65 + (shots["player_3pt"] - 85) * 0.003
)
shots["greens"] = (shots["green_pct"] * shots["attempts"]).round().astype(int)

print(f"Est. greens calculated (92 3PT → ~75% of makes)")
print(f"Loaded {len(shots)} shot sessions")

# =========================
# 2k features
# =========================
cat_cols = ["base", "release1", "release2", "speed", "patch"]
num_cols = ["player_3pt"]

X = shots[cat_cols + num_cols]
y = shots["make_pct"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols),
])

# =========================
# regularized model
# =========================
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)

model = Pipeline([
    ("preprocess", preprocess),
    ("xgb", xgb)
])

# =========================
# train data
# =========================
gss = GroupShuffleSplit(test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=shots["date"]))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)

# =========================
# model evaluation
# =========================
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

print(f"Baseline model trained + saved")
print(f"Train R2: {train_r2:.4f}")
print(f"Test  R2: {test_r2:.4f}")

joblib.dump(model, "jumpshot_model.pkl")

# =========================
#  residuals
# =========================
shots["pred_make"] = model.predict(X)
shots["residual"] = shots["make_pct"] - shots["pred_make"]

# =========================
# exploit detection
# =========================
exploit_cols = ["base", "release1", "release2", "speed"]

exploits = shots.groupby(exploit_cols).agg(
    total_attempts=("attempts", "sum"),
    avg_residual=("residual", "mean"),
    residual_std=("residual", "std"),
    n_sessions=("date", "count"),
    total_greens=("greens", "sum")
).reset_index()

# reliability filter 
exploits = exploits[exploits["total_attempts"] >= 50]

# exploit score (volume + consistency aware)
exploits["exploit_score"] = (
    exploits["avg_residual"]
    * np.log1p(exploits["total_attempts"])
    * np.sqrt(exploits["n_sessions"])
)

print("\nEXPLOIT LEADERBOARD (top 5)")
print(
    exploits.nlargest(5, "exploit_score")[
        ["base", "release1", "release2", "speed",
         "avg_residual", "exploit_score",
         "total_attempts", "total_greens"]
    ]
)

exploits.to_csv("exploit_leaderboard.csv", index=False)

# =========================
#  patch change detection
# =========================
patch_perf = shots.groupby(
    ["patch"] + exploit_cols
).agg(
    mean_make_pct=("make_pct", "mean"),
    total_attempts=("attempts", "sum")
).reset_index()

patch_perf = patch_perf.sort_values("patch")

# rolling stats per jumpshot
patch_perf["rolling_mean"] = patch_perf.groupby(exploit_cols)["mean_make_pct"] \
    .transform(lambda x: x.rolling(2).mean())

patch_perf["rolling_std"] = patch_perf.groupby(exploit_cols)["mean_make_pct"] \
    .transform(lambda x: x.rolling(2).std())

patch_perf["z_score"] = (
    patch_perf["mean_make_pct"] - patch_perf["rolling_mean"]
) / patch_perf["rolling_std"]

# significant stealth changes
patch_alerts = patch_perf[
    patch_perf["z_score"].abs() > 2
]

patch_perf.to_csv("patch_impact.csv", index=False)
patch_alerts.to_csv("patch_alerts.csv", index=False)

print("\nPATCH IMPACT (significant stealth changes)")
print(patch_alerts[[
    "patch", "base", "release1", "release2",
    "mean_make_pct", "z_score"
]])

print("Files saved: exploit_leaderboard.csv, patch_impact.csv, patch_alerts.csv, jumpshot_model.pkl")

# =========================
#  visualization
# =========================
top_exploits = exploits.nlargest(10, "exploit_score")

plt.figure(figsize=(12, 6))
plt.barh(range(len(top_exploits)), top_exploits["exploit_score"])
plt.yticks(
    range(len(top_exploits)),
    [f"{r['base']}/{r['release1']}/{r['release2']}"
     for _, r in top_exploits.iterrows()]
)
plt.xlabel("Exploit Score")
plt.title("Top 10 Meta Jumpshots (Validated Model)")
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("exploit_chart.png", dpi=300, bbox_inches='tight')
plt.show()

print("Saved: exploit_chart.png")

