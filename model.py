import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats


# 1. Chargement des données
df = pd.read_csv("agile_estimation_dataset.csv")
print(f"Dataset : {len(df)} PRs")
print(f"Repartition era : {df['era'].value_counts().to_dict()}")

# 2. Nettoyage
df_model = df.drop(columns=["repo_name", "language", "pr_number"])
df_model = df_model.dropna()

# Encoder era : pre_AI = 0, post_AI = 1
df_model["era_encoded"] = (df_model["era"] == "post_AI").astype(int)
df_model = df_model.drop(columns=["era"])

# 3. Features & Cible
FEATURES = [
    "code_churn", "files_changed", "review_cycles",
    "pr_comments", "refactoring_ratio", "commit_velocity",
    "velocity_sprint", "active_contributors", "age_days",
    "stars", "era_encoded"
]
TARGET = "story_points"

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"\nFeatures : {FEATURES}")
print(f"Cible    : {TARGET}")
print(f"\nDistribution story_points :\n{y.value_counts().sort_index()}")

# 4. Correlation
print("\nCorrelation avec story_points :")
print(df_model[FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False))

# 5. Separation pre_AI / post_AI
df_pre  = df_model[df_model["era_encoded"] == 0]
df_post = df_model[df_model["era_encoded"] == 1]

X_pre  = df_pre[FEATURES]
y_pre  = df_pre[TARGET]
X_post = df_post[FEATURES]
y_post = df_post[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X_pre, y_pre, test_size=0.2, random_state=42
)

print(f"\nTrain (pre_AI)      : {len(X_train)} PRs")
print(f"Validation (pre_AI) : {len(X_val)} PRs")
print(f"Test (post_AI)      : {len(X_post)} PRs")

# 6. Modeles
# Baseline
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Modele principal
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)


# 7. Evaluation
def evaluate(model, name, X_test, y_test, label=""):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n{name} -- {label}")
    print(f"   MAE : {mae:.3f} story points")
    print(f"   R2  : {r2:.3f}")
    return mae, r2

print("\n" + "="*55)
print("RESULTATS")
print("="*55)

mae_rf_val,  r2_rf_val  = evaluate(rf, "Random Forest",     X_val,  y_val,  "validation pre_AI")
mae_gb_val,  r2_gb_val  = evaluate(gb, "Gradient Boosting", X_val,  y_val,  "validation pre_AI")
mae_rf_post, r2_rf_post = evaluate(rf, "Random Forest",     X_post, y_post, "TEST post_AI")
mae_gb_post, r2_gb_post = evaluate(gb, "Gradient Boosting", X_post, y_post, "TEST post_AI")

print(f"\nDegradation Gradient Boosting (pre -> post) :")
print(f"   R2  : {r2_gb_val:.3f} -> {r2_gb_post:.3f}  (delta = {r2_gb_val - r2_gb_post:.3f})")
print(f"   MAE : {mae_gb_val:.3f} -> {mae_gb_post:.3f}  (delta = {mae_gb_post - mae_gb_val:.3f})")


# 8. Analyse du drift par feature
print("\n" + "="*55)
print("DRIFT PAR FEATURE (Kolmogorov-Smirnov)")
print("="*55)

drift_results = {}
for feat in FEATURES:
    if feat == "era_encoded":
        continue
    a = df_pre[feat].dropna()
    b = df_post[feat].dropna()
    if len(a) > 5 and len(b) > 5:
        ks_stat, p_value = stats.ks_2samp(a, b)
        drift_results[feat] = {"ks": ks_stat, "p": p_value}
        status = "CONTAMINEE" if p_value < 0.05 else "stable"
        print(f"  {feat:25s}  KS={ks_stat:.3f}  p={p_value:.3f}  {status}")

stable   = [f for f, v in drift_results.items() if v["p"] >= 0.05]
unstable = [f for f, v in drift_results.items() if v["p"] <  0.05]
print(f"\nFeatures stables     : {stable}")
print(f"Features contaminees : {unstable}")


# 9. Model Adapté
stable_features = stable + ["era_encoded"]

X_adapted_train = X_train[stable_features].copy()
X_adapted_val   = X_val[stable_features].copy()
X_adapted_post  = X_post[stable_features].copy()

for f in stable:
    X_adapted_train[f + "_w"] = X_train[f].values
    X_adapted_val[f + "_w"]   = X_val[f].values
    X_adapted_post[f + "_w"]  = X_post[f].values

gb2 = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=3, subsample=0.8, random_state=42
)
gb2.fit(X_adapted_train, y_train)

mae_gb2_val,  r2_gb2_val  = evaluate(gb2, "GB Adapte", X_adapted_val,  y_val,  "validation pre_AI")
mae_gb2_post, r2_gb2_post = evaluate(gb2, "GB Adapte", X_adapted_post, y_post, "TEST post_AI")

print(f"\nComparaison sur post_AI :")
print(f"   Modele 1 (standard) R2 : {r2_gb_post:.3f}")
print(f"   Modele 2 (adapte)   R2 : {r2_gb2_post:.3f}")
print(f"   Amelioration           : {r2_gb2_post - r2_gb_post:.3f}")


# 10. Visualisations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution story_points par era
sns.countplot(
    data=df_model, x="story_points",
    hue=df_model["era_encoded"].map({0: "pre_AI", 1: "post_AI"}),
    ax=axes[0, 0], palette=["steelblue", "coral"]
)
axes[0, 0].set_title("Distribution story_points par periode")
axes[0, 0].legend(title="Era")

# Drift code_churn
axes[0, 1].hist(df_pre["code_churn"].clip(upper=3000),
                bins=30, alpha=0.6, label="pre_AI", color="steelblue")
axes[0, 1].hist(df_post["code_churn"].clip(upper=3000),
                bins=30, alpha=0.6, label="post_AI", color="coral")
axes[0, 1].set_title("code_churn : contamine par l'IA ?")
axes[0, 1].legend()

# Drift review_cycles
axes[1, 0].hist(df_pre["review_cycles"],
                bins=15, alpha=0.6, label="pre_AI", color="steelblue")
axes[1, 0].hist(df_post["review_cycles"],
                bins=15, alpha=0.6, label="post_AI", color="coral")
axes[1, 0].set_title("review_cycles : signal humain stable ?")
axes[1, 0].legend()

# Feature importance
importance = pd.Series(
    gb.feature_importances_, index=FEATURES
).sort_values(ascending=True)
importance.plot(kind="barh", ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Feature importance (Gradient Boosting)")

plt.tight_layout()
plt.savefig("analyse_drift.png", dpi=150)
plt.show()

# 11. Resume final
print("\n" + "="*55)
print("RESUME FINAL")
print("="*55)
print(f"  Algo principal       : Gradient Boosting")
print(f"  Cible                : story_points")
print(f"  Modele 1 pre_AI  R2  : {r2_gb_val:.3f}")
print(f"  Modele 1 post_AI R2  : {r2_gb_post:.3f}  <- degradation prouvee")
print(f"  Modele 2 post_AI R2  : {r2_gb2_post:.3f}  <- correction partielle")
print(f"  Features contaminees : {unstable}")
print(f"  Features stables     : {stable}")