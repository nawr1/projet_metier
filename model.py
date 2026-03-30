import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# =============================
# 1. LOAD DATA
# =============================
df = pd.read_csv("repos_dataset.csv")
print("✅ Data loaded:", df.shape)
print(df.head())

# =============================
# 2. CLEAN DATA
# =============================
# Drop non-numeric columns
df = df.drop(columns=["name", "language"])

# Drop rows with missing values
df = df.dropna()

print("\n✅ Clean data shape:", df.shape)
print(df.describe())

# =============================
# 3. DEFINE TARGET & FEATURES
# =============================
# Target = effort proxy (commits_count)
# Features = everything else

X = df.drop(columns=["commits_count"])
y = df["commits_count"]

print("\n✅ Features:", list(X.columns))
print("✅ Target: commits_count")

# =============================
# 4. CORRELATION HEATMAP
# =============================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
print("\n✅ Heatmap saved as correlation_heatmap.png")

# =============================
# 5. CORRELATION WITH TARGET
# =============================
print("\n📊 Correlation with commits_count (effort):")
print(df.corr()["commits_count"].sort_values(ascending=False))

# =============================
# 6. SPLIT DATA
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Train size: {len(X_train)} | Test size: {len(X_test)}")

# =============================
# 7. LINEAR REGRESSION
# =============================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n📈 Linear Regression Results:")
print(f"   MAE  : {mae_lr:,.0f} commits")
print(f"   R²   : {r2_lr:.4f}")

# =============================
# 8. RANDOM FOREST
# =============================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n🌲 Random Forest Results:")
print(f"   MAE  : {mae_rf:,.0f} commits")
print(f"   R²   : {r2_rf:.4f}")

# =============================
# 9. FEATURE IMPORTANCE (Random Forest)
# =============================
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importance.plot(kind="bar", color="steelblue")
plt.title("Feature Importance for Effort Estimation")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("\n✅ Feature importance chart saved as feature_importance.png")

# =============================
# 10. MODEL COMPARISON CHART
# =============================
models = ["Linear Regression", "Random Forest"]
r2_scores = [r2_lr, r2_rf]
mae_scores = [mae_lr, mae_rf]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(models, r2_scores, color=["steelblue", "seagreen"])
axes[0].set_title("R² Score (higher = better)")
axes[0].set_ylim(0, 1)

axes[1].bar(models, mae_scores, color=["steelblue", "seagreen"])
axes[1].set_title("MAE (lower = better)")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("\n✅ Model comparison chart saved as model_comparison.png")

# =============================
# 11. SUMMARY
# =============================
print("\n" + "="*40)
print("📋 FINAL SUMMARY")
print("="*40)
print(f"Linear Regression → R²: {r2_lr:.4f} | MAE: {mae_lr:,.0f}")
print(f"Random Forest     → R²: {r2_rf:.4f} | MAE: {mae_rf:,.0f}")

if r2_rf > r2_lr:
    print("\n🏆 Random Forest performs better!")
else:
    print("\n🏆 Linear Regression performs better!")

print("\n🔑 Most important features for effort estimation:")
for i, (feat, score) in enumerate(importance.items()):
    print(f"   {i+1}. {feat}: {score:.4f}")