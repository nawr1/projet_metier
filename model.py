import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# 1. LOAD DATA
# =============================
df = pd.read_csv("agile_estimation_dataset.csv")

# =============================
# 2. CLEAN DATA
# =============================
# On garde une copie pour l'affichage mais on nettoie pour le modèle
df_model = df.drop(columns=["name", "language"])
df_model = df_model.dropna()

# =============================
# 3. DEFINE TARGET & FEATURES
# =============================
# TARGET = avg_lead_time_days (Le temps/coût de réalisation)
# FEATURES = Tout le reste (y compris total_commits)

X = df_model.drop(columns=["avg_lead_time_days"]) 
y = df_model["avg_lead_time_days"]

print("\nFeatures utilisées pour prédire le temps :", list(X.columns))
print("Cible (Target) : avg_lead_time_days")

# =============================
# 4. CORRELATION WITH TARGET
# =============================
print("\nCorrélation avec le Temps de réalisation (Lead Time):")
print(df_model.corr()["avg_lead_time_days"].sort_values(ascending=False))

# =============================
# 5. SPLIT DATA
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 6. RANDOM FOREST (Généralement plus performant ici)
# =============================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRésultats Random Forest (Prédiction du délai en jours) :")
print(f"   Erreur moyenne (MAE) : {mae_rf:.2f} jours")
print(f"   Score R²            : {r2_rf:.4f}")

# =============================
# 7. FEATURE IMPORTANCE
# =============================
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance.plot(kind="bar", color="darkorange")
plt.title("Quels facteurs influencent le plus le temps de réalisation (Lead Time) ?")
plt.ylabel("Score d'importance")
plt.tight_layout()
plt.show()

# =============================
# 8. RÉSUMÉ FINAL
# =============================
print("\n" + "="*40)
print("🔑 ANALYSE DU COÛT TEMPS")
print("="*40)
for i, (feat, score) in enumerate(importance.items()):
    print(f"   {i+1}. {feat}: {score:.4f}")