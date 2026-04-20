import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement des données
df = pd.read_csv("agile_raw_data.csv")

# --- SIMULATION (À retirer si tu as déjà la colonne story_points) ---
# Si tu n'as pas de points, on simule une suite de Fibonacci basée sur la complexité
def simulate_sp(row):
    complexity = (row['code_churn'] * 0.5) + (row['review_cycles'] * 20) + (row['pr_comments'] * 5)
    if complexity < 50: return 1
    if complexity < 150: return 3
    if complexity < 300: return 5
    if complexity < 600: return 8
    return 13

if 'story_points' not in df.columns:
    print("Simulation de la variable cible 'story_points'...")
    df['story_points'] = df.apply(simulate_sp, axis=1)
# --------------------------------------------------------------------

# 2. Préparation des données (Preprocessing)
# On convertit la colonne textuelle 'era' en nombres (Label Encoding)
df['era'] = df['era'].map({'pre_AI': 0, 'transition': 1, 'post_AI': 2})

# Sélection des features (X) et de la cible (y)
features = [
    "code_churn", "files_changed", "review_cycles", "pr_comments", 
    "refactoring_ratio", "commit_velocity", "active_contributors", 
    "ai_mentions", "ai_commits", "duration_hours", "era"
]
X = df[features]
y = df['story_points']

# Division Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Création et entraînement du modèle XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)

print("Entraînement du modèle XGBoost...")
model.fit(X_train, y_train)

# 4. Évaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- RÉSULTATS DU MODÈLE ---")
print(f"MAE (Erreur moyenne en points) : {mae:.2f}")
print(f"R² Score (Précision) : {r2:.2f}")

# 5. Importance des caractéristiques (Feature Importance)
# C'est ici qu'on voit si l'IA impacte les Story Points
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight')
plt.title("Facteurs influençant les Story Points (XGBoost)")
plt.show()

# 6. Analyse de l'impact IA (Optionnel)
# On regarde la différence de prédiction moyenne par Era
df['predicted_sp'] = model.predict(X)
analysis = df.groupby('era')['predicted_sp'].mean()
print("\nPoints moyens prédits par époque (0:pre, 1:trans, 2:post):")
print(analysis)