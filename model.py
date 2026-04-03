import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer # 🌟 VOICI BERT !
import joblib

print(" Chargement du dataset...")
df = pd.read_csv("ultimate_agile_dataset.csv").dropna()
print(f" Data loaded: {df.shape[0]} issues.")

# 1. PRÉPARATION DU TEXTE AVEC BERT
print(" Initialisation de BERT (Cela peut prendre 1 min la première fois)...")
# C'est un modèle BERT optimisé et rapide de HuggingFace
bert_model = SentenceTransformer('all-MiniLM-L6-v2') 

print(" Transformation du texte par BERT en cours...")
df['texte'] = df['title'] + ". " + df['description']
# BERT transforme chaque phrase en une liste de 384 nombres (Vecteurs)
text_embeddings = bert_model.encode(df['texte'].tolist()) 

# 2. FUSION DES FEATURES (Texte + Chiffres)
print(" Combinaison avec la Complexité et l'Expérience...")
# On crée le tableau final des Features (X)
X_text = pd.DataFrame(text_embeddings)
X_numeric = df[['dev_experience', 'code_complexity']].reset_index(drop=True)

# On colle le résultat de BERT avec les chiffres de l'Extracteur
X = pd.concat([X_numeric, X_text], axis=1)

# Variables à la limite (X = nos Features à tous, y = Le temps à deviner)
# On s'assure que tout est en String pour scikit-learn
X.columns = X.columns.astype(str) 
y = df["resolution_hours"] 

# 3. ENTRAÎNEMENT DU MODÈLE
print(" Entraînement du modèle Machine Learning...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. ÉVALUATION
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*40)
print(" RÉSULTATS DU MODÈLE AGILE COMPLET")
print("="*40)
print(f"L'erreur moyenne est de : {mae:.2f} heures.")
print("Plus on aura de données plus tard, plus cette erreur baissera !")

# 5. SAUVEGARDE DE L'IA POUR LE SCRIPT DE PREDICTION
joblib.dump(rf, "modele_rf_complet.pkl")
print(" Modèle final sauvegardé avec succès !")