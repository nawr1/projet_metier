import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# 1. CHARGER L'IA ET BERT
print(" Réveil de BERT et du modèle prédictif...")
try:
    rf = joblib.load("modele_rf_complet.pkl")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    print(" Fais d'abord tourner model.py !")
    exit()

print("\n" + "="*50)
print(" SIMULATEUR AGILE : IA (BERT + RADON)")
print("="*50)

# 2. INFORMATIONS DONNÉES PAR LE CHEF DE PROJET
print("\n Nouveau Ticket :")
titre = input("Titre du ticket : ")
description = input("Description de la tâche : ")

print("\n Contexte Technique :")
experience = int(input("Historique du dev assigné (Nombre de commits passés) : "))
complexite = float(input("Complexité cyclomatique estimée du code (Ex: 1.0 (facile) à 15.0 (horrible)) : "))

# 3. L'IA FAIT SON CALCUL
print("\n BERT lit le texte...")
texte_complet = titre + ". " + description
texte_emb = bert_model.encode([texte_complet])

print(" Combinaison des métriques agiles...")
X_texte_df = pd.DataFrame(texte_emb)
X_numerique_df = pd.DataFrame({"dev_experience": [experience], "code_complexity": [complexite]})

# Fusion exacte comme dans model.py
X_input = pd.concat([X_numerique_df, X_texte_df], axis=1)
X_input.columns = X_input.columns.astype(str)

# 4. VERDICT
heures_estimees = rf.predict(X_input)[0]
jours_estimes = heures_estimees / 8

print("\n" + "="*40)
print(" ESTIMATION DE L'EFFORT (TIME-TO-RESOLVE)")
print("="*40)
print(f"Heures brutes : {heures_estimees:.1f} heures de travail")
print(f"Sprint Agile  : ~{jours_estimes:.1f} Jours")
print("="*40)