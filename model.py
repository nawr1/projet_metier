#!/usr/bin/env python3

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error

CSV_FILE = "agile_pr_level.csv"

def train_models():
    try:
        # 1. Chargement et nettoyage des données
        df = pd.read_csv(CSV_FILE)
        # On supprime les lignes où les valeurs cibles sont absentes
        df = df.dropna(subset=['story_points', 'actual_duration_hours'])
    except FileNotFoundError:
        print(f"Erreur : Le fichier {CSV_FILE} n'existe pas. Lancez le script d'extraction d'abord.")
        return

    print("--- PRÉPARATION DES DONNÉES ---")
    # On fusionne le titre et la description pour l'analyse sémantique NLP
    df['full_text'] = df['pre_coding_title'].fillna('') + ' ' + df['pre_coding_description'].fillna('')

    # =====================================================================
    # MODÈLE 1 : PRÉDIRE LES STORY POINTS (Classification)
    # Entrées : Texte, longueur du texte, nombre de sous-tâches
    # =====================================================================
    print("\n[1/2] Entraînement de l'IA pour les Story Points...")
    X_sp = df[['full_text', 'pre_coding_subtasks', 'pre_coding_desc_length']]
    y_sp = df['story_points'].astype(int)

    # Cette configuration permet de mixer du Texte Brut avec des Nombres
    preprocessor_sp = ColumnTransformer(
        transformers=[
            ('text_nlp', TfidfVectorizer(max_features=1000, stop_words='english'), 'full_text'),
            ('numeric', 'passthrough', ['pre_coding_subtasks', 'pre_coding_desc_length'])
        ]
    )

    # Pipeline = Préparer les données PUIS entraîner le modèle (RandomForest)
    model_sp = Pipeline(steps=[
        ('preprocessor', preprocessor_sp),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Entraînement et Test
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_sp, y_sp, test_size=0.2, random_state=42)
    model_sp.fit(X1_train, y1_train)
    
    acc = accuracy_score(y1_test, model_sp.predict(X1_test))
    print(f"Modèle 1 prêt ! (Précision : {acc:.2%})")

    # =====================================================================
    # MODÈLE 2 : PRÉDIRE LE TEMPS EN HEURES (Régression)
    # Entrées : Les Story Points ET la case "Utilisation de l'IA"
    # =====================================================================
    print("\n[2/2] Entraînement de l'IA pour la prédiction du Temps...")
    X_time = df[['story_points', 'is_ai_assisted']]
    y_time = df['actual_duration_hours']

    model_time = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Entraînement et Test
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_time, y_time, test_size=0.2, random_state=42)
    model_time.fit(X2_train, y2_train)
    
    mae = mean_absolute_error(y2_test, model_time.predict(X2_test))
    print(f"Modèle 2 prêt ! (Marge d'erreur moyenne : ±{mae:.1f} heures)")

    # =====================================================================
    # SAUVEGARDE DES CERVEAUX DE L'IA (.pkl)
    # =====================================================================
    with open('model_story_points.pkl', 'wb') as f:
        pickle.dump(model_sp, f)
    with open('model_time.pkl', 'wb') as f:
        pickle.dump(model_time, f)
        
    print("\nSuccès : Les deux modèles ont été sauvegardés en fichiers .pkl !")

if __name__ == "__main__":
    train_models()