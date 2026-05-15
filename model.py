#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

CSV_FILE = "agile_pr_level.csv"

def clean_description(text):
    if pd.isna(text): return ""
    text = re.sub(r'[^\w\s\d\(\)\[\]\.\,\!\?\:\'\-\/]', ' ', text)
    text = text.lower()
    #  Supprimer les blocs de "Contribution Guidelines" (Boilerplate)
    text = re.sub(r"\[!!important\].*?guidelines\]\(.*?\)", "", text, flags=re.DOTALL)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les balises Markdown inutiles (##, **, etc.)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    
    #  Nettoyer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def train_model():
    try:
        df = pd.read_csv(CSV_FILE)
        # Nettoyage initial
        df = df.dropna(subset=['story_points'])
        # On garde les classes qui ont au moins 5 exemples pour que l'IA apprenne vraiment
        df = df[df.groupby('story_points')['story_points'].transform('count') >= 5]
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return

    print("--- NETTOYAGE ET FEATURE ENGINEERING ---")
    
    # Nettoyage du texte
    df['clean_text'] = (df['pre_coding_title'].fillna('') + " " + 
                        df['pre_coding_description'].apply(clean_description))

    # On convertit le booléen en chiffre (0 ou 1)
    df['is_ai_assisted'] = df['is_ai_assisted'].astype(int)

    # Sélection des colonnes (On en ajoute 3 nouvelles !)
    features = [
        'clean_text', 
        'pre_coding_subtasks', 
        'pre_coding_desc_length',
        'pre_coding_author_tenure_days',        # Nouvel indicateur
        'pre_coding_discussion_participants',   # Nouvel indicateur
        'is_ai_assisted'                        # Nouvel indicateur
    ]
    
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['story_points'].astype(int))

    # PIPELINE
    # On sépare le texte (TF-IDF) et les nombres (StandardScaler pour XGBoost)
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english'), 'clean_text'),
            ('num', StandardScaler(), [
                'pre_coding_subtasks', 
                'pre_coding_desc_length', 
                'pre_coding_author_tenure_days', 
                'pre_coding_discussion_participants',
                'is_ai_assisted'
            ])
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,            # On réduit la profondeur (évite de créer des règles trop complexes)
            min_child_weight=4,     # Empêche l'IA de se baser sur des cas trop isolés
            reg_lambda=20,          # L2 Regularization : Force l'IA à ne pas trop compter sur une seule colonne
            reg_alpha=5,            # L1 Regularization : Aide à ignorer les données peu importantes
            subsample=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entraînement sur {len(X_train)} lignes...")
    model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n================ REPORT ================")
    print(f"Nouvelle Précision : {acc:.2%}")
    # Affiche le détail par Story Point (pour voir où il se trompe)
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    with open('model_v2.pkl', 'wb') as f:
        pickle.dump({'model': model, 'encoder': le}, f)

if __name__ == "__main__":
    train_model()