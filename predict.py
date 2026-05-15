#!/usr/bin/env python3

import pandas as pd
import pickle
import re

def clean_description(text):
    if pd.isna(text): return ""
    text = text.lower()
    # Supprimer les blocs de "Contribution Guidelines"
    text = re.sub(r"\[!!important\].*?guidelines\]\(.*?\)", "", text, flags=re.DOTALL)
    # Supprimer les URLs et Markdown
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_new_model():
    """Charge le modèle XGBoost et l'encodeur depuis le fichier pkl"""
    try:
        with open('model_v2.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['encoder']
    except FileNotFoundError:
        print("Erreur : Fichier 'model_v2.pkl' introuvable.")
        exit()

def predict_ticket(title, description, subtasks, tenure, participants, ai_assisted):
    model, encoder = load_new_model()

    # Préparation des données
    cleaned_desc = clean_description(description)
    full_text = title.lower() + " " + cleaned_desc
    
    input_data = pd.DataFrame({
        'clean_text': [full_text],
        'pre_coding_subtasks': [subtasks],
        'pre_coding_desc_length': [len(description)],
        'pre_coding_author_tenure_days': [tenure],
        'pre_coding_discussion_participants': [participants],
        'is_ai_assisted': [int(ai_assisted)]
    })

    # Prédiction
    prediction_idx = model.predict(input_data)
    final_sp = encoder.inverse_transform(prediction_idx)[0]

    # Affichage des résultats
    print("\nRESULTATS DE LA PREDICTION")
    print("-" * 30)
    print(f"Ticket : {title}")
    print(f"Anciennete auteur : {tenure} jours")
    print(f"Participants : {participants}")
    print(f"Estimation : {final_sp} Story Points")
    print("-" * 30)
    
    if final_sp >= 8:
        print("Conseil : Ticket complexe, a decouper.")
    else:
        print("Conseil : Taille de ticket correcte.")
    print("\n")

if __name__ == "__main__":
    print("ESTIMATEUR STORY POINTS XGBOOST")
    
    t = input("Titre du ticket : ")
    
    if not t.strip():
        # Exemple de test par défaut
        predict_ticket(
            title="Refactor database connection pool",
            description="Update pool size and handle timeouts.",
            subtasks=2,
            tenure=365,
            participants=3,
            ai_assisted=0
        )
    else:
        d = input("Description : ")
        s = int(input("Nombre de sous-taches : ") or 0)
        ten = int(input("Anciennete auteur (jours) : ") or 30)
        part = int(input("Nombre de participants : ") or 1)
        ai = input("Assiste par IA ? (o/n) : ").lower() == 'o'

        predict_ticket(t, d, s, ten, part, ai)