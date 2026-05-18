#!/usr/bin/env python3

import pandas as pd
import pickle
import re

def clean_description(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"\[!!important\].*?guidelines\]\(.*?\)", "", text, flags=re.DOTALL)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_new_model():
    try:
        # On charge le nouveau nom de fichier sauvegardé par model.py
        with open('model_agile.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Erreur : Fichier 'model_agile.pkl' introuvable. Lancez 'python model.py' d'abord.")
        exit()

def predict_ticket(title, description, subtasks, tenure, participants, ai_assisted):
    model = load_new_model()

    cleaned_desc = clean_description(description)
    full_text = title.lower() + " " + cleaned_desc
    
    # Mêmes features que dans model.py
    input_data = pd.DataFrame({
        'clean_text': [full_text],
        'pre_coding_subtasks': [subtasks],
        'pre_coding_desc_length': [len(description)],
        'pre_coding_author_tenure_days': [tenure],
        'pre_coding_discussion_participants': [participants],
        'is_ai_assisted': [int(ai_assisted)]
    })

    # Prédiction
    prediction_idx = model.predict(input_data)[0]
    
    # Transformation de l'index (0, 1, 2) en Texte lisible
    tshirt_map = {0: "SMALL (1 à 2 SP)", 1: "MEDIUM (3 à 5 SP)", 2: "LARGE (8+ SP)"}
    final_sp = tshirt_map[prediction_idx]

    # Affichage des résultats
    print("\nRESULTATS DE LA PREDICTION")
    print("-" * 30)
    print(f"Ticket : {title}")
    print(f"Anciennete auteur : {tenure} jours")
    print(f"Participants : {participants}")
    print(f"Estimation IA : {final_sp}")
    print("-" * 30)
    
    if prediction_idx == 2: # Si LARGE
        print("Conseil : Ce ticket semble complexe, essayez de le decouper en sous-taches.")
    else:
        print("Conseil : La taille de ce ticket est correcte pour un Sprint.")
    print("\n")

if __name__ == "__main__":
    print("ESTIMATEUR STORY POINTS AGILE")
    
    t = input("Titre du ticket : ")
    
    if not t.strip():
        print("Test par defaut lance...")
        predict_ticket(
            title="Refactor database connection pool",
            description="Update pool size and handle timeouts to prevent server crash.",
            subtasks=3,
            tenure=365,
            participants=4,
            ai_assisted=0
        )
    else:
        d = input("Description : ")
        s = int(input("Nombre de sous-taches : ") or 0)
        ten = int(input("Anciennete auteur (jours) : ") or 30)
        part = int(input("Nombre de participants : ") or 1)
        ai = input("Assiste par IA ? (o/n) : ").lower() == 'o'

        predict_ticket(t, d, s, ten, part, ai)