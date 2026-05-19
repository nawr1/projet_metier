import pandas as pd
import pickle
import os
import re

MODEL_PATH = "models/model_tshirt_v1.pkl"

def clean_input(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_one():
    if not os.path.exists(MODEL_PATH):
        print("Erreur : Modèle introuvable dans models/. Entraînez-le d'abord.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print("\n--- PRÉDICTION UNIQUE ---")
    titre = input("Titre de la PR : ")
    desc = input("Description : ")
    
    # Préparation du format identique à l'entraînement
    input_data = pd.DataFrame([{
        'repo_name': 'inference_test',
        'clean_text': clean_input(titre) + " " + clean_input(desc),
        'pre_coding_subtasks': int(input("Nombre de sous-tâches (0 par défaut) : ") or 0),
        'pre_coding_desc_length': len(desc),
        'pre_coding_author_tenure_days': 180, 
        'pre_coding_discussion_participants': 1,
        'is_ai_assisted': 0
    }])

    res = model.predict(input_data)[0]
    mapping = {0: "SMALL (1-2 SP)", 1: "MEDIUM (3-5 SP)", 2: "LARGE (8+ SP)"}
    print(f"\nESTIMATION : {mapping[res]}")

if __name__ == "__main__":
    predict_one()