import pandas as pd
import pickle
import os
from tests.predict import clean_input # On réutilise le nettoyage de predict.py

MODEL_PATH = "models/model_tshirt_v1.pkl"

def estimate_full_project():
    if not os.path.exists(MODEL_PATH):
        print("Erreur : Modèle introuvable.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    nb = int(input("Combien de tâches dans ce projet ? "))
    total_sp_min = 0
    
    for i in range(nb):
        print(f"\n--- Tâche #{i+1} ---")
        titre = input("Titre : ")
        desc = input("Description : ")
        
        input_df = pd.DataFrame([{
            'repo_name': 'project_inference',
            'clean_text': clean_input(titre) + " " + clean_input(desc),
            'pre_coding_subtasks': 0,
            'pre_coding_desc_length': len(desc),
            'pre_coding_author_tenure_days': 180,
            'pre_coding_discussion_participants': 1,
            'is_ai_assisted': 0
        }])

        pred = model.predict(input_df)[0]
        # Mapping approximatif pour le calcul total
        sp_values = {0: 2, 1: 5, 2: 8} 
        total_sp_min += sp_values[pred]

    print(f"\n{'='*30}")
    print(f"ESTIMATION TOTALE DU PROJET : ~{total_sp_min} Story Points")
    print(f"{'='*30}")

if __name__ == "__main__":
    estimate_full_project()