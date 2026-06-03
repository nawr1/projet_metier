import pandas as pd
import pickle
import os
import re
from src.model.model import SemanticTransformer 

MODEL_PATH = "models/model_tshirt_v1.pkl"

def clean_input(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Modèle introuvable à {MODEL_PATH}")
        return None, None
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['pipeline'], data['label_encoder']

def get_input_data(is_project=False, idx=1):
    """Récupère les entrées utilisateur pour une PR ou une tâche de projet"""
    prefix = f" [Tâche #{idx}]" if is_project else ""
    
    print(f"\n--- Saisie des informations{prefix} ---")
    titre = input("Titre (PR ou Ticket) : ")
    desc = input("Description : ")
    subtasks = int(input("Nombre de sous-tâches (0 par défaut) : ") or 0)
    
    return pd.DataFrame([{
        'repo_name': 'inference_project',
        'clean_text': clean_input(titre) + " " + clean_input(desc),
        'pre_coding_subtasks': subtasks,
        'pre_coding_desc_length': len(desc),
        'pre_coding_author_tenure_days': 180,
        'pre_coding_discussion_participants': 1,
        'is_ai_assisted': 0
    }])

def main():
    pipeline, le = load_model()
    if not pipeline: return

    print("\n" + "="*40)
    print("      ESTIMATEUR DE STORY POINTS")
    print("="*40)
    print("1. Prédire la complexité d'une PR")
    print("2. Prédire l'effort d'un Projet complet")
    print("3. Quitter")
    
    choix = input("\nVotre choix : ")

    if choix == "1":
        # MODE PR UNIQUE
        input_df = get_input_data()
        pred_idx = pipeline.predict(input_df)[0]
        label = le.inverse_transform([pred_idx])[0]
        
        mapping_full = {"S": "SMALL (1-2 SP)", "M": "MEDIUM (3-5 SP)", "L": "LARGE (8+ SP)"}
        print(f"\nESTIMATION : {mapping_full.get(label, label)}")

    elif choix == "2":
        # MODE PROJET (Somme des tâches)
        try:
            nb = int(input("\nCombien de PR/Tâches compose le projet ? "))
            total_sp = 0
            # Mapping numérique pour le calcul du total
            points_map = {"S": 2, "M": 5, "L": 8} 
            
            summary = []
            for i in range(1, nb + 1):
                input_df = get_input_data(is_project=True, idx=i)
                pred_idx = pipeline.predict(input_df)[0]
                label = le.inverse_transform([pred_idx])[0]
                
                points = points_map.get(label, 0)
                total_sp += points
                summary.append(f"Tâche {i}: {label} (~{points} SP)")

            print(f"\n" + "-"*30)
            print("DÉTAILS DU PROJET :")
            for s in summary: print(f"  {s}")
            print(f"\n ESTIMATION TOTALE : ~{total_sp} Story Points")
            print("-"*30)
            
        except ValueError:
            print("Erreur : Veuillez entrer un nombre valide.")

    elif choix == "3":
        print("Fin du programme.")
    else:
        print("Choix invalide.")

if __name__ == "__main__":
    main()