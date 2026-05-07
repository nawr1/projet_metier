#!/usr/bin/env python3

import pandas as pd
import pickle

def load_models():
    """Charge les cerveaux entraînés par model.py"""
    try:
        with open('model_story_points.pkl', 'rb') as f:
            model_sp = pickle.load(f)
        with open('model_time.pkl', 'rb') as f:
            model_time = pickle.load(f)
        return model_sp, model_time
    except FileNotFoundError:
        print("Erreur : Modèles introuvables. Vous devez d'abord lancer 'python model.py'.")
        exit()

def predict_ticket(title, description, subtasks):
    model_sp, model_time = load_models()

    # 1. Formatage des données d'entrée comme pendant l'entraînement
    full_text = title + " " + description
    desc_length = len(description)

    # DataFrame pour le Modèle 1
    input_sp = pd.DataFrame({
        'full_text': [full_text],
        'pre_coding_subtasks': [subtasks],
        'pre_coding_desc_length': [desc_length]
    })

    # 2. L'IA devine la complexité (Story Points)
    predicted_sp = model_sp.predict(input_sp)[0]

    # 3. L'IA devine le Temps SANS assistance IA (is_ai_assisted = 0)
    input_time_human = pd.DataFrame({
        'story_points': [predicted_sp],
        'is_ai_assisted': [0]
    })
    time_human = model_time.predict(input_time_human)[0]

    # 4. L'IA devine le Temps AVEC assistance IA (is_ai_assisted = 1)
    input_time_ai = pd.DataFrame({
        'story_points': [predicted_sp],
        'is_ai_assisted': [1]
    })
    time_ai = model_time.predict(input_time_ai)[0]

    # 5. Affichage Dashboard
    print("\n" + "="*55)
    print(" RÉSULTATS DE LA SIMULATION IA")
    print("="*55)
    print(f" Ticket : {title}")
    print(f" Effort estimé (Complexité) : {predicted_sp} Story Points")
    print("-" * 55)
    print(f" Temps de réalisation prévu (Humain) : {time_human:.1f} heures")
    print(f" Temps de réalisation prévu (Copilot) : {time_ai:.1f} heures")
    
    gain = time_human - time_ai
    if gain > 0:
        print(f" Gain de productivité avec IA      : {gain:.1f} heures sauvées !")
    print("="*55 + "\n")

if __name__ == "__main__":
    print("=== ESTIMATEUR AGILE V2.0 ===")
    print("Saisissez les informations de votre futur ticket.")
    print("(Laissez vide et appuyez sur Entrée pour utiliser un exemple de test)\n")
    
    t = input("Titre du ticket : ")
    
    if not t.strip():
        print("-> Utilisation du ticket de test automatique...")
        t = "Migrer la base de données vers PostgreSQL"
        d = "La tâche implique de réécrire les requêtes SQL, mettre à jour le fichier docker-compose. - [ ] Backup - [ ] Migration schema - [ ] Tests."
        s = 3
    else:
        d = input("Description de la tâche : ")
        s_input = input("Nombre de sous-tâches (cases à cocher, ex: 2) : ")
        s = int(s_input) if s_input.isdigit() else 0

    predict_ticket(t, d, s)