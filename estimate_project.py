#!/usr/bin/env python3

import pandas as pd
import pickle
import re
import sys

# La fonction de nettoyage doit être identique à celle de model.py
def clean_description(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"\[!!important\].*?guidelines\]\(.*?\)", "", text, flags=re.DOTALL)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_resources():
    """Charge le modèle et l'encodeur sauvegardés"""
    try:
        with open('model_v2.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['encoder']
    except FileNotFoundError:
        print("Erreur : Fichier 'model_v2.pkl' introuvable. Lancez d'abord l'entrainement.")
        sys.exit()

def predict_single_task(model, encoder, title, desc, subtasks, tenure, participants, ai_assisted):
    """Prépare les données et prédit les points pour une seule tâche"""
    
    # Préparation du texte (Titre + Description nettoyée)
    full_text = title.lower() + " " + clean_description(desc)
    
    # Création du DataFrame avec les colonnes exactes du modèle
    input_df = pd.DataFrame({
        'clean_text': [full_text],
        'pre_coding_subtasks': [subtasks],
        'pre_coding_desc_length': [len(desc)],
        'pre_coding_author_tenure_days': [tenure],
        'pre_coding_discussion_participants': [participants],
        'is_ai_assisted': [int(ai_assisted)]
    })

    # Prédiction et décodage
    pred_idx = model.predict(input_df)
    return encoder.inverse_transform(pred_idx)[0]

def main():
    model, encoder = load_resources()
    
    print("ESTIMATEUR DE PROJET AGILE (Somme des PR)")
    print("-" * 40)
    
    try:
        nb_taches = int(input("Combien de taches/PR contient ce projet ? "))
    except ValueError:
        print("Erreur : Veuillez entrer un nombre entier.")
        return

    backlog = []
    
    for i in range(nb_taches):
        print(f"\n--- SAISIE TACHE #{i+1} ---")
        titre = input("Titre : ")
        description = input("Description : ")
        
        try:
            sub = int(input("Nombre de sous-taches : ") or 0)
            ten = int(input("Anciennete auteur (jours) : ") or 180)
            part = int(input("Nombre de participants : ") or 1)
            ai_val = input("Assiste par IA ? (o/n) : ").lower() == 'o'
        except ValueError:
            print("Donnee invalide, utilisation des valeurs par defaut.")
            sub, ten, part, ai_val = 0, 180, 1, False

        # Calcul des points pour cette tâche
        points = predict_single_task(model, encoder, titre, description, sub, ten, part, ai_val)
        backlog.append({"titre": titre, "points": points})
        print(f"Estimation individuelle : {points} Story Points")

    # --- RAPPORT FINAL ---
    print("\n" + "=" * 45)
    print(f"{'RESUME DE L''ESTIMATION PROJET':^45}")
    print("=" * 45)
    print(f"{'Tache':<35} | {'Points':<7}")
    print("-" * 45)
    
    total_points = 0
    for item in backlog:
        # On tronque le titre s'il est trop long pour l'affichage
        affichage_titre = (item['titre'][:32] + '..') if len(item['titre']) > 32 else item['titre']
        print(f"{affichage_titre:<35} | {item['points']:<7}")
        total_points += item['points']
    
    print("-" * 45)
    print(f"{'EFFORT TOTAL DU PROJET':<35} | {total_points:<7}")
    print("=" * 45)

    # Conclusion pour le prof
    if total_points == 0:
        print("Note : Projet vide ou erreurs de saisie.")
    elif total_points < 15:
        print("Note : Projet de petite taille. Realisable en un sprint court.")
    elif total_points < 40:
        print("Note : Projet de taille moyenne. Ideal pour un sprint de 2 semaines.")
    else:
        print("Note : Projet complexe. Envisagez de le diviser en plusieurs phases.")
    print("\n")

if __name__ == "__main__":
    main()