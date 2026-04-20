"""
main.py — Point d'entrée du projet

Usage :
    python main.py --collect    # Lance la collecte des données GitHub
    python main.py --train      # Lance l'analyse et l'entraînement
    python main.py --predict    # Lance une recommandation sur une PR exemple
    python main.py              # Lance tout (collect + train + predict)
"""

import argparse
from fetch_repos import collect_all_data
from model import run_analysis
from predict import print_pr_summary


def main():
    parser = argparse.ArgumentParser(description="GitHub Effort Estimation — AI Era")
    parser.add_argument("--collect", action="store_true", help="Collecter les données GitHub")
    parser.add_argument("--train",   action="store_true", help="Entraîner et évaluer les modèles")
    parser.add_argument("--predict", action="store_true", help="Tester la recommandation sur une PR exemple")
    args = parser.parse_args()

    # Si aucun argument → tout lancer
    run_all = not (args.collect or args.train or args.predict)

    selected_features = []

    # ------------------------------------------------------------------
    # ÉTAPE 1 : Collecte
    # ------------------------------------------------------------------
    if args.collect or run_all:
        print("=" * 60)
        print("ÉTAPE 1 — Collecte des données GitHub")
        print("Durée estimée : 20 à 40 minutes selon le rate limit")
        print("=" * 60)
        collect_all_data()

    # ------------------------------------------------------------------
    # ÉTAPE 2 : Entraînement et analyse
    # ------------------------------------------------------------------
    if args.train or run_all:
        print("\n" + "=" * 60)
        print("ÉTAPE 2 — Entraînement et analyse des modèles")
        print("=" * 60)
        results_classic, results_adaptive, drift, selected_features = run_analysis()

        print("\n--- Résumé des modèles ---")
        print("Modèle classique (pré-IA → post-IA) :")
        for model, scores in results_classic.items():
            print(f"  {model:15s} → R²={scores['R2']} | MAE={scores['MAE']} | CV R²={scores['CV_R2']}")
        print("Modèle adaptatif (post-IA + feature selection) :")
        for model, scores in results_adaptive.items():
            print(f"  {model:15s} → R²={scores['R2']} | MAE={scores['MAE']} | CV R²={scores['CV_R2']}")

        print("\n--- Métriques contaminées par l'IA ---")
        for feat, info in drift.items():
            if info["Statut"] == "CONTAMINEE":
                print(f"  {feat:25s} dérive de {info['Dérive (%)']:+.1f}%")

        print("\n--- Features fiables retenues ---")
        print(f"  {selected_features}")

    # ------------------------------------------------------------------
    # ÉTAPE 3 : Recommandation sur une PR exemple
    # ------------------------------------------------------------------
    if args.predict or run_all:
        print("\n" + "=" * 60)
        print("ÉTAPE 3 — Recommandation sur une PR exemple")
        print("=" * 60)

        # Exemple de PR avec un profil typique d'assistance IA
        exemple_pr = {
            "churn":             850,
            "additions":         800,
            "deletions":          50,
            "nb_files":           18,
            "complexity_delta":   12,
            "revision_cycles":     0,
            "nb_comments":         2,
            "nb_commits":          1,
            "velocity":         1200,
            "refactoring_ratio": 0.05,
        }

        # Si on n'a pas fait l'entraînement dans ce run, on utilise toutes les features
        if not selected_features:
            from config import FEATURE_COLS
            selected_features = FEATURE_COLS

        print_pr_summary(exemple_pr, selected_features)

    print("\n" + "=" * 60)
    print("Fichiers générés :")
    print("  data_pre_ai.csv   — données collectées pre-IA")
    print("  data_post_ai.csv  — données collectées post-IA")
    print("  results.png       — visualisations comparatives")
    print("=" * 60)


if __name__ == "__main__":
    main()