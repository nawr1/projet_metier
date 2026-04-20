"""
predict.py
----------
Donne des recommandations automatiques au développeur
basées sur les features d'une nouvelle PR.
"""


def recommend(pr_features: dict, selected_features: list) -> str:
    """
    Analyse les features d'une PR et retourne des recommandations.

    Paramètres
    ----------
    pr_features       : dict — valeurs des features de la nouvelle PR
    selected_features : list — features jugées fiables par le modèle

    Retourne
    --------
    str — recommandations textuelles
    """
    recommendations = []

    churn             = pr_features.get("churn", 0)
    nb_files          = pr_features.get("nb_files", 0)
    complexity_delta  = pr_features.get("complexity_delta", 0)
    revision_cycles   = pr_features.get("revision_cycles", 0)
    nb_comments       = pr_features.get("nb_comments", 0)
    refactoring_ratio = pr_features.get("refactoring_ratio", 0)
    velocity          = pr_features.get("velocity", 0)

    # Signal d'assistance IA probable
    if churn > 500 and refactoring_ratio < 0.2:
        recommendations.append(
            "Signal d'assistance IA probable (churn élevé + peu de retouches). "
            "Renforcer la revue humaine avant le merge."
        )

    # PR trop large
    if nb_files > 15:
        recommendations.append(
            f"Cette PR touche {nb_files} fichiers. "
            "Envisager de la découper en plusieurs PRs plus ciblées."
        )

    # Complexité élevée
    if complexity_delta > 10:
        recommendations.append(
            "Complexité cyclomatique ajoutée élevée. "
            "Ajouter des tests unitaires pour couvrir les nouveaux chemins logiques."
        )

    # Vélocité anormale
    if velocity > 1000:
        recommendations.append(
            "Vélocité de commit anormalement élevée. "
            "Vérifier si le code a été généré automatiquement et valider sa qualité."
        )

    # Aucun commentaire + cycles = PR non discutée
    if nb_comments == 0 and revision_cycles == 0 and churn > 300:
        recommendations.append(
            "Aucun commentaire ni révision pour une PR volumineuse. "
            "Demander au moins un reviewer humain avant le merge."
        )

    if not recommendations:
        recommendations.append("PR dans les normes. Aucune alerte détectée.")

    header = "=== Recommandations pour cette PR ===\n"
    return header + "\n".join(f"  - {r}" for r in recommendations)


def print_pr_summary(pr_features: dict, selected_features: list):
    """
    Affiche un résumé complet de la PR et ses recommandations.
    """
    print("\n=== Features de la PR analysée ===")
    for k, v in pr_features.items():
        flag = " (feature fiable)" if k in selected_features else ""
        print(f"  {k:25s} = {v}{flag}")

    print()
    print(recommend(pr_features, selected_features))