# Projet_metier : Mining GitHub Repositories for Intelligent Software Effort Estimation

Ce projet implémente un pipeline complet de Data Science pour automatiser l'estimation des Story Points (effort de développement) en utilisant les données des Pull Requests GitHub. Le système transforme les descriptions textuelles et les métriques techniques en catégories de taille T-Shirt (S, M, L).

---

## Architecture du projet

```
PROJET_METIER/
├── data/
│   ├── cache/        # Cache SQLite pour l'API GitHub (github_api_cache.sqlite)
│   ├── charts/       # Visualisations de l'analyse exploratoire (PNG)
│   ├── processed/    # Données nettoyées (cleaned_agile_pr_level.csv)
│   └── raw/          # Données brutes extraites (agile_pr_level.csv)
├── models/           # Modèles entraînés (model_tshirt_v1.pkl)
├── notebooks/        # Notebooks Jupyter pour l'exploration
├── src/
│   ├── features/
│   │   ├── clean_data.py       # Nettoyage sémantique et Regex
│   │   └── fetch_repos.py      # Script d'extraction API GitHub
│   ├── model/
│   │   └── model.py            # Pipeline d'entraînement et évaluation
│   └── tests/
│       ├── predict.py          # Estimation unitaire d'une tâche
│       └── estimate_project.py # Calculateur de charge pour un projet complet
├── .env              # Clés de configuration (GITHUB_TOKEN)
├── .gitignore        # Fichiers exclus du versionnage
├── main.py           # Orchestrateur central
└── requirements.txt  # Dépendances du projet
```

---

## Installation et configuration

### 1. Récupération du projet

```bash
git clone https://github.com/nawr1/projet_metier.git
cd PROJET_METIER
```

### 2. Installation des dépendances

```bash
pip install pandas scikit-learn xgboost imbalanced-learn sentence-transformers PyGithub python-dotenv requests-cache
```

### 3. Configuration du token

Créer un fichier `.env` à la racine :

```
GITHUB_TOKEN=votre_jeton_personnel_github
```

---

## Utilisation

### Extraction et nettoyage

```bash
python main.py
```

### Entraînement du modèle

```bash
python src/model/model.py
```

### Estimation d'une tâche

```bash
python src/tests/predict.py
```

### Estimation de projet

```bash
python src/tests/estimate_project.py
```

---

## Paramètres de modélisation

| Paramètre     | Valeur             |
|---------------|--------------------|
| Algorithme    | XGBoost Classifier |
| n_estimators  | 100                |
| max_depth     | 3                  |
| learning_rate | 0.05               |
| reg_alpha     | 2.0                |
| reg_lambda    | 5.0                |

> **Gap de généralisation cible : < 10%**
