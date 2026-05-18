import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

CSV_FILE = "agile_pr_level.csv"

def map_to_tshirt_size(sp):
    if sp <= 2: return 0  # SMALL
    elif sp <= 5: return 1  # MEDIUM
    else: return 2  # LARGE

# J'ai mis un modèle plus adapté au code informatique
class SemanticTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    def fit(self, X, y=None): return self
    def transform(self, X):
        print(f"Génération des embeddings pour {len(X)} lignes...")
        return self.model.encode(X.tolist(), show_progress_bar=False)

def clean_description(text):
    if pd.isna(text): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[#\*_>\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    try:
        df = pd.read_csv(CSV_FILE)
        df = df.dropna(subset=['story_points'])
        df['story_points_grouped'] = df['story_points'].apply(map_to_tshirt_size)
        df = df[df.groupby('story_points_grouped')['story_points_grouped'].transform('count') >= 5]
    except Exception as e:
        print(f"Erreur : {e}")
        return

    print(f"Distribution des classes : \n{df['story_points_grouped'].value_counts()}")

    df['clean_text'] = (df['pre_coding_title'].fillna('') + " " + 
                        df['pre_coding_description'].apply(clean_description))

    # Features exactes (SANS repo_name)
    X = df[['clean_text', 'pre_coding_subtasks', 'pre_coding_desc_length', 
            'pre_coding_author_tenure_days', 'pre_coding_discussion_participants', 'is_ai_assisted']]
    y = df['story_points_grouped']

    target_names = ["SMALL (1-2)", "MEDIUM (3-5)", "LARGE (8+)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', SemanticTransformer(), 'clean_text'),
            ('num', StandardScaler(), [
                'pre_coding_subtasks', 'pre_coding_desc_length', 
                'pre_coding_author_tenure_days', 'pre_coding_discussion_participants', 'is_ai_assisted'
            ])
        ]
    )

    # Pipeline vierge (les paramètres seront trouvés par GridSearch)
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=2)),
        ('classifier', XGBClassifier(random_state=42, eval_metric='mlogloss'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Recherche des meilleurs hyperparamètres en cours (Patientez...)...")
    
    # L'IA va tester ces différentes combinaisons
    param_grid = {
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__n_estimators': [100, 150]
    }

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)

    print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")

    # Récupération du meilleur modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nReport Final")
    print(f"Précision T-Shirt Sizing : {accuracy_score(y_test, y_pred):.2%}")
    present_classes = [target_names[i] for i in sorted(y_test.unique())]
    print(classification_report(y_test, y_pred, target_names=present_classes))

    # Sauvegarde
    with open('model_agile.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Modèle sauvegardé sous 'model_agile.pkl'.")

if __name__ == "__main__":
    train_model()