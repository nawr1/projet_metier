import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION DES CHEMINS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) if CURRENT_DIR.endswith('src') else CURRENT_DIR
INPUT_FILE = os.path.join(ROOT_DIR, "data", "processed", "cleaned_agile_pr_level.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_tshirt_v1.pkl")

# --- DÉFINITION DU TRANSFORMER SÉMANTIQUE ---
class SemanticTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            texts = X.fillna("").tolist()
        else:
            texts = [str(t) for t in X]
        return self.model.encode(texts, show_progress_bar=False)

def map_to_tshirt_size(points):
    if points <= 2: return "S"
    elif points <= 3: return "M"
    else: return "L"

def train_model():
    if not os.path.exists(INPUT_FILE):
        print(f"Erreur : {INPUT_FILE} introuvable.")
        return

    # 1. Chargement
    df = pd.read_csv(INPUT_FILE)
    
    # 2. CRÉATION DE LA COLONNE MANQUANTE (L'erreur était ici)
    print("Groupement des Story Points en tailles T-Shirt...")
    df['story_points_grouped'] = df['story_points'].apply(map_to_tshirt_size)

    # 3. Préparation des données
    features_num = ['pre_coding_subtasks', 'pre_coding_desc_length', 
                    'pre_coding_author_tenure_days', 'pre_coding_discussion_participants', 'is_ai_assisted']
    
    X = df[['repo_name', 'clean_text'] + features_num]
    
    # 4. Encodage des labels (S, M, L -> 0, 1, 2) pour XGBoost
    le = LabelEncoder()
    y = le.fit_transform(df['story_points_grouped'])

    # 5. Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('repo', OneHotEncoder(handle_unknown='ignore'), ['repo_name']),
        ('text', SemanticTransformer(), 'clean_text'),
        ('num', StandardScaler(), features_num)
    ])

    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=2)),
        ('classifier', XGBClassifier(
            n_estimators=150, 
            learning_rate=0.03, 
            max_depth=4, 
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Entraînement du modèle (cela peut prendre 1-2 minutes)...")
    model_pipeline.fit(X_train, y_train)

    # 6. Sauvegarde
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        # On sauvegarde le pipeline ET le label encoder pour pouvoir décoder plus tard
        pickle.dump({'pipeline': model_pipeline, 'label_encoder': le}, f)
    
    print(f"✔ Modèle sauvegardé dans {MODEL_PATH}")
    print(f"Score de précision : {model_pipeline.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    train_model()