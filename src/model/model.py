import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION DES CHEMINS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
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

    # 1. Chargement (AUCUNE MODIFICATION DES DONNÉES)
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Création de la cible
    df['story_points_grouped'] = df['story_points'].apply(map_to_tshirt_size)

    # 3. Préparation des données : On garde repo_name et toutes vos features
    features_num = ['pre_coding_subtasks', 'pre_coding_desc_length', 
                    'pre_coding_author_tenure_days', 'pre_coding_discussion_participants', 'is_ai_assisted']
    X = df[['repo_name', 'clean_text'] + features_num] 
    
    le = LabelEncoder()
    y = le.fit_transform(df['story_points_grouped'])

    # 4. Pipeline
    # Ajout d'une PCA pour compresser le texte (empêche le modèle de se perdre dans 384 dimensions)
    text_pipeline = SklearnPipeline([
        ('embedder', SemanticTransformer()),
        ('pca', PCA(n_components=30, random_state=42)) 
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('repo', OneHotEncoder(handle_unknown='ignore'), ['repo_name']), # On garde le OneHotEncoder
        ('text', text_pipeline, 'clean_text'),
        ('num', StandardScaler(), features_num)
    ])

    # Ajout d'hyperparamètres de régularisation (freins anti-overfitting)
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=2)),
        ('classifier', XGBClassifier(
            n_estimators=100,             # Réduit (avant: 150)
            learning_rate=0.05,           # Apprentissage plus doux
            max_depth=3,                  # Arbres moins profonds (avant: 4)
            subsample=0.8,                # Ne regarde que 80% des données à la fois
            colsample_bytree=0.8,         # Ne regarde que 80% des colonnes à la fois
            reg_alpha=2.0,                # Pénalité stricte sur les features inutiles
            reg_lambda=5.0,               # Pénalité stricte sur les poids trop forts
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Entraînement du modèle...")
    model_pipeline.fit(X_train, y_train)

    # --- CALCUL DE L'OVERFITTING ---
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    diff = train_score - test_score

    print("\n" + "="*40)
    print("      ANALYSE DE LA PERFORMANCE")
    print("="*40)
    print(f"Précision sur TRAIN (vu) : {train_score:.2%}")
    print(f"Précision sur TEST (neuf) : {test_score:.2%}")
    print(f"Écart (Gap)              : {diff:.2%}")
    
    if diff > 0.10: 
        print("=> Résultat : Risque d'OVERFITTING détecté.")
    elif diff < 0:
        print("=> Résultat : Le modèle généralise exceptionnellement bien.")
    else:
        print("=> Résultat : Modèle équilibré (bonne généralisation).")
    print("="*40)

    # 5. Sauvegarde
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'pipeline': model_pipeline, 'label_encoder': le}, f)
    
    print(f"\n✔ Modèle sauvegardé dans {MODEL_PATH}")

if __name__ == "__main__":
    train_model()