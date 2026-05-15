import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

CSV_FILE = "agile_pr_level.csv"

# --- FONCTION DE REGROUPEMENT (T-SHIRT SIZING) ---
def map_to_tshirt_size(sp):
    if sp <= 2:
        return 0  # SMALL (1-2)
    elif sp <= 5:
        return 1  # MEDIUM (3-5)
    else:
        return 2  # LARGE (8+)

class SemanticTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
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
        
        # --- ETAPE CRUCIALE : REGROUPEMENT ---
        df['story_points_grouped'] = df['story_points'].apply(map_to_tshirt_size)
        
        # On s'assure d'avoir assez d'exemples par groupe
        df = df[df.groupby('story_points_grouped')['story_points_grouped'].transform('count') >= 5]
    except Exception as e:
        print(f"Erreur : {e}")
        return

    print(f"Nouvelle distribution (S/M/L) : \n{df['story_points_grouped'].value_counts()}")

    df['clean_text'] = (df['pre_coding_title'].fillna('') + " " + 
                        df['pre_coding_description'].apply(clean_description))

    # Features incluant le repo_name
    X = df[['repo_name', 'clean_text', 'pre_coding_subtasks', 'pre_coding_desc_length', 
            'pre_coding_author_tenure_days', 'pre_coding_discussion_participants', 'is_ai_assisted']]
    
    y = df['story_points_grouped']
    # On garde les noms originaux pour le rapport final
    target_names = ["SMALL (1-2)", "MEDIUM (3-5)", "LARGE (8+)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('repo', OneHotEncoder(handle_unknown='ignore'), ['repo_name']),
            ('text', SemanticTransformer(), 'clean_text'),
            ('num', StandardScaler(), [
                'pre_coding_subtasks', 'pre_coding_desc_length', 
                'pre_coding_author_tenure_days', 'pre_coding_discussion_participants', 'is_ai_assisted'
            ])
        ]
    )

    # Paramètres optimisés pour un petit dataset et 3 classes
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=2)),
        ('classifier', XGBClassifier(
            n_estimators=150,
            learning_rate=0.03, # Apprentissage plus lent
            max_depth=4,        # Arbres moins profonds pour éviter le sur-apprentissage
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entraînement en cours...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    
    print("\n Report")
    print(f"Précision T-Shirt Sizing : {accuracy_score(y_test, y_pred):.2%}")
    # On ajuste les target_names selon les classes réellement présentes dans y_test
    present_classes = [target_names[i] for i in sorted(y_test.unique())]
    print(classification_report(y_test, y_pred, target_names=present_classes))

    # Sauvegarde du nouveau modèle
    with open('model_tshirt_v1.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

if __name__ == "__main__":
    train_model()