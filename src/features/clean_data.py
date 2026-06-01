import pandas as pd
import re
import os

# --- CONFIGURATION DES CHEMINS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CURRENT_DIR if not CURRENT_DIR.endswith('src') else os.path.dirname(CURRENT_DIR)

RAW_FILE = os.path.join(ROOT_DIR, "data", "raw", "agile_pr_level.csv")
CLEAN_FILE = os.path.join(ROOT_DIR, "data", "processed", "cleaned_agile_pr_level.csv")

def clean_description_logic(text):
    if pd.isna(text) or text == "" or text is None: 
        return ""
    
    text = str(text).lower()
    
    # 1. Supprimer les URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Supprimer les blocs [!IMPORTANT], [NOTE], etc.
    text = re.sub(r'\[\s*!?\s*(important|note|caution|warning|danger|tip)\s*\]', ' ', text, flags=re.IGNORECASE)
    
    # 3. Supprimer le texte répétitif des guidelines
    text = re.sub(r'make sure you have read our contribution guidelines.*?(?=\d\.|\s|$)', ' ', text)

    # 4. Supprimer TOUS les symboles Markdown '#' (les titres) partout
    text = re.sub(r'#+', ' ', text)
    
    # 5. Supprimer les chevrons '>' et les listes type '> > 1.'
    text = re.sub(r'>+', ' ', text)
    text = re.sub(r'\d+\.\s*', ' ', text)
    
    # 6. Supprimer les préfixes techniques 'fix(api):', 'feat:', etc.
    text = re.sub(r'\w+(\([\w\-]+\))?:\s*', ' ', text)
    
    # 7. Supprimer les cases à cocher '- [ ]' ou '- [x]'
    text = re.sub(r'-\s*\[\s*[x\s]\s*\]', ' ', text)
    
    # 8. Nettoyage final des symboles restants
    text = re.sub(r'[\[\]\(\)\{\}\*_\|`\~\+\=\!]', ' ', text)
    
    # 9. Supprimer les doubles espaces et retours à la ligne
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_standalone_cleaning():
    print(f"--- DÉBUT DU NETTOYAGE ---")
    
    if not os.path.exists(RAW_FILE):
        print(f"Erreur : Le fichier source est introuvable ici : {RAW_FILE}")
        return

    # 1. Charger les données
    print(f"Chargement de : {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE)

    # 2. Nettoyer les colonnes EXISTANTES directement
    print("Nettoyage des colonnes pre_coding_title et pre_coding_description...")
    
    df['pre_coding_title'] = df['pre_coding_title'].fillna('').apply(clean_description_logic)
    df['pre_coding_description'] = df['pre_coding_description'].fillna('').apply(clean_description_logic)

    # 3. Créer aussi la colonne combinée 'clean_text' (utile pour tes futurs modèles)
    df['clean_text'] = (df['pre_coding_title'] + " " + df['pre_coding_description']).str.strip()

    # 4. Sauvegarder le résultat
    os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
    
    # Note : encoding='utf-8-sig' permet à Excel d'ouvrir le fichier correctement sans bugs d'affichage
    df.to_csv(CLEAN_FILE, index=False, encoding='utf-8-sig')
    
    print(f"--- TERMINÉ ---")
    print(f"Fichier créé : {CLEAN_FILE}")
    print(f"Nombre de lignes traitées : {len(df)}")
    
    # Exemples de vérification
    print("\n--- VÉRIFICATION DES COLONNES NETTOYÉES ---")
    print(f"Titre 1 : {df['pre_coding_title'].iloc[0]}")
    print(f"Description 1 : {df['pre_coding_description'].iloc[0][:100]}...")

if __name__ == "__main__":
    run_standalone_cleaning()