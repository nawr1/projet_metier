from github import Github, Auth
from dotenv import load_dotenv
from radon.complexity import cc_visit # Pour calculer la complexité du code
import requests
import os
import csv
import time

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
g = Github(auth=Auth.Token(token))

LANGUAGE = "python"
MAX_REPOS = 2          # 🔹 PETIT TEST : Seulement 2 dépôts
MAX_ISSUES_PER_REPO = 5 # 🔹 PETIT TEST : 5 tickets par dépôt
CSV_FILE = "ultimate_agile_dataset.csv"

fieldnames = [
    "issue_id", "title", "description", 
    "dev_experience",        # Combien de commits a le dev ?
    "code_complexity",       # Note McCabe (Radon) du code modifié
    "resolution_hours"       # Cible (Y)
]

def analyze_complexity_from_pr(repo, pull_request_number):
    """Télécharge le code de la PR et utilise Radon pour calculer la complexité"""
    try:
        pr = repo.get_pull(pull_request_number)
        files = pr.get_files()
        total_complexity = 0
        file_count = 0
        
        for file in files:
            if file.filename.endswith(".py"): # On n'analyse que le code Python
                raw_code = requests.get(file.raw_url).text
                blocks = cc_visit(raw_code) # Utilisation de RADON
                if blocks:
                    total_complexity += sum([b.complexity for b in blocks]) / len(blocks)
                    file_count += 1
        
        return round(total_complexity / file_count, 2) if file_count > 0 else 1.0
    except:
        return 1.0 # Complexité par défaut si erreur

def get_data():
    repos = g.search_repositories(query=f"language:{LANGUAGE} stars:>5000 fork:false")
    
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo_count, repo in enumerate(repos):
            if repo_count >= MAX_REPOS: break
            print(f"\n Analyse du Dépôt : {repo.full_name}")
            
            issues = repo.get_issues(state="closed")
            issue_count = 0
            
            for issue in issues:
                if issue_count >= MAX_ISSUES_PER_REPO: break
                
                # Chercher une issue qui a été fermée avec une Pull Request
                if not issue.body or not issue.closed_at or not issue.pull_request:
                    continue
                
                print(f"   Extraction Ticket #{issue.number} + Scan Qualité du Code...")
                try:
                    # 1. Cible (Temps)
                    resolution_hours = round((issue.closed_at - issue.created_at).total_seconds() / 3600, 2)
                    
                    # 2. Expérience du Dev
                    dev_commits = repo.get_commits(author=issue.user).totalCount if issue.user else 1
                    
                    # 3. Complexité McCabe (RADON) - Extraction via l'URL de la PR associée
                    pr_number = int(issue.pull_request.url.split('/')[-1])
                    code_complexity = analyze_complexity_from_pr(repo, pr_number)

                    # 4. Nettoyage Texte
                    title = issue.title.replace('\n', ' ')
                    description = issue.body.replace('\n', ' ')[:500]

                    # 5. Sauvegarde
                    writer.writerow({
                        "issue_id": issue.number,
                        "title": title,
                        "description": description,
                        "dev_experience": min(dev_commits, 100), # On cap à 100 max
                        "code_complexity": code_complexity,
                        "resolution_hours": resolution_hours
                    })
                    issue_count += 1
                    time.sleep(1) # Ne pas se faire bloquer par GitHub
                except Exception as e:
                    continue

    print("\n Scraping Terminé !")

if __name__ == "__main__":
    get_data()