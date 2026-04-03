from github import Github, Auth
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone 
import os
import csv
import time

# 1. CONFIGURATION ET CONNEXION
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
if not token:
    print("Erreur : GITHUB_TOKEN non trouvé dans le fichier .env")
    exit()

auth = Auth.Token(token)
g = Github(auth=auth)

LANGUAGE = "python"
MAX_REPOS = 50 
CSV_FILE = "agile_expert_dataset_v3.csv"

fieldnames = [
    "name", "python_loc", "age_days", "velocity_sprint", 
    "active_contributors", "experience_factor", "avg_lead_time_days", 
    "integration_complexity", "churn_rate", "commit_size_avg",
    "success_rate", "issue_density"
]

def get_data():   
    now = datetime.now(timezone.utc)
    
    # On commence par des projets très populaires pour valider l'extraction
    query = f"language:{LANGUAGE} stars:>10000 fork:false"
    repos = g.search_repositories(query=query, sort="stars", order="desc")
    
    print(f"Recherche lancée... Analyse des dépôts stars > 10000")

    count = 0
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repos:
            if count >= MAX_REPOS: break
            
            try:
                print(f"[{count+1}/{MAX_REPOS}] Analyse : {repo.full_name}...", end=" ", flush=True)

                # 1. TAILLE
                langs = repo.get_languages()
                py_bytes = langs.get("Python", 0)
                python_loc = py_bytes // 40 
                if python_loc < 500: # Seuil abaissé pour le test
                    print("Skip (Trop petit)")
                    continue

                # 2. EQUIPE
                stats_contribs = repo.get_stats_contributors()
                if not stats_contribs:
                    print("Skip (Pas de stats)")
                    continue

                active_devs = 0
                total_commits_active = 0
                six_months_ago = now - timedelta(days=180)
                
                for s in stats_contribs:
                    if s.weeks[-1].w > six_months_ago and s.total > 2:
                        active_devs += 1
                        total_commits_active += s.total
                
                if active_devs == 0: 
                    print("Skip (Inactif)")
                    continue
                
                exp_factor = round(total_commits_active / active_devs, 2)

                # 3. CHURN & VÉLOCITÉ
                three_months_ago = now - timedelta(days=90)
                recent_commits_list = list(repo.get_commits(since=three_months_ago)[:30])
                
                total_mouvement_py = 0
                for c in recent_commits_list:
                    # On évite de scanner trop de fichiers pour économiser l'API
                    for file in c.files[:10]: 
                        if file.filename.endswith('.py'):
                            total_mouvement_py += (file.additions + file.deletions)
                
                num_recent = len(recent_commits_list)
                churn_rate = round(total_mouvement_py / python_loc, 6) if python_loc > 0 else 0
                commit_size_avg = round(total_mouvement_py / num_recent, 2) if num_recent > 0 else 0
                velocity = round((num_recent / 90) * 14, 2)

                # 4. PR & SUCCESS
                all_prs = list(repo.get_pulls(state='all', sort='created', direction='desc')[:20])
                merged_prs, closed_prs_count = 0, 0
                durations, files_changed = [], []

                for pr in all_prs:
                    if pr.state == 'closed':
                        closed_prs_count += 1
                        if pr.merged:
                            merged_prs += 1
                            if pr.closed_at:
                                diff = (pr.closed_at - pr.created_at).days
                                durations.append(max(1, diff))
                                files_changed.append(pr.changed_files)
                
                success_rate = round(merged_prs / closed_prs_count, 2) if closed_prs_count > 0 else 0
                avg_lead = sum(durations)/len(durations) if durations else 0
                comp_score = sum(files_changed)/len(files_changed) if files_changed else 0

                # 5. DENSITÉ BUGS
                total_commits_all = repo.get_commits().totalCount
                issue_density = round(repo.open_issues_count / total_commits_all, 4) if total_commits_all > 0 else 0

                # SAUVEGARDE
                writer.writerow({
                    "name": repo.name,
                    "python_loc": python_loc,
                    "age_days": (now - repo.created_at).days,
                    "velocity_sprint": velocity,
                    "active_contributors": active_devs,
                    "experience_factor": exp_factor,
                    "avg_lead_time_days": round(avg_lead, 2),
                    "integration_complexity": round(comp_score, 2),
                    "churn_rate": churn_rate,
                    "commit_size_avg": commit_size_avg,
                    "success_rate": success_rate,
                    "issue_density": issue_density
                })
                
                f.flush() 
                print(f"OK !")
                count += 1
                time.sleep(1) # Pause respectueuse entre les dépôts

            except Exception as e:
                print(f"Erreur : {e}")
                continue

    print(f"\nTerminé ! {count} dépôts dans {CSV_FILE}")

if __name__ == "__main__":
    get_data()