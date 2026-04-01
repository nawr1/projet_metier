from github import Github, Auth
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone 
import os
import csv
import time

# 1. CHARGEMENT
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

auth = Auth.Token(token)
g = Github(auth=auth)

LANGUAGE = "python"
MAX_REPOS = 10
CSV_FILE = "agile_estimation_dataset.csv"

fieldnames = [
    "name", "stars", "estimated_loc", "age_days",
    "total_commits", "velocity_sprint", 
    "active_contributors", "avg_lead_time_days", 
    "integration_complexity", "bug_ratio"
]

def get_data():
    
    now = datetime.now(timezone.utc)
    try:
        user = g.get_user()
        print(f"Connecté en tant que : {user.login}")
    except Exception as e:
        print(f"Échec de connexion : {e}")
        return

    # Recherche (on filtre les gros dépôts pour éviter les timeouts)
    repos = g.search_repositories(query=f"language:{LANGUAGE} stars:>1000 fork:false")
    count = 0
    
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repos:
            if count >= MAX_REPOS: break
            try:
                print(f"Analyse ({count+1}/{MAX_REPOS}) : {repo.full_name}...")

                # --- 1. TAILLE & ÂGE ---
                # Correction : now est aware, repo.created_at est aware
                age_days = (now - repo.created_at).days
                loc_est = repo.size * 50 

                # --- 2. EFFECTIF RÉEL (Agile) ---
                stats = repo.get_stats_contributors()
                active_devs = 0
                six_months_ago = now - timedelta(days=180)
                
                if stats:
                    for s in stats:
                        # s.weeks[-1].w est déjà un objet datetime aware dans PyGithub
                        last_commit_date = s.weeks[-1].w
                        if last_commit_date > six_months_ago and s.total > 5:
                            active_devs += 1
                
                if active_devs == 0:
                    print(f"Skip {repo.name} (Pas d'activité récente)")
                    continue

                # --- 3. VÉLOCITÉ ---
                three_months_ago = now - timedelta(days=90)
                recent_commits = repo.get_commits(since=three_months_ago).totalCount
                velocity = round((recent_commits / 90) * 14, 2)

                # --- 4. LEAD TIME & COMPLEXITÉ ---
                closed_prs = list(repo.get_pulls(state='closed', sort='updated')[:10])
                durations = []
                files_changed = []
                for pr in closed_prs:
                    if pr.closed_at:
                        # Les deux sont aware, donc la soustraction marche
                        diff = (pr.closed_at - pr.created_at).days
                        durations.append(max(1, diff))
                        files_changed.append(pr.changed_files)
                
                avg_lead = sum(durations)/len(durations) if durations else 0
                comp_score = sum(files_changed)/len(files_changed) if files_changed else 0

                # --- 5. QUALITÉ ---
                total_open = repo.open_issues_count
                bug_ratio = 0.15 if total_open > 10 else 0.05 

                # --- SAUVEGARDE ---
                writer.writerow({
                    "name": repo.name,
                    "stars": repo.stargazers_count,
                    "estimated_loc": loc_est,
                    "age_days": age_days,
                    "total_commits": repo.get_commits().totalCount,
                    "velocity_sprint": velocity,
                    "active_contributors": active_devs,
                    "avg_lead_time_days": round(avg_lead, 2),
                    "integration_complexity": round(comp_score, 2),
                    "bug_ratio": bug_ratio
                })
                
                print(f"OK.")
                count += 1
                time.sleep(1) 

            except Exception as e:
                print(f"Erreur : {e}")
                continue

if __name__ == "__main__":
    get_data()