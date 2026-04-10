from github import Github, Auth
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import os, csv, time

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
g = Github(auth=Auth.Token(token), per_page=100)  # max par page

LANGUAGE = "python"
MAX_REPOS = 10
MAX_PRS_PER_REPO = 30  # réduit de 50 → 30
CSV_FILE = "agile_estimation_dataset.csv"

fieldnames = [
    "repo_name", "language", "pr_number", "era",
    "code_churn", "files_changed", "review_cycles",
    "pr_comments", "refactoring_ratio", "commit_velocity",
    "velocity_sprint", "active_contributors", "age_days", "stars",
    "story_points"
]

def get_era(dt):
    if dt.year <= 2021: return "pre_AI"
    elif dt.year >= 2023: return "post_AI"
    return None

def compute_story_points(review_cycles, pr_comments, files_changed):
    score = (review_cycles * 3) + (pr_comments * 0.5) + (files_changed * 1.5)
    if score < 3:    return 1
    elif score < 8:  return 2
    elif score < 15: return 3
    elif score < 25: return 5
    else:            return 8

def get_data():
    now = datetime.now(timezone.utc)
    six_months_ago = now - timedelta(days=180)
    three_months_ago = now - timedelta(days=90)

    try:
        user = g.get_user()
        print(f"Connecté : {user.login}\n")
    except Exception as e:
        print(f"Échec connexion : {e}"); return

    repos = g.search_repositories(
        query=f"language:{LANGUAGE} stars:>1000 fork:false"
    )

    repo_count = 0
    total_rows = 0

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repos:
            if repo_count >= MAX_REPOS: break
            try:
                print(f"{'='*55}")
                print(f"Repo {repo_count+1}/{MAX_REPOS} : {repo.full_name}")

                # ── Contexte repo (calculs légers, pas d'appel lourd) ──
                age_days = (now - repo.created_at).days
                stars = repo.stargazers_count

                # active_contributors : on utilise get_stats_contributors()
                # qui est mis en cache par GitHub → rapide
                active_contributors = 0
                try:
                    stats = repo.get_stats_contributors()
                    if stats:
                        for s in stats:
                            if s.weeks[-1].w > six_months_ago and s.total > 5:
                                active_contributors += 1
                except:
                    active_contributors = 1  # fallback

                if active_contributors == 0:
                    print(f"  Skip : aucune activité récente\n")
                    continue

                # velocity_sprint : 1 seul appel pour tout le repo
                try:
                    recent_count = repo.get_commits(since=three_months_ago).totalCount
                    velocity_sprint = round((recent_count / 90) * 14, 2)
                except:
                    velocity_sprint = 0.0

                # ── PRs : on filtre AVANT de faire les appels lourds ──
                closed_prs = repo.get_pulls(
                    state='closed', sort='updated', direction='desc'
                )

                pr_count = 0
                checked = 0  # compteur pour ne pas boucler indéfiniment

                for pr in closed_prs:
                    if pr_count >= MAX_PRS_PER_REPO: break
                    if checked >= 80: break  # on ne regarde pas + de 80 PRs
                    checked += 1

                    # Filtre rapide SANS appel API supplémentaire
                    if not pr.merged_at: continue
                    era = get_era(pr.created_at)
                    if era is None: continue

                    # pr.additions, pr.deletions, pr.changed_files
                    # pr.comments, pr.review_comments
                    # → tous disponibles dans l'objet PR déjà chargé, PAS d'appel API
                    try:
                        code_churn    = pr.additions + pr.deletions
                        files_changed = pr.changed_files
                        pr_comments   = pr.comments + pr.review_comments

                        # review_cycles : 1 appel API par PR — on le garde
                        # mais on limite avec [:5] pour ne lire que les 5 premières reviews
                        review_cycles = 0
                        try:
                            for r in pr.get_reviews():
                                if r.state == "CHANGES_REQUESTED":
                                    review_cycles += 1
                        except:
                            review_cycles = 0

                        # refactoring_ratio : appel lourd (get_commits)
                        # → on l'approxime avec pr.commits (juste le COUNT, pas les objets)
                        # Si > 1 commit on estime un ratio fixe par tranche
                        n_commits = pr.commits  # attribut direct, PAS d'appel API
                        if n_commits == 1:
                            refactoring_ratio = 0.0
                        elif n_commits <= 3:
                            refactoring_ratio = 0.3
                        elif n_commits <= 7:
                            refactoring_ratio = 0.6
                        else:
                            refactoring_ratio = 0.85

                        # commit_velocity
                        pr_duration_hours = max(
                            (pr.merged_at - pr.created_at).total_seconds() / 3600, 1
                        )
                        commit_velocity = round(code_churn / pr_duration_hours, 2)

                        story_points = compute_story_points(
                            review_cycles, pr_comments, files_changed
                        )

                        writer.writerow({
                            "repo_name":           repo.name,
                            "language":            repo.language,
                            "pr_number":           pr.number,
                            "era":                 era,
                            "code_churn":          code_churn,
                            "files_changed":       files_changed,
                            "review_cycles":       review_cycles,
                            "pr_comments":         pr_comments,
                            "refactoring_ratio":   refactoring_ratio,
                            "commit_velocity":     commit_velocity,
                            "velocity_sprint":     velocity_sprint,
                            "active_contributors": active_contributors,
                            "age_days":            age_days,
                            "stars":               stars,
                            "story_points":        story_points
                        })

                        pr_count += 1
                        total_rows += 1
                        print(
                            f"  PR #{pr.number:5d} | {era:8s} | "
                            f"churn:{code_churn:5d} | "
                            f"cycles:{review_cycles} | "
                            f"sp:{story_points}"
                        )

                    except Exception as e:
                        print(f"  Erreur PR #{pr.number} : {e}")
                        continue

                    # Pas de sleep ici — les appels sont déjà légers
                    # On garde juste un micro-délai pour éviter le rate limit
                    time.sleep(0.1)

                print(f"  → {pr_count} PRs | {repo.name}")
                repo_count += 1
                time.sleep(1)  # réduit de 2s → 1s entre repos

            except Exception as e:
                print(f"Erreur repo {repo.full_name} : {e}")
                continue

    print(f"\n{'='*55}")
    print(f"TERMINÉ : {repo_count} repos | {total_rows} PRs → {CSV_FILE}")

if __name__ == "__main__":
    get_data()