import time
import requests
import lizard
import pandas as pd
from datetime import datetime
from config import (HEADERS, REPOS, PRE_AI_START, PRE_AI_END,
                    POST_AI_START, POST_AI_END,
                    MAX_PRS_PER_REPO, SLEEP_BETWEEN_REQUESTS)


# ==============================================================================
# ÉTAPE 0 — FILTRAGE : SIGNAUX AUTOMATIQUES D'USAGE IA
# ==============================================================================

def check_repo_age(owner, repo):
    url  = f"https://api.github.com/repos/{owner}/{repo}"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if resp.status_code != 200:
        return False
    created_year = int(resp.json().get("created_at", "9999")[:4])
    ok = created_year <= 2020
    if ok:
        print(f"    [OK]   Créé en {created_year} — comparaison pre/post valide")
    else:
        print(f"    [SKIP] Créé en {created_year} — trop récent")
    return ok


def check_copilot_coauthored_commits(owner, repo, min_occurrences=3):
    """
    Signal 1 — Co-authored-by: GitHub Copilot dans les commits.
    Généré automatiquement par VS Code / JetBrains, signal 100% machine.
    """
    url     = "https://api.github.com/search/commits"
    headers = {**HEADERS, "Accept": "application/vnd.github.cloak-preview+json"}
    params  = {
        "q":        f"repo:{owner}/{repo} author-date:>2023-01-01 Co-authored-by: GitHub Copilot",
        "per_page": 10
    }
    resp = requests.get(url, headers=headers, params=params)
    time.sleep(SLEEP_BETWEEN_REQUESTS * 2)
    if resp.status_code != 200:
        return False
    total = resp.json().get("total_count", 0)
    if total >= min_occurrences:
        print(f"    [OK]   Signal 1 : {total} commits avec 'Co-authored-by: GitHub Copilot'")
        return True
    return False


def check_copilot_bot_reviews(owner, repo, max_prs_to_check=20):
    """
    Signal 2 — Bot copilot-pull-request-reviewer[bot] dans les PR reviews.
    Bot officiel GitHub Copilot Code Review, identifiant fixe.
    """
    url    = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": max_prs_to_check}
    resp   = requests.get(url, headers=HEADERS, params=params)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if resp.status_code != 200:
        return False
    for pr in resp.json():
        merged_at = pr.get("merged_at", "") or ""
        if not merged_at or merged_at[:4] < "2023":
            continue
        rev_url  = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr['number']}/reviews"
        rev_resp = requests.get(rev_url, headers=HEADERS)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if rev_resp.status_code != 200:
            continue
        for review in rev_resp.json():
            login = (review.get("user") or {}).get("login", "")
            if "copilot-pull-request-reviewer" in login:
                print(f"    [OK]   Signal 2 : bot copilot-pull-request-reviewer trouvé (PR #{pr['number']})")
                return True
    return False


def check_copilot_instructions_file(owner, repo):
    """
    Signal 3 — Fichier .github/copilot-instructions.md présent.
    Fichier de configuration officiel Copilot — adoption institutionnelle.
    """
    url  = f"https://api.github.com/repos/{owner}/{repo}/contents/.github/copilot-instructions.md"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if resp.status_code == 200:
        print(f"    [OK]   Signal 3 : .github/copilot-instructions.md présent")
        return True
    return False


def check_copilot_in_workflows(owner, repo):
    """
    Signal 4 — Action Copilot officielle dans les workflows GitHub Actions.
    Intégration Copilot dans le pipeline CI/CD de l'équipe.
    """
    url  = f"https://api.github.com/repos/{owner}/{repo}/contents/.github/workflows"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if resp.status_code != 200:
        return False
    copilot_actions = ["github/copilot", "copilot-workspace", "actions/ai-inference"]
    for wf in resp.json():
        if not wf.get("name", "").endswith((".yml", ".yaml")):
            continue
        content_resp = requests.get(wf["download_url"])
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if content_resp.status_code != 200:
            continue
        content = content_resp.text.lower()
        for action in copilot_actions:
            if action in content:
                print(f"    [OK]   Signal 4 : action '{action}' dans {wf['name']}")
                return True
    return False


def filter_repos_with_ai_usage(repo_list):
    validated = []
    print("\n" + "=" * 60)
    print("FILTRAGE — Signaux automatiques d'usage IA")
    print("=" * 60)
    for repo_full in repo_list:
        owner, repo = repo_full.split("/")
        print(f"\n  [{repo_full}]")
        if not check_repo_age(owner, repo):
            print(f"  → REJETÉ (trop récent)\n")
            continue
        found = False
        for check_fn in [
            check_copilot_coauthored_commits,
            check_copilot_bot_reviews,
            check_copilot_instructions_file,
            check_copilot_in_workflows,
        ]:
            if check_fn(owner, repo):
                found = True
                break
        if found:
            print(f"  → VALIDÉ — usage IA confirmé")
            validated.append(repo_full)
        else:
            print(f"  → REJETÉ — aucun signal automatique d'IA trouvé")

    print(f"\n{'=' * 60}")
    print(f"Résultat : {len(validated)}/{len(repo_list)} repos validés")
    for r in validated:
        print(f"  - {r}")
    print("=" * 60 + "\n")
    return validated


# ==============================================================================
# COLLECTE DES PRs — PAGINATION INTELLIGENTE
#
# PROBLÈME PRÉCÉDENT :
#   La fonction paginait TOUTES les PRs triées par "updated" sans s'arrêter.
#   Pour vscode (10 000+ PRs), elle pouvait tourner des heures.
#
# SOLUTION :
#   1. Trier par "created" desc — les PRs créées dans la période arrivent
#      en premier, on peut s'arrêter dès qu'on sort de la fenêtre.
#   2. Arrêt immédiat si la date de création est avant `since` — inutile
#      de continuer, toutes les suivantes seront encore plus anciennes.
#   3. Afficher la progression toutes les 10 PRs collectées.
# ==============================================================================

def get_pulls(owner, repo, since, until, max_prs=MAX_PRS_PER_REPO):
    """
    Récupère les PRs mergées dans la fenêtre [since, until].

    Stratégie d'arrêt rapide :
      - Tri par created desc → les plus récentes arrivent en premier
      - Si created_at < since → toutes les suivantes sont encore plus
        anciennes → on sort de la boucle immédiatement
      - On s'arrête aussi dès que max_prs est atteint
    """
    pulls     = []
    page      = 1
    stop_loop = False

    print(f"    Collecte en cours", end="", flush=True)

    while len(pulls) < max_prs and not stop_loop:
        url    = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {
            "state":     "closed",
            "sort":      "created",   # Tri par date de création
            "direction": "desc",      # Les plus récentes en premier
            "per_page":  50,          # 50 par page pour moins d'appels
            "page":      page
        }
        resp = requests.get(url, headers=HEADERS, params=params)

        if resp.status_code == 403:
            print("\n    Rate limit. Pause 60s...")
            time.sleep(60)
            continue

        if resp.status_code != 200:
            print(f"\n    Erreur API : {resp.status_code}")
            break

        data = resp.json()
        if not data:
            break  # Plus de pages

        for pr in data:
            created = pr.get("created_at", "")[:10]
            merged  = pr.get("merged_at",  "") or ""

            # Arrêt immédiat : toutes les PRs suivantes sont plus anciennes
            if created < since:
                stop_loop = True
                break

            # Ignorer les PRs non mergées ou hors fenêtre
            if not merged or merged[:10] > until or merged[:10] < since:
                continue

            pulls.append(pr)

            if len(pulls) % 10 == 0:
                print(f" {len(pulls)}", end="", flush=True)

            if len(pulls) >= max_prs:
                stop_loop = True
                break

        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f" → {len(pulls)} PRs trouvées")
    return pulls


# ==============================================================================
# APPELS API — DÉTAILS D'UNE PR
# ==============================================================================

def get_pr_files(owner, repo, pr_number):
    url  = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp.json() if resp.status_code == 200 else []


def get_pr_reviews(owner, repo, pr_number):
    url  = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp.json() if resp.status_code == 200 else []


def get_pr_comments(owner, repo, pr_number):
    url  = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp.json() if resp.status_code == 200 else []


def get_pr_commits(owner, repo, pr_number):
    url  = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    resp = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp.json() if resp.status_code == 200 else []


# ==============================================================================
# EXTRACTION DES FEATURES
# ==============================================================================

def compute_cyclomatic_complexity(files_data):
    total, count = 0, 0
    for f in files_data:
        filename = f.get("filename", "")
        patch    = f.get("patch", "")
        if not any(filename.endswith(e) for e in [".py", ".js", ".ts", ".java", ".c", ".cpp"]):
            continue
        if not patch:
            continue
        added = "\n".join(
            l[1:] for l in patch.split("\n")
            if l.startswith("+") and not l.startswith("+++")
        )
        if not added.strip():
            continue
        try:
            analysis = lizard.analyze_file.analyze_source_code(filename, added)
            for func in analysis.function_list:
                total += func.cyclomatic_complexity
                count += 1
        except Exception:
            continue
    return total / max(count, 1)


def extract_features(pr, owner, repo):
    pr_number       = pr["number"]
    created_at      = datetime.fromisoformat(pr["created_at"].replace("Z", ""))
    merged_at       = datetime.fromisoformat(pr["merged_at"].replace("Z", ""))
    lead_time_hours = (merged_at - created_at).total_seconds() / 3600

    files            = get_pr_files(owner, repo, pr_number)
    additions        = sum(f.get("additions", 0) for f in files)
    deletions        = sum(f.get("deletions", 0) for f in files)
    churn            = additions + deletions
    nb_files         = len(files)
    complexity_delta = compute_cyclomatic_complexity(files)

    reviews         = get_pr_reviews(owner, repo, pr_number)
    revision_cycles = sum(1 for r in reviews if r.get("state") == "CHANGES_REQUESTED")

    comments    = get_pr_comments(owner, repo, pr_number)
    nb_comments = len(comments)

    commits    = get_pr_commits(owner, repo, pr_number)
    nb_commits = len(commits)

    velocity, refactoring_ratio = 0, 0
    if nb_commits >= 2:
        t0 = datetime.fromisoformat(commits[0]["commit"]["author"]["date"].replace("Z", ""))
        t1 = datetime.fromisoformat(commits[-1]["commit"]["author"]["date"].replace("Z", ""))
        duration          = (t1 - t0).total_seconds() / 3600
        velocity          = churn / max(duration, 0.1)
        refactoring_ratio = (nb_commits - 1) / nb_commits

    effort_proxy = (
        min(revision_cycles, 10) / 10 * 40 +
        min(nb_comments, 50)     / 50 * 30 +
        min(nb_files, 20)        / 20 * 30
    )

    return {
        "repo":              f"{owner}/{repo}",
        "pr_number":         pr_number,
        "merged_at":         pr["merged_at"][:10],
        "churn":             churn,
        "additions":         additions,
        "deletions":         deletions,
        "nb_files":          nb_files,
        "complexity_delta":  complexity_delta,
        "revision_cycles":   revision_cycles,
        "nb_comments":       nb_comments,
        "nb_commits":        nb_commits,
        "velocity":          round(velocity, 2),
        "refactoring_ratio": round(refactoring_ratio, 3),
        "effort_proxy":      round(effort_proxy, 2),
        "lead_time_hours":   round(lead_time_hours, 2),
    }


# ==============================================================================
# COLLECTE COMPLÈTE
# ==============================================================================

def collect_all_data():
    validated_repos = filter_repos_with_ai_usage(REPOS)

    if not validated_repos:
        print("Aucun repo validé. Ajoutez plus de candidats dans REPOS (config.py).")
        return

    with open("validated_repos.txt", "w") as f:
        f.write("\n".join(validated_repos))
    print(f"Repos validés sauvegardés dans validated_repos.txt\n")

    for period_name, since, until in [
        ("pre_ai",  PRE_AI_START,  PRE_AI_END),
        ("post_ai", POST_AI_START, POST_AI_END),
    ]:
        all_features = []
        print(f"=== Collecte : {period_name} ({since} → {until}) ===")

        for repo_full in validated_repos:
            owner, repo = repo_full.split("/")
            print(f"\n  Repo : {owner}/{repo}")

            pulls = get_pulls(owner, repo, since, until)

            for i, pr in enumerate(pulls):
                try:
                    feat           = extract_features(pr, owner, repo)
                    feat["period"] = period_name
                    all_features.append(feat)
                    print(f"    [{i+1}/{len(pulls)}] PR #{pr['number']} extraite")
                except Exception as e:
                    print(f"    Erreur PR #{pr['number']}: {e}")

        df = pd.DataFrame(all_features)
        df.to_csv(f"data_{period_name}.csv", index=False)
        print(f"\nSauvegardé : data_{period_name}.csv ({len(df)} lignes)\n")