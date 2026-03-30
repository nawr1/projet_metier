from github import Github, Auth
from dotenv import load_dotenv
import os
import csv
import time

load_dotenv()
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
g = Github(auth=auth)

# ---- SEARCH FILTERS ----
LANGUAGE = "python"
MIN_STARS = 10
MIN_SIZE_KB = 100
MAX_REPOS = 10

print("Searching repositories...")

repos = g.search_repositories(
    query=f"language:{LANGUAGE} stars:>{MIN_STARS} size:>{MIN_SIZE_KB} fork:false"
)

results = []
count = 0

for repo in repos:
    if count >= MAX_REPOS:
        break

    try:
        print(f"Processing ({count+1}): {repo.full_name}")

        # Total number of commits ever made
        commits_count = repo.get_commits().totalCount

        # Number of people who contributed code
        contributors_count = repo.get_contributors().totalCount

        # Programming language
        languages = repo.get_languages()
        total_bytes = sum(int(v) for v in languages.values() if str(v).isdigit() or isinstance(v, int))

        # Number of issues that were reported AND fixed
        closed_issues = repo.get_issues(state="closed", labels=[]).totalCount

        # Number of closed pull requests
        pull_requests = repo.get_pulls(state="closed").totalCount

        # How many days between creation and last update
        age_days = (repo.updated_at - repo.created_at).days

        # Estimated Lines of Code (bytes ÷ 40)
        estimated_loc = total_bytes / 40
        if estimated_loc < 1000:
            print(f"  Skipping {repo.full_name} — too small")
            continue

        results.append({
            "name": repo.full_name,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "language": LANGUAGE,
            "size_kb": repo.size,
            "estimated_loc": int(estimated_loc),
            "commits_count": commits_count,
            "contributors_count": contributors_count,
            "closed_issues": closed_issues,
            "pull_requests": pull_requests,
            "age_days": age_days,
        })

        count += 1
        time.sleep(1)

    except Exception as e:
        print(f"  Error on {repo.full_name}: {e}")
        continue

# ---- SAVE TO CSV ----
if results:
    csv_file = "repos_dataset.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n{len(results)} repos saved to {csv_file}")
else:
    print("\n No repos collected. Try adjusting filters.")