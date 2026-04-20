import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

REPOS = [
    "facebook/react",
    "microsoft/vscode",
    "django/django",
    "scikit-learn/scikit-learn",
    "expressjs/express"
]

PRE_AI_START  = "2019-01-01"
PRE_AI_END    = "2021-12-31"
POST_AI_START = "2023-01-01"
POST_AI_END   = "2025-12-31"

MAX_PRS_PER_REPO     = 100
SLEEP_BETWEEN_REQUESTS = 1

FEATURE_COLS = [
    "churn", "additions", "deletions", "nb_files",
    "complexity_delta", "revision_cycles", "nb_comments",
    "nb_commits", "velocity", "refactoring_ratio"
]

TARGET_COL = "effort_proxy"