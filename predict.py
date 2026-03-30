import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from github import Github, Auth
from dotenv import load_dotenv
import os

load_dotenv()
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
g = Github(auth=auth)

# =============================
# 1. RETRAIN MODEL ON FULL DATA
# =============================
df = pd.read_csv("repos_dataset.csv")
df = df.drop(columns=["name", "language"]).dropna()

X = df.drop(columns=["commits_count"])
y = df["commits_count"]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
print("✅ Model retrained on full dataset")

# =============================
# 2. FETCH NEW REPO
# =============================
repo_name = input("\n🔍 Enter repo name (e.g. facebook/react): ").strip()

try:
    repo = g.get_repo(repo_name)
    print(f"\n📦 Fetching data for: {repo.full_name}")

    # Fetch features
    languages = repo.get_languages()
    total_bytes = sum(int(v) for v in languages.values() if isinstance(v, int))
    estimated_loc = total_bytes / 40

    contributors_count = repo.get_contributors().totalCount
    closed_issues = repo.get_issues(state="closed", labels=[]).totalCount
    pull_requests = repo.get_pulls(state="closed").totalCount
    age_days = (repo.updated_at - repo.created_at).days

    new_repo = {
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "size_kb": repo.size,
        "estimated_loc": int(estimated_loc),
        "contributors_count": contributors_count,
        "closed_issues": closed_issues,
        "pull_requests": pull_requests,
        "age_days": age_days,
    }

    print("\n📊 Repo Features:")
    for key, value in new_repo.items():
        print(f"   {key}: {value:,}")

    # =============================
    # 3. PREDICT
    # =============================
    input_df = pd.DataFrame([new_repo])
    predicted_commits = rf.predict(input_df)[0]

    # Get real commits to compare
    real_commits = repo.get_commits().totalCount

    print("\n" + "="*40)
    print("🎯 PREDICTION RESULTS")
    print("="*40)
    print(f"   Predicted commits (effort) : {predicted_commits:,.0f}")
    print(f"   Real commits               : {real_commits:,}")
    diff = abs(predicted_commits - real_commits)
    accuracy = 100 - (diff / real_commits * 100)
    print(f"   Difference                 : {diff:,.0f} commits")
    print(f"   Accuracy                   : {accuracy:.1f}%")

    if accuracy > 80:
        print("\n✅ Great prediction!")
    elif accuracy > 60:
        print("\n⚠️ Decent prediction — more training data would help")
    else:
        print("\n❌ Weak prediction — we need more repos in training data (try 100)")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the repo name is correct (e.g. facebook/react)")