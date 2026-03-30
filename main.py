from github import Github, Auth
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

auth = Auth.Token(token)
g = Github(auth = auth)
user = g.get_user()
print("Connected as:", user.login)
