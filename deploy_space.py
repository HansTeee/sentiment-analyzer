"""
deploy_space.py - upload this app to a Hugging Face Space in ONE commit.

WHY a script instead of `git push`?
This repo has two READMEs: README.md (the full GitHub version) and
space/README.md (the short Space version with the YAML config header).
The Space needs space/README.md to end up at its root as README.md.
This script uploads exactly the files the Space needs, renaming that one,
in a single atomic commit, so the Space builds once with the right config.

Usage (log in once first; your token stays in the HF cache, never in code):
    hf auth login
    python deploy_space.py <your-hf-username>/<space-name>
"""

import sys

from huggingface_hub import CommitOperationAdd, HfApi

# (file in this repo, path it should have in the Space)
FILES = [
    ("space/README.md", "README.md"),
    ("app.py", "app.py"),
    ("analyzer.py", "analyzer.py"),
    ("requirements.txt", "requirements.txt"),
    (".gitattributes", ".gitattributes"),
]

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    sys.exit("Usage: python deploy_space.py <your-hf-username>/<space-name>")

space_id = sys.argv[1]
info = HfApi().create_commit(
    repo_id=space_id,
    repo_type="space",
    operations=[CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in FILES],
    commit_message="Deploy sentiment + emotion analyzer",
)
print(f"Uploaded {len(FILES)} files: {info.commit_url}")
print(f"Watch the build logs at: https://huggingface.co/spaces/{space_id}?logs=build")
