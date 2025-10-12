#!/usr/bin/env python3
import os
import subprocess
import tempfile
import sys
import traceback
from openai import OpenAI
from github import Github, Auth  # newer PyGithub versions prefer auth=…


def run_shell(cmd: str, check: bool = True) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and proc.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}", file=sys.stderr)
        print(f"stdout: {proc.stdout}", file=sys.stderr)
        print(f"stderr: {proc.stderr}", file=sys.stderr)
        sys.exit(1)
    return proc.stdout.strip()


def codex_suggest_patch(title: str, body: str) -> str:
    print(f"[INFO] Requesting patch. Title: {title!r}, body length: {len(body)}")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    instructions = (
        "You are an expert code assistant. "
        "Given an issue title and body, and full repository context, "
        "generate a minimal patch (unified diff) that fixes the issue. "
        "Only modify what's necessary; do not refactor unrelated code."
    )
    user_input = f"Issue title: {title}\nIssue body: {body}\nGenerate patch diff:"
    try:
        resp = client.responses.create(
            model="gpt-4o", instructions=instructions, input=user_input
        )
        patch = resp.output_text or ""
    except Exception as e:
        print("[ERROR] OpenAI API error:", file=sys.stderr)
        traceback.print_exc()
        patch = ""
    print(f"[INFO] Patch snippet: {patch[:200]!r}")
    return patch


def apply_patch(patch_text: str) -> bool:
    if not patch_text.strip():
        print("[WARN] Patch is empty or whitespace.")
        return False

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
        tf.write(patch_text)
        tf.flush()
        path = tf.name

    print(f"[INFO] Applying patch from file {path}")
    run_shell(f"git apply --reject --whitespace=fix {path}")

    # Stage all changes (new or modified)
    run_shell("git add .")

    # Check staged changes (including new files)
    staged = run_shell("git diff --cached --name-only").strip()
    print(f"[INFO] Staged files: {staged!r}")

    if staged:
        return True
    else:
        print("[INFO] No files staged even after git add.")
        return False


def commit_changes(branch: str):
    print(f"[INFO] Creating and switching to branch {branch}")
    run_shell(f"git checkout -b {branch}")
    run_shell("git config user.name github-actions")
    run_shell("git config user.email github-actions@github.com")
    run_shell("git add .")
    run_shell('git commit -m "fix: auto patch via Codex"')
    print("[INFO] Commit completed")


def push_branch(branch: str):
    print(f"[INFO] Pushing branch {branch} to origin")
    # set upstream so PR base will see it
    run_shell(f"git push --set-upstream origin {branch}")


def create_pr(branch: str, base: str, title: str, body: str) -> str:
    print(f"[INFO] Opening PR: {branch} → {base} with title {title!r}")
    # Use newer auth syntax if possible
    token = os.environ.get("GITHUB_TOKEN")
    # For PyGithub v2+, they recommend `auth=Auth.Token(...)`
    gh = (
        Github(auth=Auth.Token(str(token))) if hasattr(Github, "__init__") else Github(token)
    )
    repo = gh.get_repo(str(os.environ.get("GITHUB_REPOSITORY")))
    pr = repo.create_pull(title=title, body=body, head=branch, base=base)
    print(f"[INFO] PR created number {pr.number}")
    return str(pr.number)


def main():
    issue_number = os.getenv("ISSUE_NUMBER")
    issue_title = os.getenv("ISSUE_TITLE", "")
    issue_body = os.getenv("ISSUE_BODY", "")

    print(f"[INFO] Starting auto-fix for issue #{issue_number}, title {issue_title!r}")

    if not issue_number:
        print("[WARN] ISSUE_NUMBER not provided. Exiting.")
        sys.exit(0)

    patch = codex_suggest_patch(issue_title, issue_body)
    if not patch.strip():
        print("[INFO] No patch generated. Exiting without PR.")
        sys.exit(0)

    changed = apply_patch(patch)
    if not changed:
        print("[INFO] Patch applied, but no changes detected. Exiting.")
        sys.exit(0)

    branch = f"codex/auto-fix-issue-{issue_number}"
    commit_changes(branch)

    # Push to remote so PR creation sees the branch
    push_branch(branch)

    pr_title = f"Auto-fix issue #{issue_number}"
    pr_body = f"Issue: {issue_title}\n\nAuto-generated patch via Codex."
    try:
        pr_num = create_pr(branch, base="main", title=pr_title, body=pr_body)
    except Exception as e:
        print("[ERROR] Could not create PR:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Set output for GitHub Actions (new style)
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"pr_number={pr_num}\n")
    else:
        print(f"::set-output name=pr_number::{pr_num}")


if __name__ == "__main__":
    main()
