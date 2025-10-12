import os
import subprocess
import tempfile
import sys

from openai import OpenAI
from github import Github


def run_shell(cmd: str, check: bool = True) -> str:
    """Run a shell command, return stdout (strip). Exits on error if check=True."""
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and proc.returncode != 0:
        print(
            f"Command failed: {cmd}\nstdout: {proc.stdout}\nstderr: {proc.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
    return proc.stdout.strip()


def codex_suggest_patch(title: str, body: str) -> str:
    """Call OpenAI to get a minimal patch (unified diff) for the given issue."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    instructions = (
        "You are an expert code assistant. "
        "Given an issue title and body, and full repository context, "
        "generate a minimal patch (unified diff) that fixes the issue. "
        "Change only what is necessary. Do not refactor unrelated code."
    )
    user_input = f"Issue title: {title}\nIssue body: {body}\nGenerate patch diff:"
    # Use whatever endpoint interface your OpenAI SDK requires
    resp = client.responses.create(
        model="gpt-4o", instructions=instructions, input=user_input
    )
    return resp.output_text or ""


def apply_patch(patch_text: str) -> bool:
    """Apply the patch, return True if changes were made."""
    if not patch_text.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
        tf.write(patch_text)
        tf.flush()
        patch_path = tf.name
    # Use git apply (rejects or fixes whitespace)
    run_shell(f"git apply --reject --whitespace=fix {patch_path}")
    # Check whether there are any changes
    changed_files = run_shell("git diff --name-only")
    return bool(changed_files.strip())


def commit_changes(branch: str):
    run_shell(f"git checkout -b {branch}")
    run_shell("git config user.name github-actions")
    run_shell("git config user.email github-actions@github.com")
    run_shell("git add .")
    run_shell('git commit -m "fix: auto patch via Codex"')


def create_pr(branch: str, base: str, title: str, body: str) -> str:
    gh = Github(os.environ.get("GITHUB_TOKEN"))
    repo = gh.get_repo(str(os.environ.get("GITHUB_REPOSITORY")))
    pr = repo.create_pull(title=title, body=body, head=branch, base=base)
    return str(pr.number)


def main():
    issue_number = os.getenv("ISSUE_NUMBER")
    issue_title = os.getenv("ISSUE_TITLE", "")
    issue_body = os.getenv("ISSUE_BODY", "")

    if not issue_number:
        print("ISSUE_NUMBER not set, aborting.")
        sys.exit(0)

    patch = codex_suggest_patch(issue_title, issue_body)
    if not patch.strip():
        print("No patch suggested; exiting.")
        sys.exit(0)

    changed = apply_patch(patch)
    if not changed:
        print("Patch applied but no change detected; exiting.")
        sys.exit(0)

    branch = f"codex/auto-fix-issue-{issue_number}"
    commit_changes(branch)

    pr_title = f"Auto-fix issue #{issue_number}"
    pr_body = f"Issue: {issue_title}\n\nAuto-generated patch via Codex."
    pr_num = create_pr(branch, base="main", title=pr_title, body=pr_body)
    print(f"::set-output name=pr_number::{pr_num}")


if __name__ == "__main__":
    main()
