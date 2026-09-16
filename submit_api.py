#!/usr/bin/env python3
"""Submit a ZIP to the NeurIPS Predictive AI Evaluation Competition.

Usage:
    python submit_api.py /path/to/submission.zip
    python submit_api.py --login
    python submit_api.py --list
    python submit_api.py --status SUBMISSION_ID

Requires: requests (python -m pip install requests)
Your CodaBench account must already be approved for this competition.

--login saves an API token with owner-only permissions outside the repository.
Later commands reuse it. CODABENCH_TOKEN takes precedence; otherwise credentials
are prompted for when no saved token is available. Passwords are never saved.
"""

import argparse
import getpass
import os
import tempfile
import uuid
import zipfile
from pathlib import Path

import requests

BASE = "https://www.codabench.org/api"
COMPETITION_ID = 17828
PHASE_ID = 29785
TOKEN_FILE = Path(
    os.environ.get(
        "CODABENCH_TOKEN_FILE", str(Path.home() / ".config/paiec/codabench-token")
    )
).expanduser()


def api(session, method, path, **kwargs):
    response = session.request(method, f"{BASE}/{path}", timeout=60, **kwargs)
    if not response.ok:
        raise RuntimeError(
            f"{method} {path}: HTTP {response.status_code}\n{response.text}"
        )
    return response.json() if response.content else None


def authenticate(session, save=False):
    token = os.environ.get("CODABENCH_TOKEN")
    if not token and not save and TOKEN_FILE.is_file():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        auth = api(
            session,
            "POST",
            "api-token-auth/",
            json={
                "username": os.environ.get("CODABENCH_USERNAME")
                or input("CodaBench username: "),
                "password": os.environ.get("CODABENCH_PASSWORD")
                or getpass.getpass("CodaBench password: "),
            },
        )
        token = auth["token"]
    session.headers["Authorization"] = f"Token {token}"
    if save:
        # Verify access before replacing a saved token. Never print the token.
        api(session, "GET", "my_profile/")
        TOKEN_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".codabench-token-", dir=TOKEN_FILE.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(token + "\n")
            os.replace(name, TOKEN_FILE)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        print(f"Saved Codabench token to {TOKEN_FILE} (mode 600).")


def show_status(session, submission_id):
    status = api(session, "GET", f"submissions/{submission_id}/")
    print(f"Submission: {status['id']}")
    print("Status:", status["status"])
    print("Details:", status.get("status_details") or "(none)")
    print("Scores:", status.get("scores", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zip_path", type=Path, nargs="?", help="Competition submission ZIP"
    )
    parser.add_argument(
        "--login", action="store_true", help="Save a private API token locally"
    )
    parser.add_argument(
        "--list",
        dest="list_submissions",
        action="store_true",
        help="List recent submissions accessible to this account",
    )
    parser.add_argument(
        "--status",
        type=int,
        metavar="ID",
        help="Read an existing submission's status without submitting",
    )
    args = parser.parse_args()
    if (
        sum(
            (
                args.zip_path is not None,
                args.login,
                args.list_submissions,
                args.status is not None,
            )
        )
        != 1
    ):
        parser.error("choose a ZIP, --login, --list, or --status ID")
    if args.zip_path is not None:
        zip_path = args.zip_path.expanduser()
        if not zip_path.is_file():
            parser.error(f"File not found: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            parser.error(f"Not a valid ZIP archive: {zip_path}")
        file_size = zip_path.stat().st_size

    with requests.Session() as session:
        # 1. Authenticate; read-only commands return before the upload path.
        authenticate(session, save=args.login)
        if args.login:
            return
        if args.status is not None:
            show_status(session, args.status)
            return
        if args.list_submissions:
            submissions = api(
                session,
                "GET",
                "submissions/",
                params={
                    "phase": PHASE_ID,
                    "page_size": 20,
                },
            )
            for submission in submissions.get("results", []):
                print(
                    submission["id"],
                    submission["status"],
                    submission.get("created_when", ""),
                )
            return

        # 2. Check approval and submission limits before uploading.
        competition = api(session, "GET", f"competitions/{COMPETITION_ID}/")
        eligibility = api(session, "GET", f"can_make_submission/{PHASE_ID}/")
        if not eligibility["can"]:
            raise RuntimeError(eligibility.get("reason") or "Cannot submit")

        # 3. Create a dataset record and obtain a signed upload URL.
        dataset = api(
            session,
            "POST",
            "datasets/",
            json={
                "name": f"submission-{uuid.uuid4().hex}",
                "type": "submission",
                "competition": COMPETITION_ID,
                "is_public": False,
                "request_sassy_file_name": zip_path.name,
                "file_name": zip_path.name,
                "file_size": file_size,
            },
        )

        # 4. Send raw ZIP bytes without sending the API token to storage.
        print(f"Uploading {zip_path.name} ({file_size:,} bytes)...", flush=True)
        with zip_path.open("rb") as file:
            try:
                response = requests.put(
                    dataset["sassy_url"],
                    data=file,
                    headers={"Content-Type": "application/zip"},
                    timeout=600,
                )
            except requests.RequestException:
                # Do not include the signed storage URL in error output.
                raise RuntimeError("ZIP upload failed due to a network error") from None
        if not response.ok:
            raise RuntimeError(f"ZIP upload failed: HTTP {response.status_code}")

        # 5. Mark the upload complete.
        api(session, "PUT", f"datasets/completed/{dataset['key']}/")

        # 6. Create the submission and show its initial status.
        submission = api(
            session,
            "POST",
            "submissions/",
            json={
                "phase": PHASE_ID,
                "data": dataset["key"],
                "queue": (competition.get("queue") or {}).get("id"),
            },
        )

        submission_id = submission["id"]
        print(f"Submission created: {submission_id}", flush=True)
        print(f"Status endpoint: {BASE}/submissions/{submission_id}/")
        show_status(session, submission_id)


if __name__ == "__main__":
    main()
