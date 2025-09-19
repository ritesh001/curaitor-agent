# tools/gmail_send.py
from __future__ import annotations
import os, base64
from dotenv import load_dotenv
from typing import Optional, List
from email.message import EmailMessage

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(ROOT_DIR, os.getenv("GMAIL_CREDENTIALS_PATH"))
TOKEN_PATH = os.path.join(ROOT_DIR, os.getenv("GMAIL_TOKEN_PATH"))
print(f"CREDENTIALS_PATH: {CREDENTIALS_PATH}")
print(f"TOKEN_PATH: {TOKEN_PATH}")

class AuthRequired(Exception):
    def __init__(self, auth_url: str, original_message: str = ""):
        super().__init__(original_message or "Authentication required")
        self.auth_url = auth_url
        self.original_message = original_message


def _svc():
    creds = None
    # print(f"CREDENTIALS_PATH: {CREDENTIALS_PATH}")
    # print(f"TOKEN_PATH: {TOKEN_PATH}")
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        try:
            # Preferred path: launches a local server and opens the browser.
            creds = flow.run_local_server(port=0)
        except Exception as e:
            # Common headless error: "no method available for opening 'https:...'"
            # Fall back to returning an explicit URL so the caller can prompt the user.
            auth_url, _state = flow.authorization_url(
                access_type="offline",
                include_granted_scopes="true",
                prompt="consent",
            )
            raise AuthRequired(auth_url=auth_url, original_message=str(e)) from e

        # Persist new token on success
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

def _build_message(
    to: str,
    subject: str,
    body: str,
    *,
    html: bool = False,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    from_alias: Optional[str] = None,
) -> EmailMessage:
    msg = EmailMessage()
    msg["To"] = to
    if cc: msg["Cc"] = cc
    if bcc: msg["Bcc"] = bcc
    msg["Subject"] = subject
    if reply_to: msg["Reply-To"] = reply_to
    if from_alias: msg["From"] = from_alias  # must be a verified alias in Gmail settings
    if html:
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)
    return msg

def gmail_send(
    to: str = "nsophonrat2@gmail.com",
    subject: str = "Test",
    body: str = "This is a test email from Curaitor Agent.",
    *,
    html: bool = False,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    from_alias: Optional[str] = None,
) -> dict:
    """
    Send an email via Gmail API. Returns {"id": ..., "threadId": ...} on success.
    """
    svc = _svc()
    msg = _build_message(
        to=to, subject=subject, body=body, html=html,
        cc=cc, bcc=bcc, reply_to=reply_to, from_alias=from_alias,
    )
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"id": sent.get("id"), "threadId": sent.get("threadId")}

# if __name__ == "__main__":
#     result = gmail_send()
        