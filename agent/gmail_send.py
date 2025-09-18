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
CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH")
TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH")
# print(f"CREDENTIALS_PATH: {CREDENTIALS_PATH}")
# print(f"TOKEN_PATH: {TOKEN_PATH}")

def _svc():
    creds = None
    # print(f"CREDENTIALS_PATH: {CREDENTIALS_PATH}")
    # print(f"TOKEN_PATH: {TOKEN_PATH}")
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=0)
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
        