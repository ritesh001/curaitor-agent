
import smtplib
from email.mime.text import MIMEText
import os

# === MCP TOOL: Literature Review Notification ===
def send_literature_review_report(email: str, summary: str, results: list[dict]):
    """
    Send a literature review report to the specified Gmail address.
    Args:
        email: Recipient's Gmail address.
        summary: One-sentence summary of the review.
        results: List of dicts with paper info (title, authors, date, pdf_url, snippet, etc).
    """
    GMAIL_USER = os.getenv("GMAIL_USER")
    GMAIL_PASS = os.getenv("GMAIL_PASS")
    if not GMAIL_USER or not GMAIL_PASS:
        raise RuntimeError("GMAIL_USER and GMAIL_PASS must be set in environment variables.")

    # Format the report
    body = f"Literature Review Summary:\n{summary}\n\nResults:\n"
    for i, r in enumerate(results, 1):
        body += f"[{i}] {r.get('title','')[:80]}\n"
        body += f"    Authors: {r.get('authors','Unknown')}\n"
        body += f"    Date: {r.get('published','')}\n"
        body += f"    PDF: {r.get('pdf_url','')}\n"
        body += f"    Snippet: {r.get('snippet','')[:200]}\n\n"

    msg = MIMEText(body)
    msg["Subject"] = "Your Literature Review Report"
    msg["From"] = GMAIL_USER
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, [email], msg.as_string())
    except Exception as e:
        raise RuntimeError(f"Failed to send literature review report: {e}")

# ---- Config (usually load from env or secret manager) ----
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
