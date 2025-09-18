# This file is used to create OAuth2 token for Gmail API and save it to a token file.
# Only need to run this once to generate the token file.
# Make sure you have credentials.json from Google Cloud Console. 
# And also add yourself to the "Test users" in OAuth consent screen if your app is in testing mode.

import os
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH")
TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH")

def create_token():

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=0)  # opens browser for consent

    with open(TOKEN_PATH, "w") as token:
        token.write(creds.to_json())

    print(f"Token saved to {TOKEN_PATH}")

if __name__ == "__main__":
    create_token()
