from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import smtplib
from email.mime.text import MIMEText
import httpx
import os
from abc import ABC, abstractmethod

# ---- Config (usually load from env or secret manager) ----
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL")  # e.g., Twilio or Meta API
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WECHAT_API_URL = os.getenv("WECHAT_API_URL")
WECHAT_TOKEN = os.getenv("WECHAT_TOKEN")

# ---- Models ----
class NotificationRequest(BaseModel):
    notification_types: list[str] | str  # e.g., 'email', 'whatsapp', 'wechat', 'slack', or list
    email: EmailStr | None = None
    phone: str | None = None
    slack_user: str | None = None
    wechat_user: str | None = None
    message: str


# ---- Base Abstraction ----
class NotificationProvider(ABC):
    @abstractmethod
    async def send(self, recipient: str, message: str):
        pass


# ---- Concrete Providers ----
class EmailProvider(NotificationProvider):
    async def send(self, recipient: EmailStr, message: str):
        try:
            msg = MIMEText(message)
            msg["Subject"] = "MCP Notification"
            msg["From"] = GMAIL_USER
            msg["To"] = recipient

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_USER, GMAIL_PASS)
                server.sendmail(GMAIL_USER, [recipient], msg.as_string())
        except Exception as e:
            raise RuntimeError(f"Email send failed: {e}")


class SlackProvider(NotificationProvider):
    async def send(self, recipient: str, message: str):
        async with httpx.AsyncClient() as client:
            resp = await client.post(SLACK_WEBHOOK_URL, json={"text": message})
            if resp.status_code != 200:
                raise RuntimeError(f"Slack send failed: {resp.text}")


class WhatsAppProvider(NotificationProvider):
    async def send(self, recipient: str, message: str):
        payload = {
            "to": recipient,
            "type": "text",
            "text": {"body": message}
        }
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(WHATSAPP_API_URL, json=payload, headers=headers)
            if resp.status_code >= 300:
                raise RuntimeError(f"WhatsApp send failed: {resp.text}")


class WeChatProvider(NotificationProvider):
    async def send(self, recipient: str, message: str):
        payload = {
            "touser": recipient,
            "msgtype": "text",
            "text": {"content": message}
        }
        headers = {"Authorization": f"Bearer {WECHAT_TOKEN}"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(WECHAT_API_URL, json=payload, headers=headers)
            if resp.status_code >= 300:
                raise RuntimeError(f"WeChat send failed: {resp.text}")


# ---- Provider Registry ----
class ProviderRegistry:
    _providers = {
        "email": EmailProvider(),
        "slack": SlackProvider(),
        "whatsapp": WhatsAppProvider(),
        "wechat": WeChatProvider(),
    }

    _failover_priority = ["whatsapp", "slack", "wechat", "email"]

    @classmethod
    def get_provider(cls, channel: str):
        provider = cls._providers.get(channel)
        if not provider:
            raise ValueError(f"Unsupported channel: {channel}")
        return provider

    @classmethod
    def get_failover_sequence(cls):
        return [cls._providers[c] for c in cls._failover_priority if c in cls._providers]


# ---- Dispatcher ----
class NotificationDispatcher:
    @staticmethod
    async def dispatch(req: NotificationRequest):
        types = req.notification_types
        if isinstance(types, str):
            if types == "auto":
                # try all providers in priority order until one succeeds
                last_error = None
                for provider in ProviderRegistry.get_failover_sequence():
                    try:
                        recipient = NotificationDispatcher._get_recipient_for_provider(provider, req)
                        await provider.send(recipient, req.message)
                        return
                    except Exception as e:
                        last_error = e
                        continue
                raise RuntimeError(f"All providers failed. Last error: {last_error}")
            else:
                types = [types]
        results = {}
        for channel in types:
            try:
                provider = ProviderRegistry.get_provider(channel)
                recipient = NotificationDispatcher._get_recipient_for_channel(channel, req)
                await provider.send(recipient, req.message)
                results[channel] = "sent"
            except Exception as e:
                results[channel] = f"failed: {e}"
        return results

    @staticmethod
    def _get_recipient_for_channel(channel, req):
        if channel == "email":
            if not req.email:
                raise ValueError("Email address required for email notification.")
            return req.email
        elif channel == "whatsapp":
            if not req.phone:
                raise ValueError("Phone number required for WhatsApp notification.")
            return req.phone
        elif channel == "wechat":
            if not req.wechat_user:
                raise ValueError("WeChat user ID required for WeChat notification.")
            return req.wechat_user
        elif channel == "slack":
            if not req.slack_user:
                raise ValueError("Slack user ID required for Slack notification.")
            return req.slack_user
        else:
            raise ValueError(f"Unknown notification channel: {channel}")


# ---- FastAPI app ----
app = FastAPI()


@app.post("/notify")
async def notify(req: NotificationRequest):
    try:
        result = await NotificationDispatcher.dispatch(req)
        return {"status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

