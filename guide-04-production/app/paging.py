import os
import smtplib
from email.message import EmailMessage


def page(subject: str, body: str) -> None:
    """Out-of-band alert by email. It shares nothing with Slack, so it
    still arrives when Slack, or the Slack token, is what broke."""
    msg = EmailMessage()
    msg["Subject"] = f"[ALERT] {subject}"
    msg["From"] = os.environ["ALERT_FROM"]
    msg["To"] = os.environ["ALERT_TO"]
    msg.set_content(body)
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
