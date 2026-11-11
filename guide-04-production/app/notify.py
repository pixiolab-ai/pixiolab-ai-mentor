import os

from slack_sdk import WebClient

_slack = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))


def slack_post(text: str, channel: str) -> None:
    """Raises SlackApiError on failure, so callers can retry."""
    _slack.chat_postMessage(channel=channel, text=text)
