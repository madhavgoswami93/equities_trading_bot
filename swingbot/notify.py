"""Alerts: send the daily report (or a failure) to Telegram.

Set up once: talk to @BotFather to create a bot, send it a message, then put
TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env and `telegram = true` under
[alerts] in the strategy file. Alerts never stop a run: a failed send is logged.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)
LIMIT = 4000  # Telegram caps messages at 4096 characters


def telegram(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("telegram alerts enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
        return False
    body = json.dumps({"chat_id": chat, "text": f"<pre>{_escape(text[:LIMIT])}</pre>", "parse_mode": "HTML"})
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body.encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except Exception as e:  # noqa: BLE001
        log.warning("telegram alert failed: %s", e)
        return False


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
