"""Configuration loaded from environment variables."""

import logging
import os
from pathlib import Path

PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")

_raw_keys = os.getenv("QWEN_CODE_API_KEY", "")
API_KEYS: list[str] | None = (
    [k.strip() for k in _raw_keys.split(",") if k.strip()]
    if _raw_keys.strip()
    else None
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "coder-model")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY_S = float(os.getenv("RETRY_DELAY_MS", "1000")) / 1000
QWEN_CODE_AUTH_USE = os.getenv("QWEN_CODE_AUTH_USE", "true").lower() != "false"

QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
TOKEN_REFRESH_BUFFER_S = 30

QWEN_DIR = Path.home() / ".qwen"
CREDS_FILE = QWEN_DIR / "oauth_creds.json"

LOG_LEVEL = os.getenv("LOG_LEVEL", "error")
logging.basicConfig(
    level=logging.DEBUG if LOG_LEVEL == "debug" else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("qwen-proxy")
