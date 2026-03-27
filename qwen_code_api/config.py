"""Configuration loaded from environment variables using pydantic-settings."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server configuration
    port: int = Field(..., alias="PORT")
    address: str = Field(..., alias="ADDRESS")

    # API authentication
    qwen_code_api_key: str = Field(..., alias="QWEN_CODE_API_KEY")
    qwen_code_auth_use: bool = Field(..., alias="QWEN_CODE_AUTH_USE")

    # Model configuration
    default_model: str = Field(..., alias="DEFAULT_MODEL")

    # Retry configuration
    max_retries: int = Field(..., alias="MAX_RETRIES")
    retry_delay_ms: int = Field(..., alias="RETRY_DELAY_MS")

    # Qwen API configuration (hardcoded, not from env)
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_oauth_token_url: str = "https://chat.qwen.ai/api/v1/oauth2/token"
    qwen_oauth_client_id: str = "f0304373b74a44d2b584a3fb70ca9e56"
    token_refresh_buffer_s: int = 30

    # File paths
    qwen_dir: Path = Path.home() / ".qwen"
    creds_file: Path = Path.home() / ".qwen" / "oauth_creds.json"

    # Logging
    log_level: str = Field(..., alias="LOG_LEVEL")
    log_requests: bool = Field(..., alias="LOG_REQUESTS")

    @property
    def api_keys(self) -> list[str] | None:
        """Parse API keys from comma-separated string."""
        if not self.qwen_code_api_key:
            return None
        keys = [k.strip() for k in self.qwen_code_api_key.split(",") if k.strip()]
        return keys if keys else None

    @property
    def retry_delay_s(self) -> float:
        """Convert retry delay from milliseconds to seconds."""
        return self.retry_delay_ms / 1000


settings = Settings.model_validate({})


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
        }

        message = record.getMessage()
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            payload["message"] = message
        else:
            if isinstance(parsed, dict):
                payload.update(parsed)
            else:
                payload["message"] = message

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


# Configure standard logging.
# When the service runs under opentelemetry-instrument, the root logger gets
# the OTel handler automatically; this fallback keeps local runs structured.
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(JsonFormatter())
logging.basicConfig(
    level=logging.DEBUG if settings.log_level == "debug" else logging.INFO,
    handlers=[handler],
)
log = logging.getLogger("qwen_code_api")
