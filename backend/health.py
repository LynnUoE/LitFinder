"""
Dependency checks for /api/health_check.

The app starts and serves pages with an LLM API key the provider no longer
accepts (expired or revoked), but every search then fails. LLMKeyCheck asks
the provider to list its models, which needs a valid key and costs nothing,
and remembers the answer for a few minutes, so frequent health checks (Docker
polls every 30 seconds, and the endpoint is public) make at most one request
per interval.
"""
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import openai

CHECK_INTERVAL = 300  # Seconds between requests to the provider
CHECK_TIMEOUT = 4.0   # Docker's health check gives the whole endpoint 5 seconds

logger = logging.getLogger("health")


def create_check_client(api_key: str, base_url: Optional[str] = None) -> openai.OpenAI:
    """A client that fails fast: no retries, short timeout."""
    return openai.OpenAI(api_key=api_key, base_url=base_url or None, max_retries=0, timeout=CHECK_TIMEOUT)


class LLMKeyCheck:
    """Whether the LLM provider accepts the API key: "ok", "invalid_key", or "unknown" when it can't tell."""

    def __init__(self, client_factory: Callable[[], openai.OpenAI], interval: float = CHECK_INTERVAL,
                 clock: Callable[[], float] = time.monotonic):
        self.client_factory = client_factory
        self.interval = interval
        self.clock = clock
        self._client: Optional[openai.OpenAI] = None
        self._result: Optional[Dict] = None
        self._checked = 0.0
        self._lock = threading.Lock()

    def status(self) -> Dict:
        # One request at a time; concurrent health checks wait for it and share the answer
        with self._lock:
            if self._result is None or self.clock() - self._checked >= self.interval:
                self._result = self._check()
                self._checked = self.clock()
            return self._result

    def _check(self) -> Dict:
        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            if self._client is None:
                self._client = self.client_factory()
            self._client.models.list()
            return {"status": "ok", "checked_at": checked_at}
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            # The provider's message can quote part of the key, so only the log gets it
            logger.error(f"LLM provider rejected the API key: {e}")
            return {"status": "invalid_key", "checked_at": checked_at,
                    "message": f"The LLM provider rejected the API key (HTTP {e.status_code}); "
                               "replace OPENAI_API_KEY / LLM_API_KEY and restart"}
        except Exception as e:
            # Timeouts, outages, or a provider without a models endpoint: not proof the key is bad
            logger.warning(f"LLM key check inconclusive: {type(e).__name__}: {e}")
            return {"status": "unknown", "checked_at": checked_at,
                    "message": f"Could not verify the API key ({type(e).__name__})"}
