"""
Utility functions for Chimera.
"""

import os
import httpx
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_llm_client(model: str = "minimax-m2.7") -> Optional[object]:
    """Load MiniMax-compatible client using Anthropic /v1/messages endpoint."""
    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[INFO] No API key found. Running in MOCK mode with simulated responses.")
        return None
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
    return MiniMaxClient(api_key=api_key, base_url=base_url, model=model)


class MiniMaxClient:
    """Anthropic-compatible client for MiniMax API with retry and circuit breaker."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = "minimax-m2.7"
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0
        self._max_retries = 3
        self._base_delay = 1.0
        self._max_delay = 30.0
        self._timeout = 180.0

    def messages_create(self, messages: list, system: str = "", **kwargs) -> dict:
        """Call Anthropic /v1/messages endpoint with retry + circuit breaker."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        if system:
            body["system"] = system

        timeout = kwargs.get("timeout", self._timeout)

        for attempt in range(self._max_retries):
            if self._circuit_open:
                if time.time() < self._circuit_open_until:
                    wait_time = self._circuit_open_until - time.time()
                    logger.warning(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Circuit open, waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                self._circuit_open = False

            try:
                resp = httpx.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )

                if resp.status_code == 529:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 529 error (attempt {attempt + 1}/{self._max_retries})"
                    )
                    if self._consecutive_failures > 3:
                        self._circuit_open = True
                        self._circuit_open_until = time.time() + 60.0
                        logger.warning(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Circuit breaker opened, pausing 60s"
                        )
                else:
                    if self._consecutive_failures > 0:
                        self._consecutive_failures = 0

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 529:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 529 HTTP error (attempt {attempt + 1}/{self._max_retries})"
                    )
                    if self._consecutive_failures > 3:
                        self._circuit_open = True
                        self._circuit_open_until = time.time() + 60.0
                        logger.warning(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Circuit breaker opened, pausing 60s"
                        )
                if attempt < self._max_retries - 1:
                    delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                    logger.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Retrying in {delay:.1f}s (attempt {attempt + 2}/{self._max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise
            except httpx.TimeoutException as e:
                self._consecutive_failures += 1
                logger.warning(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Timeout error (attempt {attempt + 1}/{self._max_retries})"
                )
                if self._consecutive_failures > 3:
                    self._circuit_open = True
                    self._circuit_open_until = time.time() + 60.0
                    logger.warning(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Circuit breaker opened, pausing 60s"
                    )
                if attempt < self._max_retries - 1:
                    delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                    logger.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Retrying in {delay:.1f}s (attempt {attempt + 2}/{self._max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise
            except httpx.HTTPError as e:
                if attempt < self._max_retries - 1:
                    delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                    logger.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HTTP error: {e}, retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    raise


def pretty_print_results(results: dict):
    """Nicely format comparison results."""
    print("\n" + "="*70)
    print("CHIMERA EVALUATION RESULTS")
    print("="*70)
    for mode, res in results.items():
        print(f"\n{mode.upper()} MODE")
        print(f"  Quality Score : {res.get('quality_score', 'N/A')}")
        print(f"  Tokens (est.) : {res.get('total_tokens_estimate', 'N/A'):,}")
        if 'efficiency_gain' in res:
            print(f"  Efficiency    : {res['efficiency_gain']}")
        print(f"  Output length : {len(str(res.get('report', '')))} chars")
    print("\n" + "="*70)