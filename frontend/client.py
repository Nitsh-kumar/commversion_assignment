"""HTTP client for the Ashcombe & Rowe Estates FastAPI backend (zero backend imports)."""

from __future__ import annotations

from typing import Any

import httpx


class ChatbotApiClient:
    """Strictly HTTP client communicating with the FastAPI backend."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if self._client is not None:
            resp = self._client.request(method, url, json=json, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                resp = client.request(method, url, json=json)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, Any]:
        """Check FastAPI backend health."""
        return self._request("GET", "/health", timeout=5.0)

    def start_session(self) -> dict[str, Any]:
        """Create a new backend session via POST /sessions."""
        return self._request("POST", "/sessions", timeout=30.0)

    def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        """Send one visitor message via POST /chat."""
        return self._request(
            "POST",
            "/chat",
            json={"session_id": session_id, "message": message},
            timeout=60.0,
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Fetch session state and turn traces via GET /sessions/{session_id}."""
        return self._request("GET", f"/sessions/{session_id}", timeout=15.0)
