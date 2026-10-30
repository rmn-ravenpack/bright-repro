"""Minimal Bigdata Content API client for enrich-document uploads.

Independent copy of the POST → PUT flow from the public Content API:
https://docs.bigdata.com/api-reference/documents/enrich-document
"""

from __future__ import annotations

import logging
import threading
import time

import requests

API_BASE_URL_DEFAULT = "https://api.bigdata.com"
DOCUMENTS_PATH = "/contents/v1/documents"


class RateLimiter:
    """Sliding 60-second window; counts REST calls to api.bigdata.com (not S3 PUTs)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max(1, max_per_minute)
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self.max_per_minute:
                sleep_time = 60.0 - (now - self._timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            self._timestamps.append(now)


def api_headers(api_key: str) -> dict[str, str]:
    return {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }


class BigdataContentClient:
    def __init__(
        self,
        api_key: str,
        rate_limiter: RateLimiter,
        base_url: str = API_BASE_URL_DEFAULT,
    ):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.base_url = base_url.rstrip("/")

    def create_document(
        self,
        file_name: str,
        *,
        published_ts: str | None = None,
        tags: list[str] | None = None,
        share_with_org: bool = False,
        enrichments: list[str] | None = None,
        timeout: int = 30,
    ) -> tuple[dict | None, int]:
        self.rate_limiter.acquire()
        url = f"{self.base_url}{DOCUMENTS_PATH}"
        payload: dict = {
            "file_name": file_name,
            "share_with_org": share_with_org,
        }
        if published_ts:
            payload["published_ts"] = published_ts
        if tags:
            payload["tags"] = tags
        if enrichments:
            payload["enrichments"] = enrichments
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=api_headers(self.api_key),
                timeout=timeout,
            )
            data = resp.json() if resp.text else None
            if resp.status_code >= 400:
                return data, resp.status_code
            return data, resp.status_code
        except requests.RequestException as exc:
            logging.warning("POST /contents/v1/documents failed: %s", exc)
            return None, 0

    def put_bytes(self, upload_url: str, payload: bytes, timeout: int = 300) -> tuple[bool, int]:
        """PUT file bytes to the pre-signed URL with no extra headers (signature-sensitive)."""
        try:
            resp = requests.put(upload_url, data=payload, headers={}, timeout=timeout)
            if resp.status_code >= 400:
                logging.warning(
                    "PUT failed status=%s body=%s",
                    resp.status_code,
                    (resp.text or "")[:300],
                )
                return False, resp.status_code
            return True, resp.status_code
        except requests.RequestException as exc:
            logging.warning("PUT to pre-signed URL failed: %s", exc)
            return False, 0

    def get_document(self, content_id: str, timeout: int = 30) -> tuple[dict | None, int]:
        self.rate_limiter.acquire()
        url = f"{self.base_url}{DOCUMENTS_PATH}/{content_id}"
        try:
            resp = requests.get(
                url,
                headers={"X-API-KEY": self.api_key},
                timeout=timeout,
            )
            data = resp.json() if resp.text else None
            return data, resp.status_code
        except requests.RequestException as exc:
            logging.warning("GET /contents/v1/documents/%s failed: %s", content_id, exc)
            return None, 0

    def delete_document(self, content_id: str, timeout: int = 30) -> tuple[dict | None, int]:
        """DELETE /contents/v1/documents/{id}. 200 means removed."""
        self.rate_limiter.acquire()
        url = f"{self.base_url}{DOCUMENTS_PATH}/{content_id}"
        try:
            resp = requests.delete(
                url,
                headers={"X-API-KEY": self.api_key},
                timeout=timeout,
            )
            data = None
            if resp.text:
                try:
                    data = resp.json()
                except ValueError:
                    data = {"raw": resp.text[:300]}
            return data, resp.status_code
        except requests.RequestException as exc:
            logging.warning("DELETE /contents/v1/documents/%s failed: %s", content_id, exc)
            return None, 0

    def list_documents(
        self,
        *,
        tags: list[str] | None = None,
        page: int = 1,
        page_size: int = 100,
        ownership: str = "owned",
        origin: str | None = "file_upload",
        timeout: int = 30,
    ) -> tuple[dict | None, int]:
        self.rate_limiter.acquire()
        url = f"{self.base_url}{DOCUMENTS_PATH}"
        params: dict = {
            "page": page,
            "page_size": page_size,
            "ownership": ownership,
            "sort_by": "created_at",
            "sort_order": "desc",
        }
        if origin:
            params["origin"] = origin
        if tags:
            params["tags"] = tags
        try:
            resp = requests.get(
                url,
                headers={"X-API-KEY": self.api_key},
                params=params,
                timeout=timeout,
            )
            data = resp.json() if resp.text else None
            return data, resp.status_code
        except requests.RequestException as exc:
            logging.warning("GET /contents/v1/documents failed: %s", exc)
            return None, 0
