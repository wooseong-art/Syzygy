"""HTTP GET with a global time budget, retries and a reproducibility log.

Every download is logged to stdout as one JSON line: url, fetch time, HTTP
status, size, sha256 and the server's Last-Modified header.
"""
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "syzygy-solarwind-forecast/1.0"
_deadline = None  # time.monotonic() value after which no new request starts


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_log_lock = threading.Lock()


def log(event, **fields):
    line = json.dumps({"event": event, "at": utcnow_iso(), **fields}, ensure_ascii=False, default=str) + "\n"
    with _log_lock:                                  # one write per line: download threads must not interleave
        sys.stdout.write(line)
        sys.stdout.flush()


def set_deadline(monotonic_deadline):
    global _deadline
    _deadline = monotonic_deadline


def remaining():
    return float("inf") if _deadline is None else _deadline - time.monotonic()


def get(url, timeout=30.0, retries=2, headers=None):
    """Return the response body (bytes), or None on 404 / repeated failure / exhausted budget."""
    for attempt in range(retries + 1):
        left = remaining()
        if left < 3:
            log("fetch_skipped", url=url, reason="time budget exhausted")
            return None
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        t = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=min(timeout, left)) as r:
                body = r.read()
                log("fetch", url=url, status=r.status, bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                    last_modified=r.headers.get("Last-Modified"),
                    elapsed_s=round(time.monotonic() - t, 2))
                return body
        except urllib.error.HTTPError as e:
            log("fetch_error", url=url, status=e.code, attempt=attempt)
            if e.code == 404:
                return None
        except Exception as e:  # timeouts, DNS, connection resets, ...
            log("fetch_error", url=url, error=repr(e)[:300], attempt=attempt)
        if attempt < retries:
            time.sleep(max(0.0, min(2.0 * (attempt + 1), remaining() - 3)))
    return None
