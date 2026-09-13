"""
Space-Track.org Authenticated Client & Rate-Limiting Engine
Handles session-based login, cookie caching, automatic 401 re-authentication,
sliding-window rate throttling (max 30 req/min, 300 req/hr), and exponential backoff.
"""

import time
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import requests
from config import get_settings

logger = logging.getLogger("spacetrack_client")

# Safety rate limits mandated by Space-Track.org
MAX_REQUESTS_PER_MINUTE = 30
MAX_REQUESTS_PER_HOUR = 300
WINDOW_MINUTE_SEC = 60.0
WINDOW_HOUR_SEC = 3600.0

SPACE_TRACK_BASE_URL = "https://www.space-track.org"
SPACE_TRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding window rate limiter ensuring requests never exceed
    Space-Track's 30/min and 300/hr caps to eliminate IP-ban risks.
    """

    def __init__(self, max_per_minute: int = MAX_REQUESTS_PER_MINUTE, max_per_hour: int = MAX_REQUESTS_PER_HOUR):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_window = deque()
        self.hour_window = deque()
        self.lock = threading.Lock()

    def _prune(self, now: float):
        """Evict timestamps outside the sliding windows."""
        cutoff_min = now - WINDOW_MINUTE_SEC
        while self.minute_window and self.minute_window[0] < cutoff_min:
            self.minute_window.popleft()

        cutoff_hour = now - WINDOW_HOUR_SEC
        while self.hour_window and self.hour_window[0] < cutoff_hour:
            self.hour_window.popleft()

    def acquire(self):
        """
        Block until a request token is available in both sliding windows.
        Thread-safe and sleep-backed.
        """
        while True:
            with self.lock:
                now = time.time()
                self._prune(now)

                wait_sec = 0.0
                if len(self.minute_window) >= self.max_per_minute:
                    wait_sec = max(wait_sec, WINDOW_MINUTE_SEC - (now - self.minute_window[0]) + 0.1)

                if len(self.hour_window) >= self.max_per_hour:
                    wait_sec = max(wait_sec, WINDOW_HOUR_SEC - (now - self.hour_window[0]) + 0.1)

                if wait_sec <= 0.0:
                    self.minute_window.append(now)
                    self.hour_window.append(now)
                    return

            logger.info(f"Rate limiter throttling: sleeping {wait_sec:.2f}s for Space-Track quota window.")
            time.sleep(wait_sec)

    def get_stats(self) -> Dict[str, Any]:
        """Return current rate limiter utilization."""
        with self.lock:
            now = time.time()
            self._prune(now)
            return {
                "max_per_minute": self.max_per_minute,
                "max_per_hour": self.max_per_hour,
                "current_minute_requests": len(self.minute_window),
                "current_hour_requests": len(self.hour_window)
            }


class SpaceTrackClient:
    """
    Session-authenticated client for Space-Track.org with cookie caching,
    re-authentication on 401, exponential backoff, and graceful fallback.
    """

    def __init__(self):
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AEGIS-Orbit-AI/1.0 (Space Debris Defense Research; contact@aegis-orbit.local)"
        })
        self.rate_limiter = SlidingWindowRateLimiter()
        self.is_authenticated = False
        self.last_login_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._auth_lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        return self.settings.enable_spacetrack

    @property
    def has_credentials(self) -> bool:
        return self.settings.has_spacetrack_credentials

    def authenticate(self) -> bool:
        """
        Authenticate with Space-Track via POST to /ajaxauth/login.
        Caches the session cookie in self.session.
        """
        if not self.is_enabled:
            self.last_error = "Space-Track is disabled (ENABLE_SPACETRACK=false)."
            self.is_authenticated = False
            return False

        if not self.has_credentials:
            self.last_error = "Space-Track credentials not configured."
            self.is_authenticated = False
            return False

        with self._auth_lock:
            try:
                self.rate_limiter.acquire()
                payload = {
                    "identity": self.settings.spacetrack_username,
                    "password": self.settings.spacetrack_password
                }
                logger.info("Authenticating with Space-Track.org session login...")
                resp = self.session.post(SPACE_TRACK_LOGIN_URL, data=payload, timeout=8.0)

                # Space-Track returns 200 with error text on failed auth, or sets session cookie on success
                if resp.status_code == 200 and "Login Failed" not in resp.text and len(self.session.cookies) > 0:
                    self.is_authenticated = True
                    self.last_login_time = datetime.now(timezone.utc)
                    self.last_error = None
                    logger.info("Space-Track authentication successful. Session cookie cached.")
                    return True
                else:
                    err_msg = resp.text.strip() if resp.text else f"HTTP {resp.status_code}"
                    self.last_error = f"Authentication rejected: {err_msg[:80]}"
                    self.is_authenticated = False
                    logger.warning(f"Space-Track login failed: {self.last_error}")
                    return False
            except Exception as e:
                self.last_error = f"Connection error during login: {str(e)[:80]}"
                self.is_authenticated = False
                logger.warning(f"Space-Track authentication exception: {self.last_error}")
                return False

    def request(self, endpoint_url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Optional[requests.Response]:
        """
        Execute rate-throttled request against Space-Track with automatic 401 re-login
        and exponential backoff retry logic.
        """
        if not self.is_enabled or not self.has_credentials:
            return None

        # Ensure active authentication
        if not self.is_authenticated:
            if not self.authenticate():
                return None

        for attempt in range(max_retries):
            try:
                self.rate_limiter.acquire()
                resp = self.session.get(endpoint_url, params=params, timeout=10.0)

                # Check if session expired (401 Unauthorized or redirected to login)
                if resp.status_code == 401 or "ajaxauth/login" in resp.url or "Login" in resp.text[:120]:
                    logger.warning("Space-Track session expired (401/redirect). Re-authenticating...")
                    self.is_authenticated = False
                    if self.authenticate():
                        # Retry once with refreshed session
                        self.rate_limiter.acquire()
                        resp = self.session.get(endpoint_url, params=params, timeout=10.0)
                        if resp.status_code == 200:
                            return resp

                if resp.status_code == 200:
                    return resp
                elif resp.status_code in [429, 500, 502, 503, 504]:
                    # Server overload or rate limit warning - exponential backoff
                    delay = (2 ** attempt) + 1.0
                    logger.warning(f"Space-Track HTTP {resp.status_code}. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    logger.warning(f"Space-Track returned non-retryable HTTP {resp.status_code}: {resp.text[:100]}")
                    return resp
            except (requests.ConnectionError, requests.Timeout) as e:
                delay = (2 ** attempt) + 1.0
                logger.warning(f"Space-Track network error: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected Space-Track request error: {e}")
                break

        return None

    def fetch_latest_tles_for_norad_ids(self, norad_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch latest TLE records for specific NORAD catalog IDs from Space-Track GP query.
        """
        if not norad_ids or not self.is_enabled or not self.has_credentials:
            return []

        id_list = ",".join(norad_ids[:50])
        url = f"{SPACE_TRACK_BASE_URL}/basicspacedata/v1/query/class/gp/NORAD_CAT_ID/{id_list}/orderby/NORAD_CAT_ID/format/json"
        
        resp = self.request(url)
        if resp and resp.status_code == 200:
            try:
                return resp.json()
            except Exception as e:
                logger.error(f"Failed to parse Space-Track JSON response: {e}")
        return []

    def get_health(self) -> Dict[str, Any]:
        """
        Return structured health report for Space-Track data source.
        Reports authentication state, rate limit consumption, and operational mode.
        """
        if not self.is_enabled:
            status = "disabled"
            msg = "Space-Track integration disabled (ENABLE_SPACETRACK=false). Running in CelesTrak-only mode."
        elif not self.has_credentials:
            status = "not_configured"
            msg = "Space-Track enabled but credentials not configured in .env. Running in CelesTrak-only mode."
        elif self.is_authenticated:
            status = "authenticated"
            msg = "Space-Track session active and authenticated."
        else:
            status = "unauthenticated"
            msg = f"Space-Track authentication pending or failed: {self.last_error or 'No active session'}"

        return {
            "source_id": "spacetrack",
            "name": "Space-Track.org API",
            "status": status,
            "auth_required": True,
            "authenticated": self.is_authenticated,
            "enabled": self.is_enabled,
            "configured": self.has_credentials,
            "last_login": self.last_login_time.isoformat() if self.last_login_time else None,
            "rate_limits": self.rate_limiter.get_stats(),
            "message": msg
        }


# Global singleton instance
spacetrack_client = SpaceTrackClient()
