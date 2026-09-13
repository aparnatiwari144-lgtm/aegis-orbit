"""
NOAA SWPC Space Weather Ingestion Engine
Fetches the 10.7 cm solar radio flux (F10.7) from NOAA Space Weather Prediction Center.
Solar flux directly drives thermospheric density expansion, which governs satellite drag.
"""

import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger("space_weather")

NOAA_F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
# Baseline average for Solar Cycle 25 (moderate-to-high solar activity)
BASELINE_F107_SFU = 148.5 

_cache = {
    "f107_sfu": BASELINE_F107_SFU,
    "last_fetched": 0.0,
    "status": "INITIALIZING",
    "source": "NOAA SWPC Baseline",
    "details": "Default Solar Cycle 25 baseline flux"
}


def fetch_noaa_solar_flux(timeout_sec: float = 3.5) -> Dict[str, Any]:
    """
    Fetch the latest 10.7 cm Solar Radio Flux (F10.7) from NOAA SWPC API.
    Caches the result for 1 hour to respect NOAA rate limits and network latency.
    """
    now = time.time()
    # Return cached value if fetched within the last hour (3600s)
    if _cache["status"] in ["LIVE", "CACHED"] and (now - _cache["last_fetched"]) < 3600:
        return {
            "f107_sfu": _cache["f107_sfu"],
            "status": _cache["status"],
            "source": _cache["source"],
            "details": _cache["details"],
            "drag_density_multiplier": calculate_drag_density_multiplier(_cache["f107_sfu"])
        }

    headers = {
        "User-Agent": "AEGIS-Orbit-SpaceDebrisAI/1.0 (NOAA-SWPC-Research; contact: contact@aegis-orbit.local)"
    }

    try:
        resp = requests.get(NOAA_F107_URL, headers=headers, timeout=timeout_sec)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                # Last entry is the most recent observation
                latest_entry = data[-1]
                flux_val = float(latest_entry.get("flux", BASELINE_F107_SFU))
                obs_time = latest_entry.get("time_tag", datetime.now(timezone.utc).isoformat())

                _cache["f107_sfu"] = round(flux_val, 1)
                _cache["last_fetched"] = now
                _cache["status"] = "LIVE"
                _cache["source"] = "NOAA SWPC API (Live)"
                _cache["details"] = f"Observed flux: {flux_val} sfu at {obs_time}"
                logger.info(f"Updated NOAA F10.7 solar flux: {flux_val} sfu.")
                return {
                    "f107_sfu": _cache["f107_sfu"],
                    "status": "LIVE",
                    "source": _cache["source"],
                    "details": _cache["details"],
                    "drag_density_multiplier": calculate_drag_density_multiplier(flux_val)
                }
    except Exception as e:
        logger.warning(f"NOAA SWPC API unreachable ({type(e).__name__}). Using Solar Cycle 25 baseline.")

    # Fallback to smoothed baseline
    _cache["last_fetched"] = now
    _cache["status"] = "CACHED"
    _cache["source"] = "Solar Cycle 25 Empirical Baseline"
    _cache["details"] = "NOAA SWPC offline/timeout. Using 148.5 sfu empirical baseline."
    return {
        "f107_sfu": _cache["f107_sfu"],
        "status": "CACHED",
        "source": _cache["source"],
        "details": _cache["details"],
        "drag_density_multiplier": calculate_drag_density_multiplier(_cache["f107_sfu"])
    }


def calculate_drag_density_multiplier(f107_sfu: float) -> float:
    """
    Calculate atmospheric thermospheric density multiplier relative to quiet Sun (F10.7 = 70 sfu).
    Thermospheric density roughly scales exponentially with solar radio flux:
    rho / rho_quiet ~= exp(0.0075 * (F10.7 - 70))
    """
    delta = max(0.0, f107_sfu - 70.0)
    # Density amplification factor (ranges ~1.0 at quiet solar min to ~2.5+ at active solar max)
    multiplier = round(1.0 + 0.0085 * delta, 3)
    return multiplier


def get_current_solar_flux() -> float:
    """Quick helper returning latest solar flux float."""
    return fetch_noaa_solar_flux()["f107_sfu"]
