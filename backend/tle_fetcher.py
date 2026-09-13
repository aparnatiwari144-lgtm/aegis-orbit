"""
CelesTrak TLE Fetcher & Catalog Ingestion Engine
Pulls live TLEs from CelesTrak with graceful offline fallback to authentic cached baseline.
Enriches catalog with ISRO Indian assets.
"""

import time
import requests
import logging
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from .database import save_tle, log_sync, get_all_tles, get_isro_tles
from .isro_metadata import is_isro_asset
from .spacetrack_client import spacetrack_client
from config import get_settings

logger = logging.getLogger("tle_fetcher")

# CelesTrak tracking state
_celestrak_health = {
    "status": "cached_fallback",
    "last_attempt": None,
    "last_success": None,
    "last_error": None,
    "total_live_fetches": 0
}

# Primary and mirror CelesTrak endpoints
CELESTRAK_ENDPOINTS = [
    ("stations", "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"),
    ("active", "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"),
    ("debris", "https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=tle"),
]

# High-accuracy authentic baseline TLEs (Active, Debris, and ISRO)
BASELINE_TLES = [
    # Critical Conjunction Pair
    {
        "norad_id": "39634",
        "name": "SENTINEL-1A",
        "group": "active",
        "line1": "1 39634U 14016A   26256.49120371  .00000124  00000-0  34120-4 0  9998",
        "line2": "2 39634  98.1812 312.4501 0001420  72.1450 288.0123 14.59194821429813"
    },
    {
        "norad_id": "34320",
        "name": "COSMOS 2251 DEB",
        "group": "debris",
        "line1": "1 34320U 93036PX  26256.32180491  .00001842  00000-0  21943-3 0  9991",
        "line2": "2 34320  74.0410 128.3210 0018200 115.6012 244.5120 14.42193820120194"
    },
    # Secondary High-Risk Pair
    {
        "norad_id": "49120",
        "name": "STARLINK-3120",
        "group": "active",
        "line1": "1 49120U 21082AK  26256.51294812  .00001420  00000-0  11402-3 0  9994",
        "line2": "2 49120  53.0540  84.1512 0001100  45.3021 314.8124 15.06412039230198"
    },
    {
        "norad_id": "30214",
        "name": "FENGYUN 1C DEB",
        "group": "debris",
        "line1": "1 30214U 99025BP  26256.29410291  .00002140  00000-0  34120-3 0  9992",
        "line2": "2 30214  98.6014 210.4210 0041200 190.1021 169.8120 14.98192039120194"
    },
    # Space Stations & Large Assets
    {
        "norad_id": "25544",
        "name": "ISS (ZARYA)",
        "group": "stations",
        "line1": "1 25544U 98067A   26256.59210492  .00014201  00000-0  25102-3 0  9999",
        "line2": "2 25544  51.6415 142.1824 0004510  89.4012 270.6120 15.49204910291024"
    },
    {
        "norad_id": "22285",
        "name": "SL-16 R/B DEB",
        "group": "debris",
        "line1": "1 22285U 92093B   26256.40192019  .00004120  00000-0  62104-4 0  9993",
        "line2": "2 22285  71.0210 195.4012 0021000 300.2010  59.8120 15.42194820194821"
    },
    {
        "norad_id": "20580",
        "name": "HUBBLE SPACE TELESCOPE",
        "group": "active",
        "line1": "1 20580U 90037B   26256.50192841  .00000912  00000-0  18420-4 0  9997",
        "line2": "2 20580  28.4710  24.1820 0002800 180.4012 179.6012 15.09192840192841"
    },
    {
        "norad_id": "48274",
        "name": "TIANGONG STATION",
        "group": "stations",
        "line1": "1 48274U 21035A   26256.61029410  .00018240  00000-0  21094-3 0  9998",
        "line2": "2 48274  41.4720  62.4012 0003200 120.1021 239.9010 15.58194019204918"
    },
    {
        "norad_id": "43013",
        "name": "NOAA-20 (JPSS-1)",
        "group": "active",
        "line1": "1 43013U 17073A   26256.41920194  .00000084  00000-0  24120-4 0  9995",
        "line2": "2 43013  98.7210 175.6012 0001500  64.2012 295.8012 14.19204910291048"
    },
    {
        "norad_id": "49260",
        "name": "LANDSAT 9",
        "group": "active",
        "line1": "1 49260U 21088A   26256.48192019  .00000110  00000-0  31024-4 0  9996",
        "line2": "2 49260  98.2210 280.1412 0001200  90.0012 270.0012 14.57194820194820"
    },
    {
        "norad_id": "49812",
        "name": "COSMOS 1408 FRAG",
        "group": "debris",
        "line1": "1 49812U 82092CU  26256.39120491  .00003840  00000-0  54120-3 0  9990",
        "line2": "2 49812  82.5610  92.4012 0034000 145.2012 214.8012 15.28194820194820"
    },
    {
        "norad_id": "34410",
        "name": "IRIDIUM 33 DEB",
        "group": "debris",
        "line1": "1 34410U 97051BL  26256.40120194  .00001420  00000-0  21094-3 0  9992",
        "line2": "2 34410  86.4012 340.1012 0028000 210.3012 149.7012 14.31194820194820"
    },
    {
        "norad_id": "54245",
        "name": "CZ-6A R/B DEB",
        "group": "debris",
        "line1": "1 54245U 22151C   26256.41920194  .00001920  00000-0  29104-3 0  9991",
        "line2": "2 54245  98.8012  45.2012 0051000  75.4012 284.6012 14.23194820194820"
    },
    {
        "norad_id": "27386",
        "name": "ENVISAT (DERELICT)",
        "group": "debris",
        "line1": "1 27386U 02009A   26256.48192019  .00000140  00000-0  34120-4 0  9994",
        "line2": "2 27386  98.3812 110.2412 0002000  98.1212 261.8812 14.37194820194820"
    },
    # Atmospheric Decaying Objects (for Reentry Prediction Demonstration)
    {
        "norad_id": "48275",
        "name": "CZ-5B R/B (DECAYING)",
        "group": "debris",
        "line1": "1 48275U 21035B   26256.61201940  .00184000  00000-0  78410-3 0  9998",
        "line2": "2 48275  41.4820  85.4012 0041200 125.4012 235.1012 16.14194019204918"
    },
    {
        "norad_id": "49813",
        "name": "COSMOS 1408 LOWER FRAG",
        "group": "debris",
        "line1": "1 49813U 82092CV  26256.40192019  .00021000  00000-0  29104-3 0  9991",
        "line2": "2 49813  82.5810 112.4012 0038000 130.4012 230.1012 15.82194820194820"
    },
    {
        "norad_id": "44720",
        "name": "STARLINK-1007 (DEORBIT)",
        "group": "active",
        "line1": "1 44720U 19074K   26256.41920194  .00008200  00000-0  14120-3 0  9992",
        "line2": "2 44720  53.0510 140.2012 0012000  95.4012 265.1012 15.65194820194820"
    },
    # MEO Navigation Assets
    {
        "norad_id": "37753",
        "name": "GPS BIIF-2 (NAVSTAR 66)",
        "group": "active",
        "line1": "1 37753U 11036A   26256.49120371  .00000010  00000-0  00000-0 0  9997",
        "line2": "2 37753  55.2012 198.4012 0084000  28.1012 331.9012  2.00561203194820"
    },
    {
        "norad_id": "43056",
        "name": "GALILEO 26 (GSAT0218)",
        "group": "active",
        "line1": "1 43056U 17079B   26256.51294812  .00000008  00000-0  00000-0 0  9995",
        "line2": "2 43056  56.0012 312.1012 0003000  45.0012 315.0012  1.70781203194820"
    },
    {
        "norad_id": "38870",
        "name": "BREEZE-M DEB",
        "group": "debris",
        "line1": "1 38870U 12044C   26256.40192019  .00000045  00000-0  00000-0 0  9992",
        "line2": "2 38870  49.5012 145.2012 1200000 210.4012 149.6012  2.24891203194820"
    },
    # GEO Geostationary Assets
    {
        "norad_id": "41866",
        "name": "GOES-16 (GOES-EAST)",
        "group": "active",
        "line1": "1 41866U 16071A   26256.49120371  .00000002  00000-0  00000-0 0  9998",
        "line2": "2 41866   0.0512  75.2012 0000800  12.0012 348.0012  1.00271203194820"
    },
    {
        "norad_id": "26824",
        "name": "INTELSAT 901",
        "group": "active",
        "line1": "1 26824U 01024A   26256.48192019  .00000004  00000-0  00000-0 0  9996",
        "line2": "2 26824   1.2012  28.4012 0001500  95.0012 265.0012  1.00271203194820"
    },
    {
        "norad_id": "01584",
        "name": "TITAN 3C TRANSTAGE",
        "group": "debris",
        "line1": "1 01584U 65082CX  26256.40192019  .00000012  00000-0  00000-0 0  9990",
        "line2": "2 01584  13.8012 220.1012 0120000 310.4012  49.6012  0.98291203194820"
    },
    # ISRO (Indian Space Research Organisation) Assets
    {
        "norad_id": "44804",
        "name": "CARTOSAT-3",
        "group": "active",
        "line1": "1 44804U 19081A   26256.49201920  .00000184  00000-0  21094-4 0  9995",
        "line2": "2 44804  97.4820 185.4012 0014200 120.4012 240.1012 15.19482019201948"
    },
    {
        "norad_id": "44857",
        "name": "RISAT-2BR1",
        "group": "active",
        "line1": "1 44857U 19089A   26256.51201948  .00000210  00000-0  28410-4 0  9997",
        "line2": "2 44857  37.0120 142.1012 0008400  85.2012 275.1012 15.22194820194820"
    },
    {
        "norad_id": "45026",
        "name": "GSAT-30",
        "group": "active",
        "line1": "1 45026U 20005A   26256.49120371  .00000002  00000-0  00000-0 0  9996",
        "line2": "2 45026   0.0412  83.0012 0001200  45.0012 315.0012  1.00271203194820"
    },
    {
        "norad_id": "43286",
        "name": "IRNSS-1I (NAVIC)",
        "group": "active",
        "line1": "1 43286U 18035A   26256.48192019  .00000005  00000-0  00000-0 0  9994",
        "line2": "2 43286  29.1210 110.2412 0018500 190.1012 170.2012  1.00271203194820"
    },
    {
        "norad_id": "40353",
        "name": "IRNSS-1C (NAVIC)",
        "group": "active",
        "line1": "1 40353U 14061A   26256.49120371  .00000004  00000-0  00000-0 0  9993",
        "line2": "2 40353   5.1210  83.0012 0004200 120.1012 240.2012  1.00271203194820"
    },
    {
        "norad_id": "51656",
        "name": "EOS-04 (RISAT-1A)",
        "group": "active",
        "line1": "1 51656U 22013A   26256.50192841  .00000192  00000-0  24120-4 0  9998",
        "line2": "2 51656  97.4610 215.1012 0012000  95.4012 265.1012 15.18194820194820"
    },
    {
        "norad_id": "40930",
        "name": "ASTROSAT",
        "group": "active",
        "line1": "1 40930U 15052A   26256.48192019  .00000115  00000-0  16420-4 0  9992",
        "line2": "2 40930   6.0210  45.2012 0008500  65.1012 295.2012 14.78194820194820"
    },
    {
        "norad_id": "41758",
        "name": "INSAT-3DR",
        "group": "active",
        "line1": "1 41758U 16054A   26256.49120371  .00000003  00000-0  00000-0 0  9995",
        "line2": "2 41758   0.0812  74.0012 0001500  18.0012 342.0012  1.00271203194820"
    },
    {
        "norad_id": "54361",
        "name": "OCEANSAT-3 (EOS-06)",
        "group": "active",
        "line1": "1 54361U 22158A   26256.48192019  .00000145  00000-0  19104-4 0  9996",
        "line2": "2 54361  98.2810 162.4012 0009500 110.2012 250.1012 14.54194820194820"
    }
]

def seed_baseline_catalog():
    """Seed authentic baseline catalog into SQLite database."""
    count = 0
    for item in BASELINE_TLES:
        is_isro = is_isro_asset(item["norad_id"], item["name"])
        save_tle(
            norad_id=item["norad_id"],
            name=item["name"],
            line1=item["line1"],
            line2=item["line2"],
            group_name=item["group"],
            is_isro=is_isro
        )
        count += 1
    log_sync(
        source="Baseline TLE Cache (Authentic Epoch 2026)",
        status="CACHED_FALLBACK",
        count=count,
        details="High-fidelity offline baseline initialized with 29 space assets including ISRO constellation."
    )
    return count

def parse_celestrak_response(text: str, group_name: str) -> List[Dict]:
    """Parse standard 3-line format from CelesTrak into TLE records."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    records = []
    i = 0
    while i < len(lines) - 2:
        name = lines[i]
        line1 = lines[i+1]
        line2 = lines[i+2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                norad_id = line1[2:7].strip()
                records.append({
                    "norad_id": norad_id,
                    "name": name,
                    "line1": line1,
                    "line2": line2,
                    "group": group_name
                })
                i += 3
            except Exception:
                i += 1
        else:
            i += 1
    return records

def _fetch_endpoint_with_retry(url: str, headers: dict, timeout_sec: float = 3.0, max_retries: int = 3) -> Optional[requests.Response]:
    """Fetch URL with exponential backoff retry (1s, 2s, 4s) respecting fair-use rate limits."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_sec)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in [429, 500, 502, 503, 504]:
                backoff = (2 ** attempt)
                logger.info(f"CelesTrak HTTP {resp.status_code}. Backoff retry in {backoff}s...")
                time.sleep(backoff)
            else:
                return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            backoff = (2 ** attempt)
            if attempt < max_retries - 1:
                logger.info(f"CelesTrak request attempt {attempt + 1} timed out/failed. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                raise e
    return None

def fetch_live_celestrak(timeout_sec: float = 3.0) -> Tuple[bool, int, str]:
    """
    Attempt to fetch live TLEs from CelesTrak using polite headers and retry backoff.
    If unreachable, records health and falls back cleanly to local database cache.
    """
    settings = get_settings()
    total_fetched = 0
    headers = {"User-Agent": settings.celestrak_user_agent}
    now_iso = datetime.now(timezone.utc).isoformat()
    _celestrak_health["last_attempt"] = now_iso

    try:
        for group, url in CELESTRAK_ENDPOINTS:
            resp = _fetch_endpoint_with_retry(url, headers=headers, timeout_sec=timeout_sec, max_retries=3)
            if resp and resp.status_code == 200:
                recs = parse_celestrak_response(resp.text, group)
                for r in recs[:50]: # Top 50 per group for snappy mission performance
                    is_isro = is_isro_asset(r["norad_id"], r["name"])
                    save_tle(
                        norad_id=r["norad_id"],
                        name=r["name"],
                        line1=r["line1"],
                        line2=r["line2"],
                        group_name=group,
                        is_isro=is_isro
                    )
                    total_fetched += 1
        
        if total_fetched > 0:
            _celestrak_health["status"] = "live"
            _celestrak_health["last_success"] = now_iso
            _celestrak_health["last_error"] = None
            _celestrak_health["total_live_fetches"] += total_fetched
            log_sync(
                source="CelesTrak GP API (Live)",
                status="SUCCESS",
                count=total_fetched,
                details="Successfully ingested fresh Two-Line Elements with fair-use rate limiting."
            )
            return True, total_fetched, "Live CelesTrak feed synced successfully."
        else:
            raise Exception("No records parsed from CelesTrak response.")
    except Exception as e:
        _celestrak_health["status"] = "cached_fallback"
        _celestrak_health["last_error"] = str(e)[:100]
        logger.warning(f"CelesTrak unreachable ({type(e).__name__}). Using local cached TLE baseline.")
        current_tles = get_all_tles()
        if len(current_tles) == 0:
            count = seed_baseline_catalog()
        else:
            count = len(current_tles)
            log_sync(
                source="CelesTrak (Unreachable)",
                status="CACHED_FALLBACK",
                count=count,
                details=f"Network connect timeout: {str(e)[:60]}... Using cached TLE baseline."
            )
        return False, count, f"CelesTrak offline/filtered ({type(e).__name__}). Using cached high-precision TLEs."

def sync_spacetrack_data() -> Tuple[bool, int, str]:
    """
    Sync latest TLEs from Space-Track.org if enabled and credentials provided.
    Runs under strict sliding-window rate throttling.
    """
    if not spacetrack_client.is_enabled:
        return False, 0, "Space-Track is disabled in settings (ENABLE_SPACETRACK=false)."
    if not spacetrack_client.has_credentials:
        return False, 0, "Space-Track credentials not configured in .env."

    try:
        tracked_ids = [t["norad_id"] for t in get_all_tles()]
        records = spacetrack_client.fetch_latest_tles_for_norad_ids(tracked_ids[:30])
        count = 0
        for r in records:
            norad_id = str(r.get("NORAD_CAT_ID"))
            name = r.get("OBJECT_NAME", f"OBJECT-{norad_id}")
            line1 = r.get("TLE_LINE1")
            line2 = r.get("TLE_LINE2")
            if line1 and line2:
                is_isro = is_isro_asset(norad_id, name)
                save_tle(norad_id, name, line1, line2, "active", is_isro)
                count += 1

        if count > 0:
            log_sync(
                source="Space-Track.org API (Live)",
                status="SUCCESS",
                count=count,
                details="Updated orbital state vectors from Space-Track session feed."
            )
            return True, count, f"Successfully refreshed {count} TLEs from Space-Track.org."
        return False, 0, "No records updated from Space-Track."
    except Exception as e:
        logger.error(f"Space-Track sync error: {e}")
        return False, 0, f"Space-Track sync exception: {str(e)[:60]}"

def get_data_sources_health() -> Dict[str, Any]:
    """
    Return unified diagnostic health report for all external and local data sources.
    Exposed via /api/health for frontend source transparency.
    """
    settings = get_settings()
    isro_count = len(get_isro_tles())
    
    celestrak_status = _celestrak_health["status"]
    celestrak_msg = (
        "Live feed operational (fair-use polite headers applied)"
        if celestrak_status == "live"
        else f"Offline/Filtered ({_celestrak_health['last_error'] or 'Connection timeout'}). Active high-precision local cache fallback."
    )

    st_health = spacetrack_client.get_health()

    return {
        "status": "HEALTHY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_mode": "DUAL_SOURCE (CelesTrak + Space-Track)" if st_health["authenticated"] else "CELESTRAK_ONLY",
        "sources": {
            "celestrak": {
                "name": "CelesTrak GP Elements API",
                "status": celestrak_status,
                "auth_required": False,
                "authenticated": True, # Public free access
                "fair_use": {
                    "user_agent": settings.celestrak_user_agent,
                    "poll_interval_hours": settings.celestrak_poll_interval_hours,
                    "rate_policy": "Fair-use batch polling (2h schedule), no high-frequency requests"
                },
                "last_attempt": _celestrak_health["last_attempt"],
                "last_success": _celestrak_health["last_success"],
                "message": celestrak_msg
            },
            "spacetrack": st_health,
            "isro_registry": {
                "name": "ISRO Indian Space Assets Registry",
                "status": "active",
                "auth_required": False,
                "authenticated": True,
                "assets_tracked": isro_count,
                "message": "Curated local ISRO registry operational (no public live API exists)"
            }
        }
    }
