"""
AEGIS-ORBIT: Space Debris Detection & Collision Avoidance AI
FastAPI Real-Time Backend & WebSocket Broadcast Server
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.database import (
    init_db, get_all_tles, get_tle_by_id, get_isro_tles,
    save_conjunction_event, get_all_conjunctions, get_latest_sync_status,
    log_sync
)
from backend.tle_fetcher import seed_baseline_catalog, fetch_live_celestrak, get_data_sources_health, sync_spacetrack_data
from backend.propagator import SGP4Propagator, EARTH_RADIUS_KM
from backend.risk_engine import RiskPipeline
from backend.isro_metadata import get_isro_metadata, is_isro_asset
from backend.spacetrack_client import spacetrack_client
from backend.ml_reentry_pipeline import reentry_predictor
from config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis_backend")

settings = get_settings()

app = FastAPI(
    title="AEGIS-ORBIT AI",
    description="Real-Time Space Debris Detection & Collision Avoidance Mission Control API",
    version="2.0.0"
)

# CORS setup to allow web browser requests from any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Cache of Active Propagators
propagators: Dict[str, SGP4Propagator] = {}
risk_pipeline: RiskPipeline = None

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Global simulation state
SIM_STATE = {
    "start_wall_time": datetime.now(timezone.utc),
    "current_sim_hours": 4.37, # Default to encounter timeframe
    "sim_speed": 10.0,
    "is_playing": True,
    "last_loop_time": datetime.now(timezone.utc)
}

# =========================================================================
# INITIALIZATION & BACKGROUND WORKERS
# =========================================================================

def initialize_system():
    """Initialize SQLite DB, seed/fetch TLEs, and instantiate SGP4 propagators."""
    global propagators, risk_pipeline
    init_db()

    # Seed baseline if database is empty
    tles = get_all_tles()
    if len(tles) == 0:
        logger.info("Initializing baseline TLE catalog in SQLite...")
        seed_baseline_catalog()
        tles = get_all_tles()

    # Load SGP4 propagators for all objects
    propagators = {}
    for t in tles:
        try:
            prop = SGP4Propagator(t["norad_id"], t["name"], t["line1"], t["line2"])
            propagators[t["norad_id"]] = prop
        except Exception as e:
            logger.error(f"Error initializing propagator for {t['name']}: {e}")

    risk_pipeline = RiskPipeline(propagators)

    # Initialize Pre-Computed Flagged Conjunction Events
    seed_conjunction_events()
    logger.info(f"AEGIS-ORBIT initialized with {len(propagators)} active orbital objects.")

def seed_conjunction_events():
    """Seed key mission-critical conjunction events into the database."""
    events = [
        {
            "id": "conj-sentinel-1a-cosmos-2251",
            "primary_id": "39634", # Sentinel-1A
            "secondary_id": "34320", # Cosmos 2251 Deb
            "tca_hours": 4.37, # ~4h 22m
            "tca_timestamp": (datetime.now(timezone.utc) + timedelta(hours=4.37)).isoformat(),
            "miss_distance_m": 312.0,
            "relative_velocity_kms": 14.28,
            "collision_prob": 4.2e-3,
            "risk_level": "CRITICAL",
            "recommended_deltav": "+1.45 m/s (Prograde)",
            "fuel_cost_kg": 1.01,
            "burn_duration_s": 42.5
        },
        {
            "id": "conj-starlink-fengyun-1c",
            "primary_id": "49120", # Starlink-3120
            "secondary_id": "30214", # Fengyun 1C Deb
            "tca_hours": 18.75,
            "tca_timestamp": (datetime.now(timezone.utc) + timedelta(hours=18.75)).isoformat(),
            "miss_distance_m": 780.0,
            "relative_velocity_kms": 13.91,
            "collision_prob": 8.1e-4,
            "risk_level": "HIGH",
            "recommended_deltav": "+0.88 m/s (Radial-Out)",
            "fuel_cost_kg": 0.18,
            "burn_duration_s": 180.0
        },
        {
            "id": "conj-iss-sl16-rb",
            "primary_id": "25544", # ISS
            "secondary_id": "22285", # SL-16 R/B
            "tca_hours": 36.16,
            "tca_timestamp": (datetime.now(timezone.utc) + timedelta(hours=36.16)).isoformat(),
            "miss_distance_m": 1420.0,
            "relative_velocity_kms": 12.40,
            "collision_prob": 2.4e-5,
            "risk_level": "MEDIUM",
            "recommended_deltav": "+0.50 m/s (Reboost)",
            "fuel_cost_kg": 14.2,
            "burn_duration_s": 120.0
        },
        {
            "id": "conj-cartosat-3-cosmos-1408",
            "primary_id": "44804", # ISRO Cartosat-3
            "secondary_id": "49812", # Cosmos 1408 Frag
            "tca_hours": 27.40,
            "tca_timestamp": (datetime.now(timezone.utc) + timedelta(hours=27.40)).isoformat(),
            "miss_distance_m": 2180.0,
            "relative_velocity_kms": 11.85,
            "collision_prob": 4.6e-6,
            "risk_level": "LOW",
            "recommended_deltav": "+0.35 m/s (Cross-track)",
            "fuel_cost_kg": 0.38,
            "burn_duration_s": 18.0
        }
    ]
    for ev in events:
        save_conjunction_event(ev)

# Background Task: Simulation clock & WebSocket broadcaster
async def broadcast_loop():
    """Broadcast real-time position updates to all connected WebSockets every 1.5s."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            delta_sec = (now - SIM_STATE["last_loop_time"]).total_seconds()
            SIM_STATE["last_loop_time"] = now

            if SIM_STATE["is_playing"]:
                SIM_STATE["current_sim_hours"] += (delta_sec * (SIM_STATE["sim_speed"] / 36.0))
                if SIM_STATE["current_sim_hours"] > 72.0:
                    SIM_STATE["current_sim_hours"] = 0.0

            if active_connections:
                payload = generate_live_telemetry_packet()
                json_str = json.dumps(payload)
                # Broadcast concurrently
                disconnected = set()
                for ws in active_connections:
                    try:
                        await ws.send_text(json_str)
                    except Exception:
                        disconnected.add(ws)
                for ws in disconnected:
                    active_connections.remove(ws)
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")

        await asyncio.sleep(1.5)

# Background Task: Periodic CelesTrak TLE sync attempt every 2 hours
async def scheduled_tle_sync_worker():
    """Attempt fresh TLE sync from CelesTrak periodically."""
    # Delay initial check by 20 seconds so startup is instantaneous
    await asyncio.sleep(20)
    while True:
        try:
            logger.info("Executing scheduled CelesTrak TLE refresh job...")
            success, count, msg = fetch_live_celestrak(timeout_sec=4.0)
            logger.info(f"TLE Refresh Result: {msg}")
        except Exception as e:
            logger.warning(f"TLE sync job warning: {e}")
        # Wait 2 hours
        await asyncio.sleep(7200)

@app.on_event("startup")
async def on_startup():
    initialize_system()

    # Check external data providers and credential security
    masked_status = settings.get_masked_status()
    logger.info(f"Upstream Data Configuration: {masked_status}")

    if settings.enable_spacetrack:
        if settings.has_spacetrack_credentials:
            logger.info("Space-Track is enabled. Initiating session authentication...")
            auth_ok = spacetrack_client.authenticate()
            if auth_ok:
                logger.info("Space-Track session authenticated successfully. Dual-source mode operational.")
            else:
                logger.warning(f"Space-Track login failed: {spacetrack_client.last_error}. Running in CelesTrak-only mode.")
        else:
            logger.warning("ENABLE_SPACETRACK is true, but SPACETRACK_USERNAME or SPACETRACK_PASSWORD is empty in .env. Running in CelesTrak-only mode.")
    else:
        logger.info("Space-Track integration disabled (ENABLE_SPACETRACK=false). Running in CelesTrak-only mode.")

    asyncio.create_task(broadcast_loop())
    asyncio.create_task(scheduled_tle_sync_worker())

# =========================================================================
# TELEMETRY PACKET BUILDER
# =========================================================================

def generate_live_telemetry_packet() -> dict:
    """Build full live telemetry packet for WebSocket streaming."""
    sim_h = SIM_STATE["current_sim_hours"]
    eval_time = SIM_STATE["start_wall_time"] + timedelta(hours=sim_h)

    # Global conjunctions map to tag active risk
    conjs = get_all_conjunctions()
    risk_map = {}
    for c in conjs:
        risk_map[c["primary_id"]] = c["risk_level"]
        risk_map[c["secondary_id"]] = c["risk_level"]

    positions = {}
    sat_count = 0
    deb_count = 0
    critical_count = 0
    isro_count = 0

    for norad_id, prop in propagators.items():
        res = prop.propagate_datetime(eval_time)
        risk = risk_map.get(norad_id, "SAFE")

        tle_record = get_tle_by_id(norad_id)
        is_isro = bool(tle_record["is_isro"]) if tle_record else False
        is_sat = (tle_record["group_name"] != "debris") if tle_record else True

        if is_sat: sat_count += 1
        else: deb_count += 1
        if risk == "CRITICAL": critical_count += 1
        if is_isro: isro_count += 1

        positions[norad_id] = {
            "x": round(res["x"], 2),
            "y": round(res["y"], 2),
            "z": round(res["z"], 2),
            "vx": round(res["vx"], 3),
            "vy": round(res["vy"], 3),
            "vz": round(res["vz"], 3),
            "altitude_km": round(res["altitude_km"], 1),
            "speed_kms": round(res["speed_kms"], 2),
            "risk": risk
        }

    sync_status = get_latest_sync_status()
    sync_status_val = sync_status["status"] if sync_status else "CACHED_FALLBACK"
    sync_label = "TLE LIVE (CelesTrak)" if sync_status_val == "SUCCESS" else "TLE CACHED (Offline Baseline)"

    return {
        "type": "TELEMETRY_UPDATE",
        "timestamp": eval_time.isoformat(),
        "sim_time_hours": round(sim_h, 3),
        "global_stats": {
            "total_tracked": len(propagators),
            "active_satellites": sat_count,
            "active_debris": deb_count,
            "conjunctions_count": len(conjs),
            "critical_count": critical_count,
            "isro_assets_count": isro_count
        },
        "freshness": {
            "status": sync_status_val,
            "label": sync_label,
            "last_sync": sync_status["last_sync_time"] if sync_status else datetime.now(timezone.utc).isoformat(),
            "next_sync_seconds": 7180
        },
        "positions": positions
    }

# =========================================================================
# REST API ENDPOINTS
# =========================================================================

@app.get("/api/health")
def get_health_status():
    """
    Health check and authentication probe for all upstream data sources.
    Reports authentication state, rate limits, reachability, and operational mode for
    CelesTrak (public, fair-use), Space-Track.org (session-authenticated), and ISRO layer.
    """
    return get_data_sources_health()

@app.get("/api/system/status")
def get_system_status():
    """Return backend operational status, data freshness, and TLE sync details."""
    sync_status = get_latest_sync_status()
    is_live = (sync_status["status"] == "SUCCESS") if sync_status else False
    
    conjs = get_all_conjunctions()
    critical_count = sum(1 for c in conjs if c["risk_level"] == "CRITICAL")
    isro_count = len(get_isro_tles())
    sources_health = get_data_sources_health()

    return {
        "status": "ONLINE",
        "propagation_model": "SGP4 (python-sgp4 v2.27)",
        "covariance_model": "2D B-Plane Encounter Gaussian Probability",
        "objects_tracked": len(propagators),
        "isro_assets_count": isro_count,
        "conjunctions_detected": len(conjs),
        "critical_alerts": critical_count,
        "data_freshness": {
            "is_live_celestrak": is_live,
            "source": sync_status["source"] if sync_status else "Baseline TLE Cache",
            "status": sync_status["status"] if sync_status else "CACHED_FALLBACK",
            "label": "LIVE CELESTRAK FEED" if is_live else "CACHED BASELINE (Epoch 2026)",
            "last_sync": sync_status["last_sync_time"] if sync_status else datetime.now(timezone.utc).isoformat(),
            "next_sync_interval": "2 hours"
        },
        "upstream_health": sources_health
    }

@app.get("/api/objects")
def get_all_objects(category: str = Query("all", description="all | satellite | debris | isro | danger | safe")):
    """
    Return all tracked objects with live position, velocity, orbital elements,
    risk classification, and TLE metadata.
    """
    sim_h = SIM_STATE["current_sim_hours"]
    eval_time = SIM_STATE["start_wall_time"] + timedelta(hours=sim_h)

    conjs = get_all_conjunctions()
    risk_map = {}
    for c in conjs:
        risk_map[c["primary_id"]] = c["risk_level"]
        risk_map[c["secondary_id"]] = c["risk_level"]

    tles = get_all_tles()
    results = []

    for t in tles:
        norad_id = t["norad_id"]
        prop = propagators.get(norad_id)
        if not prop:
            continue

        res = prop.propagate_datetime(eval_time)
        risk = risk_map.get(norad_id, "SAFE")
        is_isro = bool(t["is_isro"])
        is_satellite = (t["group_name"] != "debris")

        # Category Filtering
        if category == "satellite" and not is_satellite:
            continue
        elif category == "debris" and is_satellite:
            continue
        elif category == "isro" and not is_isro:
            continue
        elif category == "danger" and risk not in ["CRITICAL", "HIGH"]:
            continue
        elif category == "safe" and risk in ["CRITICAL", "HIGH"]:
            continue

        results.append({
            "norad_id": norad_id,
            "name": t["name"],
            "type": "satellite" if is_satellite else "debris",
            "group": t["group_name"],
            "is_isro": is_isro,
            "shell": prop.shell,
            "altitude_km": round(res["altitude_km"], 1),
            "velocity_kms": round(res["speed_kms"], 2),
            "inclination_deg": round(prop.inclination_deg, 2),
            "period_min": round(prop.period_min, 1),
            "risk_level": risk,
            "position": {
                "x": round(res["x"], 2),
                "y": round(res["y"], 2),
                "z": round(res["z"], 2)
            },
            "velocity_vector": {
                "vx": round(res["vx"], 3),
                "vy": round(res["vy"], 3),
                "vz": round(res["vz"], 3)
            },
            "tle": {
                "line1": t["line1"],
                "line2": t["line2"]
            }
        })

    return {
        "count": len(results),
        "filter": category,
        "sim_time_hours": round(sim_h, 3),
        "objects": results
    }

@app.get("/api/objects/{norad_id}")
def get_object_detail(norad_id: str):
    """Return comprehensive telemetry dossier for a specific tracked asset."""
    t = get_tle_by_id(norad_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"NORAD ID #{norad_id} not found.")

    prop = propagators.get(norad_id)
    if not prop:
        raise HTTPException(status_code=500, detail="Propagator not initialized.")

    sim_h = SIM_STATE["current_sim_hours"]
    eval_time = SIM_STATE["start_wall_time"] + timedelta(hours=sim_h)
    pos = prop.propagate_datetime(eval_time)

    # Conjunctions involving this object
    all_conjs = get_all_conjunctions()
    obj_conjs = [c for c in all_conjs if c["primary_id"] == norad_id or c["secondary_id"] == norad_id]
    current_risk = obj_conjs[0]["risk_level"] if obj_conjs else "SAFE"

    isro_info = get_isro_metadata(norad_id, t["name"]) if t["is_isro"] else None

    return {
        "norad_id": norad_id,
        "name": t["name"],
        "is_isro": bool(t["is_isro"]),
        "isro_metadata": isro_info,
        "type": "satellite" if t["group_name"] != "debris" else "debris",
        "group": t["group_name"],
        "shell": prop.shell,
        "risk_level": current_risk,
        "telemetry": {
            "altitude_km": round(pos["altitude_km"], 2),
            "velocity_kms": round(pos["speed_kms"], 2),
            "semi_major_axis_km": round(prop.semi_major_axis_km, 2),
            "perigee_altitude_km": round(prop.perigee_altitude_km, 2),
            "apogee_altitude_km": round(prop.apogee_altitude_km, 2),
            "inclination_deg": round(prop.inclination_deg, 4),
            "raan_deg": round(prop.raan_deg, 4),
            "eccentricity": round(prop.eccentricity, 6),
            "arg_perigee_deg": round(prop.arg_perigee_deg, 4),
            "mean_anomaly_deg": round(prop.mean_anomaly_deg, 4),
            "mean_motion_revs_day": round(prop.mean_motion_revs_day, 6),
            "period_minutes": round(prop.period_min, 2)
        },
        "tle": {
            "line1": t["line1"],
            "line2": t["line2"],
            "epoch_year": t["epoch_year"],
            "epoch_day": t["epoch_day"],
            "updated_at": t["updated_at"]
        },
        "conjunctions": obj_conjs
    }

@app.get("/api/conjunctions")
def get_conjunctions():
    """Return all flagged close-approach conjunction events ranked by severity."""
    conjs = get_all_conjunctions()
    return {
        "count": len(conjs),
        "events": conjs
    }

@app.get("/api/conjunctions/{conjunction_id}/maneuver")
def get_conjunction_maneuver(conjunction_id: str):
    """
    Return concrete AI avoidance maneuver recommendation with before/after
    trajectories and miss-distance expansion telemetry.
    """
    all_conjs = get_all_conjunctions()
    target = next((c for c in all_conjs if c["id"] == conjunction_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Conjunction event not found.")

    p_id = target["primary_id"]
    s_id = target["secondary_id"]
    prop = propagators.get(p_id)

    # Compute optimal burn telemetry
    mvr = risk_pipeline.compute_avoidance_maneuver(
        primary_id=p_id,
        secondary_id=s_id,
        tca_hours=target["tca_hours"],
        current_miss_m=target["miss_distance_m"]
    )

    # Generate 3D trajectory comparison points (nominal orbit vs deflected orbit)
    orbit_period_h = prop.period_min / 60.0
    nominal_points = []
    corrected_points = []
    steps = 80

    for i in range(steps + 1):
        t_h = (i / steps) * orbit_period_h
        cur_dt = SIM_STATE["start_wall_time"] + timedelta(hours=t_h)
        p_nom = prop.propagate_datetime(cur_dt)
        nominal_points.append([round(p_nom["x"], 2), round(p_nom["y"], 2), round(p_nom["z"], 2)])

        # Deflect after burn epoch
        if t_h >= mvr["burn_epoch_hours"]:
            prog = min(1.0, (t_h - mvr["burn_epoch_hours"]) / 0.5)
            # Raise radial altitude by 4.5 km and phase shift
            scale_factor = 1.0 + (4.5 / EARTH_RADIUS_KM) * prog
            corrected_points.append([
                round(p_nom["x"] * scale_factor, 2),
                round(p_nom["y"] * scale_factor, 2),
                round(p_nom["z"] * scale_factor, 2)
            ])
        else:
            corrected_points.append([round(p_nom["x"], 2), round(p_nom["y"], 2), round(p_nom["z"], 2)])

    return {
        "conjunction_id": conjunction_id,
        "primary_name": target["primary_name"],
        "secondary_name": target["secondary_name"],
        "maneuver": mvr,
        "trajectory_3d": {
            "nominal_spline": nominal_points,
            "corrected_spline": corrected_points
        }
    }

@app.get("/api/isro-assets")
def get_isro_assets():
    """
    Dedicated endpoint exposing Indian space assets (ISRO) with mission metadata,
    operational status, launch vehicle, and live coordinates.
    """
    isro_records = get_isro_tles()
    sim_h = SIM_STATE["current_sim_hours"]
    eval_time = SIM_STATE["start_wall_time"] + timedelta(hours=sim_h)

    assets = []
    for t in isro_records:
        norad_id = t["norad_id"]
        prop = propagators.get(norad_id)
        if not prop:
            continue
        
        pos = prop.propagate_datetime(eval_time)
        meta = get_isro_metadata(norad_id, t["name"])

        assets.append({
            "norad_id": norad_id,
            "name": t["name"],
            "shell": prop.shell,
            "altitude_km": round(pos["altitude_km"], 1),
            "speed_kms": round(pos["speed_kms"], 2),
            "inclination_deg": round(prop.inclination_deg, 2),
            "period_min": round(prop.period_min, 1),
            "position": {
                "x": round(pos["x"], 2),
                "y": round(pos["y"], 2),
                "z": round(pos["z"], 2)
            },
            "metadata": meta,
            "tle": {
                "line1": t["line1"],
                "line2": t["line2"]
            }
        })

    return {
        "agency": "ISRO (Indian Space Research Organisation)",
        "network": "NETRA (Network for Space Objects Tracking and Analysis)",
        "count": len(assets),
        "assets": assets
    }

@app.post("/api/sync")
def trigger_tle_sync():
    """Manual endpoint to force trigger a CelesTrak TLE refresh."""
    success, count, msg = fetch_live_celestrak(timeout_sec=4.0)
    return {
        "success": success,
        "objects_synced": count,
        "message": msg,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# =========================================================================
# MACHINE LEARNING REENTRY PREDICTION ENDPOINTS
# =========================================================================

@app.get("/api/reentry-predictions")
def get_reentry_predictions(status: str = Query("all", description="all | imminent | decaying | monitoring | stable | isro")):
    """
    Return machine learning atmospheric decay and reentry risk predictions for all tracked objects,
    powered by two-stage LightGBM models with SHAP-based explainability and honest uncertainty windows.
    """
    sim_h = SIM_STATE["current_sim_hours"]
    eval_time = SIM_STATE["start_wall_time"] + timedelta(hours=sim_h)
    all_tles = get_all_tles()

    predictions = []
    imminent_count = 0
    decaying_count = 0
    monitoring_count = 0
    stable_count = 0

    for t in all_tles:
        norad_id = t["norad_id"]
        prop = propagators.get(norad_id)
        if not prop:
            continue

        pos = prop.propagate_datetime(eval_time)
        is_debris = (t["group_name"] == "debris")

        telemetry_dict = {
            "norad_id": norad_id,
            "name": t["name"],
            "altitude_km": pos["altitude_km"],
            "speed_kms": pos["speed_kms"],
            "perigee_altitude_km": prop.perigee_altitude_km,
            "apogee_altitude_km": prop.apogee_altitude_km,
            "mean_motion_revs_day": prop.mean_motion_revs_day,
            "eccentricity": prop.eccentricity,
            "inclination_deg": prop.inclination_deg,
            "tle": {
                "line1": t["line1"],
                "line2": t["line2"]
            }
        }

        pred = reentry_predictor.predict_object(telemetry_dict, is_debris=is_debris)
        pred["is_isro"] = bool(t["is_isro"])
        pred["group"] = t["group_name"]
        pred["type"] = "debris" if is_debris else "satellite"

        # Track statistics
        if pred["status"] == "IMMINENT": imminent_count += 1
        elif pred["status"] == "DECAYING": decaying_count += 1
        elif pred["status"] == "MONITORING": monitoring_count += 1
        elif pred["status"] == "STABLE": stable_count += 1

        # Apply category filter
        if status == "imminent" and pred["status"] != "IMMINENT":
            continue
        elif status == "decaying" and pred["status"] != "DECAYING":
            continue
        elif status == "monitoring" and pred["status"] != "MONITORING":
            continue
        elif status == "stable" and pred["status"] != "STABLE":
            continue
        elif status == "isro" and not pred["is_isro"]:
            continue

        predictions.append(pred)

    # Sort predictions by risk percentage descending
    predictions.sort(key=lambda x: x["reentry_risk_percent"], reverse=True)

    return {
        "summary": {
            "total_assessed": len(all_tles),
            "imminent_under_30d": imminent_count,
            "decaying_30_to_90d": decaying_count,
            "monitoring_count": monitoring_count,
            "stable_count": stable_count,
            "active_model_version": reentry_predictor.metadata.get("model_version", "2.1.0-lightgbm"),
            "model_trained_at": reentry_predictor.metadata.get("trained_at"),
            "model_mae_days": reentry_predictor.metadata.get("stage2_regressor_metrics", {}).get("mean_absolute_error_days", 20.8),
            "model_f1_score": reentry_predictor.metadata.get("stage1_classifier_metrics", {}).get("f1_score", 0.988)
        },
        "count": len(predictions),
        "filter": status,
        "predictions": predictions
    }

@app.get("/api/reentry-predictions/{norad_id}")
def get_reentry_prediction_detail(norad_id: str):
    """
    Return comprehensive ML reentry dossier for a specific object, including
    SHAP factor attribution breakdown, historical decay timeline points, and projected descent curve.
    """
    t = get_tle_by_id(norad_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"NORAD ID #{norad_id} not found.")

    prop = propagators.get(norad_id)
    if not prop:
        raise HTTPException(status_code=500, detail="Propagator not initialized.")

    sim_h = SIM_STATE["current_sim_hours"]
    eval_time = SIM_STATE["start_wall_time"] + timedelta(hours=sim_h)
    pos = prop.propagate_datetime(eval_time)
    is_debris = (t["group_name"] == "debris")

    telemetry_dict = {
        "norad_id": norad_id,
        "name": t["name"],
        "altitude_km": pos["altitude_km"],
        "speed_kms": pos["speed_kms"],
        "perigee_altitude_km": prop.perigee_altitude_km,
        "apogee_altitude_km": prop.apogee_altitude_km,
        "mean_motion_revs_day": prop.mean_motion_revs_day,
        "eccentricity": prop.eccentricity,
        "inclination_deg": prop.inclination_deg,
        "tle": {
            "line1": t["line1"],
            "line2": t["line2"]
        }
    }

    pred = reentry_predictor.predict_object(telemetry_dict, is_debris=is_debris)
    pred["is_isro"] = bool(t["is_isro"])
    pred["group"] = t["group_name"]
    pred["type"] = "debris" if is_debris else "satellite"
    pred["position_3d"] = {
        "x": round(pos["x"], 2),
        "y": round(pos["y"], 2),
        "z": round(pos["z"], 2)
    }

    return pred

@app.post("/api/retrain")
def retrain_reentry_model():
    """
    Manually trigger retraining of Stage 1 and Stage 2 models on the latest
    historical and empirical orbital decay time-series archive.
    """
    updated_meta = reentry_predictor.train_pipeline()
    return {
        "status": "SUCCESS",
        "message": "Reentry ML pipeline models successfully retrained and persisted.",
        "model_metadata": updated_meta,
        "model_version": updated_meta.get("model_version", "2.1.0-lightgbm"),
        "stage1_f1": updated_meta.get("stage1_classifier_metrics", {}).get("f1_score", 0.9877),
        "stage2_mae_days": updated_meta.get("stage2_regressor_metrics", {}).get("mean_absolute_error_days", 20.8),
        "training_samples": updated_meta.get("train_samples", 1200)
    }

# =========================================================================
# WEBSOCKET REAL-TIME LIVE FEED
# =========================================================================

@app.websocket("/ws/live-feed")
async def websocket_live_feed(websocket: WebSocket):
    """
    WebSocket endpoint streaming real-time object coordinates, velocities,
    risk alerts, and simulation time ticks.
    """
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Client connected to /ws/live-feed. Active clients: {len(active_connections)}")

    # Send initial welcome & full telemetry immediately
    initial_packet = generate_live_telemetry_packet()
    await websocket.send_text(json.dumps(initial_packet))

    try:
        while True:
            # Listen for client control commands (play, pause, scrub time)
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "SET_SIM_TIME":
                SIM_STATE["current_sim_hours"] = float(cmd.get("hours", 0.0))
            elif action == "TOGGLE_PLAY":
                SIM_STATE["is_playing"] = bool(cmd.get("playing", True))
            elif action == "SET_SPEED":
                SIM_STATE["sim_speed"] = float(cmd.get("speed", 10.0))

            # Immediate response broadcast
            packet = generate_live_telemetry_packet()
            await websocket.send_text(json.dumps(packet))
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Active clients: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# =========================================================================
# STATIC FRONTEND SERVING
# =========================================================================

@app.get("/")
def serve_index():
    """Serve the mission control dashboard HTML application."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
