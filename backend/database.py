"""
Database Layer: SQLite schema for TLE catalog, sync history, and conjunction events.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aegis_orbit.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # TLE records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tle_records (
            norad_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            line1 TEXT NOT NULL,
            line2 TEXT NOT NULL,
            group_name TEXT NOT NULL,
            is_isro INTEGER DEFAULT 0,
            epoch_year INTEGER,
            epoch_day REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Flagged conjunction events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conjunction_events (
            id TEXT PRIMARY KEY,
            primary_id TEXT NOT NULL,
            secondary_id TEXT NOT NULL,
            tca_hours REAL NOT NULL,
            tca_timestamp TEXT NOT NULL,
            miss_distance_m REAL NOT NULL,
            relative_velocity_kms REAL NOT NULL,
            collision_prob REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommended_deltav TEXT,
            fuel_cost_kg REAL,
            burn_duration_s REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(primary_id) REFERENCES tle_records(norad_id),
            FOREIGN KEY(secondary_id) REFERENCES tle_records(norad_id)
        )
    """)

    # TLE Sync logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            details TEXT,
            last_sync_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            next_sync_time TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_tle(norad_id: str, name: str, line1: str, line2: str, group_name: str, is_isro: bool = False):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Parse epoch from line 1 (columns 19-32)
    try:
        epoch_str = line1[18:32].strip()
        epoch_year = int(epoch_str[:2])
        epoch_day = float(epoch_str[2:])
    except Exception:
        epoch_year = 26
        epoch_day = 0.0

    cursor.execute("""
        INSERT INTO tle_records (norad_id, name, line1, line2, group_name, is_isro, epoch_year, epoch_day, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(norad_id) DO UPDATE SET
            name = excluded.name,
            line1 = excluded.line1,
            line2 = excluded.line2,
            group_name = excluded.group_name,
            is_isro = excluded.is_isro,
            epoch_year = excluded.epoch_year,
            epoch_day = excluded.epoch_day,
            updated_at = excluded.updated_at
    """, (
        norad_id,
        name.strip(),
        line1.strip(),
        line2.strip(),
        group_name,
        1 if is_isro else 0,
        epoch_year,
        epoch_day,
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()

def get_all_tles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tle_records ORDER BY is_isro DESC, name ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_tle_by_id(norad_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tle_records WHERE norad_id = ?", (norad_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_isro_tles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tle_records WHERE is_isro = 1 ORDER BY name ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def save_conjunction_event(event_data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conjunction_events (
            id, primary_id, secondary_id, tca_hours, tca_timestamp,
            miss_distance_m, relative_velocity_kms, collision_prob,
            risk_level, recommended_deltav, fuel_cost_kg, burn_duration_s, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            tca_hours = excluded.tca_hours,
            tca_timestamp = excluded.tca_timestamp,
            miss_distance_m = excluded.miss_distance_m,
            relative_velocity_kms = excluded.relative_velocity_kms,
            collision_prob = excluded.collision_prob,
            risk_level = excluded.risk_level,
            recommended_deltav = excluded.recommended_deltav,
            fuel_cost_kg = excluded.fuel_cost_kg,
            burn_duration_s = excluded.burn_duration_s,
            created_at = excluded.created_at
    """, (
        event_data["id"],
        event_data["primary_id"],
        event_data["secondary_id"],
        event_data["tca_hours"],
        event_data.get("tca_timestamp", ""),
        event_data["miss_distance_m"],
        event_data["relative_velocity_kms"],
        event_data["collision_prob"],
        event_data["risk_level"],
        event_data.get("recommended_deltav", ""),
        event_data.get("fuel_cost_kg", 0.0),
        event_data.get("burn_duration_s", 0.0),
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()

def get_all_conjunctions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*, 
               p.name as primary_name, p.group_name as primary_group, p.is_isro as primary_isro,
               s.name as secondary_name, s.group_name as secondary_group, s.is_isro as secondary_isro
        FROM conjunction_events c
        JOIN tle_records p ON c.primary_id = p.norad_id
        JOIN tle_records s ON c.secondary_id = s.norad_id
        ORDER BY 
            CASE c.risk_level
                WHEN 'CRITICAL' THEN 1
                WHEN 'HIGH' THEN 2
                WHEN 'MEDIUM' THEN 3
                WHEN 'LOW' THEN 4
                ELSE 5
            END,
            c.miss_distance_m ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def log_sync(source: str, status: str, count: int, details: str = "", next_hours: float = 2.0):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc)
    next_time = datetime.fromtimestamp(now.timestamp() + next_hours * 3600, tz=timezone.utc)
    cursor.execute("""
        INSERT INTO sync_logs (source, status, object_count, details, last_sync_time, next_sync_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        source,
        status,
        count,
        details,
        now.isoformat(),
        next_time.isoformat()
    ))
    conn.commit()
    conn.close()

def get_latest_sync_status():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sync_logs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
