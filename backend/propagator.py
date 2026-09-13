"""
SGP4 Orbital Propagation Engine
Uses official sgp4 library to compute positions (x,y,z in km), velocities (vx,vy,vz in km/s),
and Keplerian orbital parameters across arbitrary simulation time.
"""

import math
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
import numpy as np

EARTH_RADIUS_KM = 6378.137
MU_EARTH = 398600.4418 # km^3 / s^2

class SGP4Propagator:
    def __init__(self, norad_id: str, name: str, line1: str, line2: str):
        self.norad_id = norad_id
        self.name = name
        self.line1 = line1
        self.line2 = line2
        self.sat = Satrec.twoline2rv(line1, line2)
        self.parse_orbital_elements()

    def parse_orbital_elements(self):
        """Parse core orbital elements directly from Line 2 of TLE."""
        try:
            self.inclination_deg = float(self.line2[8:16].strip())
            self.raan_deg = float(self.line2[17:25].strip())
            self.eccentricity = float("0." + self.line2[26:33].strip())
            self.arg_perigee_deg = float(self.line2[34:42].strip())
            self.mean_anomaly_deg = float(self.line2[43:51].strip())
            self.mean_motion_revs_day = float(self.line2[52:63].strip())
            self.period_min = 1440.0 / max(0.0001, self.mean_motion_revs_day)

            # Semi-major axis from mean motion: n in rad/s = n_revs_day * 2pi / 86400
            n_rad_s = self.mean_motion_revs_day * (2 * math.pi / 86400.0)
            self.semi_major_axis_km = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)

            # Apogee & Perigee
            self.perigee_altitude_km = self.semi_major_axis_km * (1.0 - self.eccentricity) - EARTH_RADIUS_KM
            self.apogee_altitude_km = self.semi_major_axis_km * (1.0 + self.eccentricity) - EARTH_RADIUS_KM
            self.mean_altitude_km = (self.perigee_altitude_km + self.apogee_altitude_km) / 2.0

            # Shell classification
            if self.mean_altitude_km < 2000.0:
                self.shell = "LEO"
            elif self.mean_altitude_km < 35000.0:
                self.shell = "MEO"
            elif abs(self.mean_altitude_km - 35786.0) < 2000.0 and self.inclination_deg < 25.0:
                self.shell = "GEO"
            else:
                self.shell = "HEO"
        except Exception as e:
            # Fallback for irregular TLE formats
            self.inclination_deg = 0.0
            self.raan_deg = 0.0
            self.eccentricity = 0.0
            self.arg_perigee_deg = 0.0
            self.mean_anomaly_deg = 0.0
            self.mean_motion_revs_day = 14.5
            self.period_min = 99.0
            self.semi_major_axis_km = 7000.0
            self.perigee_altitude_km = 600.0
            self.apogee_altitude_km = 600.0
            self.mean_altitude_km = 600.0
            self.shell = "LEO"

    def propagate_minutes(self, tsince_minutes: float):
        """Propagate position and velocity using SGP4 minutes since epoch."""
        e, r, v = self.sat.sgp4_tsince(tsince_minutes)
        if e != 0:
            # SGP4 error fallback (e.g. decayed orbit or numerical divergence)
            # Use Keplerian analytical fallback
            r, v = self._keplerian_fallback(tsince_minutes)
        
        pos_km = np.array(r, dtype=float)
        vel_kms = np.array(v, dtype=float)
        dist_from_center = float(np.linalg.norm(pos_km))
        altitude_km = dist_from_center - EARTH_RADIUS_KM
        speed_kms = float(np.linalg.norm(vel_kms))

        return {
            "pos": pos_km,
            "vel": vel_kms,
            "x": float(pos_km[0]),
            "y": float(pos_km[1]),
            "z": float(pos_km[2]),
            "vx": float(vel_kms[0]),
            "vy": float(vel_kms[1]),
            "vz": float(vel_kms[2]),
            "altitude_km": altitude_km,
            "speed_kms": speed_kms,
            "error_code": e
        }

    def propagate_datetime(self, dt: datetime):
        """Propagate using standard UTC datetime object."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
        e, r, v = self.sat.sgp4(jd, fr)
        if e != 0:
            # Fallback
            minutes_since_epoch = (dt.timestamp() - self.get_epoch_datetime().timestamp()) / 60.0
            r, v = self._keplerian_fallback(minutes_since_epoch)

        pos_km = np.array(r, dtype=float)
        vel_kms = np.array(v, dtype=float)
        dist_from_center = float(np.linalg.norm(pos_km))
        altitude_km = dist_from_center - EARTH_RADIUS_KM
        speed_kms = float(np.linalg.norm(vel_kms))

        return {
            "pos": pos_km,
            "vel": vel_kms,
            "x": float(pos_km[0]),
            "y": float(pos_km[1]),
            "z": float(pos_km[2]),
            "vx": float(vel_kms[0]),
            "vy": float(vel_kms[1]),
            "vz": float(vel_kms[2]),
            "altitude_km": altitude_km,
            "speed_kms": speed_kms,
            "error_code": e
        }

    def get_epoch_datetime(self) -> datetime:
        """Extract epoch datetime from Satrec object."""
        try:
            # sat.jdsatepoch is Julian day of epoch
            jd = self.sat.jdsatepoch + self.sat.jdsatepochF
            # Convert Julian date to timestamp
            # J2000 = 2451545.0 = 2000-01-01 12:00:00 UTC
            days_since_j2000 = jd - 2451545.0
            timestamp = 946728000.0 + days_since_j2000 * 86400.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _keplerian_fallback(self, tsince_min: float):
        """Analytical Keplerian fallback if SGP4 has numerical singularity."""
        inc = math.radians(self.inclination_deg)
        raan = math.radians(self.raan_deg)
        argp = math.radians(self.arg_perigee_deg)
        
        # Mean anomaly at time
        m_dot = (self.mean_motion_revs_day * 2 * math.pi) / 1440.0 # rad/min
        m = math.radians(self.mean_anomaly_deg) + m_dot * tsince_min
        u = m + argp # true anomaly approx for near-circular orbits

        r = self.semi_major_axis_km * (1.0 - self.eccentricity * math.cos(m))
        xp = r * math.cos(u)
        yp = r * math.sin(u)

        # Coordinate transformation to TEME
        x = xp * math.cos(raan) - yp * math.cos(inc) * math.sin(raan)
        y = yp * math.sin(inc)
        z = xp * math.sin(raan) + yp * math.cos(inc) * math.cos(raan)

        v_mag = math.sqrt(MU_EARTH / max(100.0, r))
        vx = -v_mag * math.sin(u)
        vy = v_mag * math.cos(u) * math.cos(inc)
        vz = v_mag * math.cos(u) * math.sin(inc)

        return [x, y, z], [vx, vy, vz]
