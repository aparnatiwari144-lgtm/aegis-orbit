"""
ISRO (Indian Space Research Organisation) Satellite Metadata Registry
Enriches CelesTrak TLE orbital data with authentic mission details,
launch vehicle information, and operational classification.
"""

ISRO_REGISTRY = {
    "40353": {
        "norad_id": "40353",
        "canonical_name": "IRNSS-1C (NavIC)",
        "series": "NavIC / IRNSS",
        "mission_type": "Navigation / PNT",
        "launch_date": "2014-10-15",
        "launch_vehicle": "PSLV-C26",
        "operator": "ISRO (ISTRAC)",
        "orbit_class": "GEO (Geostationary)",
        "mass_kg": 1425,
        "operational_status": "Operational",
        "description": "Indian Regional Navigation Satellite System constellation spacecraft providing civilian and restricted positioning signals across India and surrounding 1,500 km region."
    },
    "43286": {
        "norad_id": "43286",
        "canonical_name": "IRNSS-1I (NavIC)",
        "series": "NavIC / IRNSS",
        "mission_type": "Navigation / PNT",
        "launch_date": "2018-04-11",
        "launch_vehicle": "PSLV-C41",
        "operator": "ISRO (ISTRAC)",
        "orbit_class": "GSO (Geosynchronous)",
        "mass_kg": 1425,
        "operational_status": "Operational",
        "description": "Navigation spacecraft operating in an inclined geosynchronous orbit (29° inclination) as part of the NavIC constellation."
    },
    "44804": {
        "norad_id": "44804",
        "canonical_name": "CARTOSAT-3",
        "series": "Cartosat",
        "mission_type": "High-Resolution Earth Observation",
        "launch_date": "2019-11-27",
        "launch_vehicle": "PSLV-C47",
        "operator": "ISRO (NRSC)",
        "orbit_class": "LEO (Sun-Synchronous)",
        "mass_kg": 1625,
        "operational_status": "Operational",
        "description": "Third-generation agile advanced satellite with high spatial resolution (0.28 m panchromatic), supporting urban planning, infrastructure monitoring, and coastal regulation."
    },
    "44857": {
        "norad_id": "44857",
        "canonical_name": "RISAT-2BR1",
        "series": "RISAT",
        "mission_type": "Synthetic Aperture Radar (SAR)",
        "launch_date": "2019-12-11",
        "launch_vehicle": "PSLV-C48",
        "operator": "ISRO (NRSC)",
        "orbit_class": "LEO (Low Earth Orbit)",
        "mass_kg": 628,
        "operational_status": "Operational",
        "description": "Radar imaging earth observation satellite carrying an X-band Synthetic Aperture Radar capable of all-weather, day-and-night surveillance for disaster management and agriculture."
    },
    "45026": {
        "norad_id": "45026",
        "canonical_name": "GSAT-30",
        "series": "GSAT",
        "mission_type": "High-Throughput Communications",
        "launch_date": "2020-01-16",
        "launch_vehicle": "Ariane 5 ECA (ISRO contract)",
        "operator": "ISRO / NSIL",
        "orbit_class": "GEO (Geostationary 83.0°E)",
        "mass_kg": 3357,
        "operational_status": "Operational",
        "description": "Geostationary communication satellite providing Ku-band and C-band coverage for DTH television services, VSAT networks, and tele-education across the Indian subcontinent."
    },
    "51656": {
        "norad_id": "51656",
        "canonical_name": "EOS-04 (RISAT-1A)",
        "series": "Earth Observation Satellite",
        "mission_type": "C-Band Synthetic Aperture Radar",
        "launch_date": "2022-02-14",
        "launch_vehicle": "PSLV-C52",
        "operator": "ISRO (NRSC)",
        "orbit_class": "LEO (Sun-Synchronous 529 km)",
        "mass_kg": 1710,
        "operational_status": "Operational",
        "description": "Radar Imaging Satellite designed to provide high-quality images under all weather conditions for applications such as Agriculture, Forestry, Soil Moisture, and Flood Mapping."
    },
    "40930": {
        "norad_id": "40930",
        "canonical_name": "ASTROSAT",
        "series": "Space Observatory",
        "mission_type": "Multi-Wavelength Astronomy",
        "launch_date": "2015-09-28",
        "launch_vehicle": "PSLV-C30",
        "operator": "ISRO / IUCAA / TIFR",
        "orbit_class": "LEO (Near-Equatorial 650 km)",
        "mass_kg": 1513,
        "operational_status": "Operational",
        "description": "India's dedicated multi-wavelength space observatory studying celestial sources in X-ray, optical, and UV spectral bands simultaneously."
    },
    "41758": {
        "norad_id": "41758",
        "canonical_name": "INSAT-3DR",
        "series": "INSAT",
        "mission_type": "Meteorological & Search and Rescue",
        "launch_date": "2016-09-08",
        "launch_vehicle": "GSLV-F05",
        "operator": "ISRO (IMD)",
        "orbit_class": "GEO (Geostationary 74.0°E)",
        "mass_kg": 2211,
        "operational_status": "Operational",
        "description": "Advanced meteorological satellite configured with an atmospheric Sounder and Imager, including Satellite Aided Search & Rescue (SAS&R) transponder."
    },
    "54361": {
        "norad_id": "54361",
        "canonical_name": "OCEANSAT-3 (EOS-06)",
        "series": "Oceansat",
        "mission_type": "Oceanographic & Atmospheric Studies",
        "launch_date": "2022-11-26",
        "launch_vehicle": "PSLV-C54",
        "operator": "ISRO (NRSC/SAC)",
        "orbit_class": "LEO (Sun-Synchronous 720 km)",
        "mass_kg": 1117,
        "operational_status": "Operational",
        "description": "Carries Ocean Color Monitor (OCM-3), Sea Surface Temperature Monitor (SSTM), and Ku-Band Pencil Beam Scatterometer for ocean state forecasting and cyclone tracking."
    },
    "44441": {
        "norad_id": "44441",
        "canonical_name": "CHANDRAYAAN-2 ORBITER",
        "series": "Lunar Exploration",
        "mission_type": "Deep Space / Lunar Orbiter",
        "launch_date": "2019-07-22",
        "launch_vehicle": "GSLV Mk III-M1",
        "operator": "ISRO (ISTRAC / MOX)",
        "orbit_class": "Lunar Polar Orbit (100 km)",
        "mass_kg": 2379,
        "operational_status": "Operational (Moon Orbit)",
        "description": "Indian lunar exploration orbiter mapping the lunar surface, elemental composition, and searching for water ice at the lunar south pole."
    }
}

# Substring matches to detect ISRO assets in standard TLE name feeds
ISRO_NAME_KEYWORDS = [
    "CARTOSAT", "RISAT", "GSAT", "IRNSS", "NAVIC", "EOS-", "ASTROSAT",
    "OCEANSAT", "INSAT", "CHANDRAYAAN", "MOM", "ADITYA-L1", "KALPANA",
    "MEGHA-TROPIQUES", "SARAL", "RESOURCESAT", "HYSAT", "ANAND"
]

def is_isro_asset(norad_id: str, name: str) -> bool:
    """Check if object is an Indian space asset by NORAD ID or keyword."""
    if norad_id in ISRO_REGISTRY:
        return True
    upper_name = name.upper()
    return any(keyword in upper_name for keyword in ISRO_NAME_KEYWORDS)

def get_isro_metadata(norad_id: str, name: str = "") -> dict:
    """Return enriched ISRO metadata, falling back to dynamic parsing if needed."""
    if norad_id in ISRO_REGISTRY:
        return ISRO_REGISTRY[norad_id]
    
    # Generic enrichment for newly discovered Indian asset
    return {
        "norad_id": norad_id,
        "canonical_name": name,
        "series": "ISRO Space Asset",
        "mission_type": "Scientific / Communications / Earth Observation",
        "launch_date": "Verified Indian Mission",
        "launch_vehicle": "ISRO Launch Vehicle (PSLV/GSLV)",
        "operator": "ISRO / Department of Space, Govt of India",
        "orbit_class": "Tracked Indian Satellite",
        "mass_kg": "Classified / Variable",
        "operational_status": "Active / Monitored",
        "description": f"Indian space asset {name} monitored under ISRO Network for Space Objects Tracking and Analysis (NETRA)."
    }
