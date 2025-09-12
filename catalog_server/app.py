import requests
import uvicorn
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Astropy for coordinate handling and transformations
# get_body requires the 'jplephem' package to be installed for ephemeris data
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u

# --- Configuration ---
# This is the API that directly controls the telescope hardware
TELESCOPE_API_HOSTNAME = os.environ.get("TELESCOPE_API_HOSTNAME", "localhost")
TELESCOPE_API_URL = f"http://{TELESCOPE_API_HOSTNAME}:8001"
TELESCOPE_SLEW_ENDPOINT = f"{TELESCOPE_API_URL}/slew/radec"
TELESCOPE_HEALTH_ENDPOINT = f"{TELESCOPE_API_URL}/healthz"

# Default location (Castro Valley, CA) if not provided in the API call
OBSERVER_LAT_DEFAULT = 37.7 * u.deg
OBSERVER_LON_DEFAULT = -122.1 * u.deg
OBSERVER_ELEVATION_DEFAULT = 23 * u.m
# Define the minimum altitude for an object to be considered "visible"
MINIMUM_ALTITUDE_DEG = 15.0
# List of solar system bodies that need special real-time calculation
SOLAR_SYSTEM_BODIES = [
    'sun', 'moon', 'mercury', 'venus', 'mars','luna',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'io', 'europa', 'ganymede', 'callisto', # Jupiter's moons
    'titan', 'rhea', 'iapetus', 'dione'     # Saturn's moons
]

# --- FastAPI Application ---
app = FastAPI(
    title="Celestial Object Resolver API",
    description="Resolves common object names to coordinates and slews a telescope to them."
)

# --- Pydantic Models ---
class SlewRequest(BaseModel):
    object_name: str = Field(
        ...,
        example="Titan",
        description="The common name of the celestial object (e.g., 'M42', 'Titan', 'Mars')."
    )
    latitude: Optional[float] = Field(
        None,
        example=37.7,
        description="Observer's latitude in decimal degrees. Defaults to Castro Valley, CA if not provided."
    )
    longitude: Optional[float] = Field(
        None,
        example=-122.1,
        description="Observer's longitude in decimal degrees. Defaults to Castro Valley, CA if not provided."
    )
    elevation: Optional[float] = Field(
        None,
        example=125.0,
        description="Observer's elevation in meters. Defaults to 0 if lat/lon are provided but elevation is not."
    )
class HealthStatus(BaseModel):
    resolver_api_status: str
    #telescope_api_status: str
    #details: Optional[str] = None

class SlewResponse(BaseModel):
    message: str
    object_name: str
    resolved_ra: str
    resolved_dec: str
    current_altitude: float
    current_azimuth: float
    slew_command_sent: bool

# --- API Endpoint ---
@app.get("/healthz", response_model=HealthStatus, summary="Check service health")
def health_check():
    # """
    # Checks the status of this service and its dependency (the Telescope API).
    # """
    # telescope_status = "unhealthy"
    # details = "Could not connect to the telescope API."
    # try:
    #     response = requests.get(TELESCOPE_HEALTH_ENDPOINT, timeout=5)
    #     if response.status_code == 200:
    #         telescope_status = "ok"
    #         details = response.json()
    #     else:
    #         details = f"Telescope API returned status code {response.status_code}."
    # except requests.exceptions.RequestException as e:
    #     details = str(e)

    return {
        "resolver_api_status": "ok",
        # "telescope_api_status": telescope_status,
        # "details": details
    }

@app.post("/slew-by-name", response_model=SlewResponse, summary="Resolve object name and slew telescope")
def slew_by_name(slew_request: SlewRequest):
    """
    Takes an object's common name, resolves it to RA/Dec, checks if it's
    currently above the horizon for a given location, and sends a slew command
    to the telescope API.
    """
    start_time = time.monotonic()

    object_name = slew_request.object_name
    print(f"Received request to slew to: {object_name}")

    # 1. Determine observer location from request or use defaults
    if slew_request.latitude is not None and slew_request.longitude is not None:
        obs_lat = slew_request.latitude * u.deg
        obs_lon = slew_request.longitude * u.deg
        # Default elevation to 0 if not provided alongside lat/lon
        obs_elev = (slew_request.elevation or 0.0) * u.m
        print(f"Using provided observer location: Lat {obs_lat}, Lon {obs_lon}, Elev {obs_elev}")
    else:
        obs_lat = OBSERVER_LAT_DEFAULT
        obs_lon = OBSERVER_LON_DEFAULT
        obs_elev = OBSERVER_ELEVATION_DEFAULT
        print("Using default observer location (Greenwich, UK)")

    observer_location = EarthLocation(lat=obs_lat, lon=obs_lon, height=obs_elev)
    current_time = Time.now()

    # 2. Resolve the object name to celestial coordinates
    try:
        if object_name.lower() in SOLAR_SYSTEM_BODIES:
            print(f"'{object_name}' is a solar system body. Calculating its current position...")
            target_coords = get_body(object_name, current_time, observer_location)
        else:
            print(f"'{object_name}' is a deep-sky object. Querying database...")
            target_coords = SkyCoord.from_name(object_name)
    except Exception as e:
        print(f"Error resolving name '{object_name}': {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Could not find celestial object named '{object_name}'. Make sure 'jplephem' is installed for solar system bodies."
        )
    resolve_time = time.monotonic()

    # 3. Check if the object is currently visible from the observer's location
    altaz_frame = AltAz(obstime=current_time, location=observer_location)
    target_altaz = target_coords.transform_to(altaz_frame)
    current_altitude = target_altaz.alt.degree
    current_azimuth = target_altaz.az.degree

    print(f"'{object_name}' is currently at Altitude: {current_altitude:.2f}°, Azimuth: {current_azimuth:.2f}°")

    if current_altitude < MINIMUM_ALTITUDE_DEG:
        raise HTTPException(
            status_code=400,
            detail=f"Object '{object_name}' is currently below the horizon or too low to observe (Altitude: {current_altitude:.2f}°). Minimum required altitude is {MINIMUM_ALTITUDE_DEG}°."
        )
    visibility_time = time.monotonic()

    # Format RA/Dec strings to ultra precision for the telescope mount.
    ra_precision = 2  # HH:MM:SS.SS
    dec_precision = 1  # sDD:MM:SS.S
    ra_str = target_coords.ra.to_string(unit=u.hourangle, sep=':', precision=ra_precision, pad=True)
    dec_str = target_coords.dec.to_string(unit=u.degree, sep=':', precision=dec_precision, alwayssign=True)
    print("Using ULTRA precision for coordinates.")

    print(f"Formatted coordinates for telescope API -> RA: {ra_str}, Dec: {dec_str}")

    # 5. Send the command to the telescope control API
    try:
        payload = {"ra": ra_str, "dec": dec_str}
        response = requests.post(TELESCOPE_SLEW_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the telescope API: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not send slew command. Ensure the telescope API is running at {TELESCOPE_API_URL}."
        )
    slew_api_time = time.monotonic()
    timing_data = {
        "resolve_object_name": round((resolve_time - start_time) * 1000, 2),
        "check_visibility": round((visibility_time - resolve_time) * 1000, 2),
        "call_slew_api": round((slew_api_time - visibility_time) * 1000, 2),
        "total_duration": round((slew_api_time - start_time) * 1000, 2)
    }
    print(timing_data)
    return {
        "message": f"Object '{object_name}' is visible. Slew command sent successfully.",
        "object_name": object_name,
        "resolved_ra": ra_str,
        "resolved_dec": dec_str,
        "current_altitude": round(current_altitude, 2),
        "current_azimuth": round(current_azimuth, 2),
        "slew_command_sent": True
    }

# --- To run this API ---
# 1. Install requirements: pip install -r requirements.txt
# 2. Make sure the telescope_api.py service is running on port 8001.
# 3. Run this file: uvicorn object_resolver_api:app --reload --port 8002
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

