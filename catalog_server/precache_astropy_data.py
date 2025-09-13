# This script is designed to be run during a Docker build process.
# It forces Astropy to download and cache data that would otherwise be
# fetched on the first run, causing significant "cold start" delays.

from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import get_body, SkyCoord, EarthLocation, AltAz
import astropy.units as u


print("--- Starting Astropy data pre-caching ---")

# --- 1. Cache IERS Data ---
# This data is needed for high-precision time and coordinate transformations.
# It accounts for the Earth's irregular rotation.
try:
    print("Attempting to cache IERS-A data for Earth orientation...")
    iers.IERS_A.open()
    print("Successfully downloaded and cached IERS-A data.")
except Exception as e:
    print(f"An error occurred while pre-caching IERS data: {e}")
    # We don't want to fail the build if the download server is temporarily down,
    # but we should be aware of it. The app will try again at runtime if needed.
    pass

# --- 2. Cache Solar System Ephemeris Data ---
# This data (JPL ephemerides, e.g., de430.bsp) is needed by get_body() to
# calculate the positions of planets, moons, etc.
try:
    print("Attempting to cache solar system ephemeris data (jplephem)...")
    # To pre-cache the ephemeris file, we simply call get_body for any
    # solar system object. This triggers the download mechanism.
    dummy_time = Time("2024-01-01")
    get_body('jupiter', dummy_time)
    print("Successfully downloaded and cached solar system ephemeris data.")
except Exception as e:
    print(f"An error occurred while pre-caching solar system ephemeris: {e}")
    pass

# --- 3. Force-initialize the coordinate transformation engine ---
# By performing a dummy transformation, we ensure that all necessary
# calculations and data (like the IERS tables) are not just downloaded
# but also fully loaded and parsed. This prevents a final startup delay.
try:
    print("Performing a dummy coordinate transformation to initialize all components...")
    dummy_coord = SkyCoord(0, 0, unit='deg', frame='icrs')
    dummy_location = EarthLocation(lat=0*u.deg, lon=0*u.deg, height=0*u.m)
    dummy_time = Time.now()
    dummy_coord.transform_to(AltAz(obstime=dummy_time, location=dummy_location))
    print("Successfully initialized coordinate transformation engine.")
except Exception as e:
    print(f"An error occurred during dummy transformation: {e}")
    pass


print("--- Astropy pre-caching finished ---")

