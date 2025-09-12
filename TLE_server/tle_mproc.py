# tle_server.py  (SGP4 + parallel compute)
import os, time, math
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from sgp4.api import Satrec, jday
import re

def jd_to_datetime_utc(jd_full: float) -> datetime:
    # Convert (astronomical) Julian Date to UTC datetime, no 12h bias.
    Z = int(jd_full + 0.5)
    F = (jd_full + 0.5) - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    day_int = int(day)
    frac = day - day_int
    hour = int(frac * 24)
    minute = int((frac * 24 - hour) * 60)
    sec_float = frac * 86400 - (hour * 3600 + minute * 60)
    second = int(sec_float)
    microsecond = int(round((sec_float - second) * 1e6))
    # Handle rounding to 60.000000s
    if microsecond >= 1_000_000:
        microsecond -= 1_000_000
        second += 1
    return datetime(year, month, day_int, hour, minute, second, microsecond, tzinfo=timezone.utc)

# ---------- Config ----------
DEFAULT_TLE_URL = os.environ.get("TLE_URL","https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS","7200"))
VERBOSE = os.environ.get("VERBOSE","1") != "0"
PRINT_PROGRESS_EVERY = int(os.environ.get("PRINT_PROGRESS_EVERY","300"))
SAT_NAME_FILTER = os.environ.get("SAT_NAME_FILTER","").strip()
POOL_WORKERS = int(os.environ.get("POOL_WORKERS", str(min(32, cpu_count()))))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))   # satellites per task

# Earth constants
WGS84_A_KM = 6378.137; WGS84_F = 1.0/298.257223563
WGS84_E2 = WGS84_F*(2-WGS84_F); EARTH_MEAN_RADIUS_KM = 6371.0
OMEGA_EARTH = 7.2921150e-5

# ---------- App / cache / pool ----------
app = FastAPI(title="Overhead (SGP4) API — parallel", version="1.3")
_tle_cache: Dict[str, Any] = {"fetched_at":0.0, "satellites":[]}
_pool = ProcessPoolExecutor(max_workers=POOL_WORKERS)

# ---------- Models ----------
class SatOut(BaseModel):
    name: str; norad_id: int
    az_deg: float; el_deg: float
    range_km: float; range_rate_km_s: float
    orbital_alt_km: float
    tle_name: str; tle1: str; tle2: str

class OverheadResponse(BaseModel):
    count: int; epoch_utc: str
    min_el_deg: float; min_orbital_alt_km: float; max_orbital_alt_km: float
    sats: List[SatOut]

class UpcomingSat(BaseModel):
    name: str; norad_id: int
    rise_time_utc: str; az_at_rise_deg: float; el_at_rise_deg: float
    peak_time_utc: str; peak_el_deg: float; az_at_peak_deg: float
    orbital_alt_km_at_rise: float
    tle_name: str; tle1: str; tle2: str

class UpcomingResponse(BaseModel):
    count: int; now_utc: str; window_min: int; step_s: int
    min_el_deg: float; min_orbital_alt_km: float; max_orbital_alt_km: float
    sats: List[UpcomingSat]

# ---------- Helpers ----------
def fetch_tles(url: str) -> List[Tuple[str,str,str]]:
    if VERBOSE: print(f"[TLE] GET {url}")
    r = requests.get(url, timeout=30); r.raise_for_status()
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    out = []; i = 0
    while i+2 < len(lines):
        name,l1,l2 = lines[i],lines[i+1],lines[i+2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            if SAT_NAME_FILTER and SAT_NAME_FILTER.lower() not in name.lower():
                i += 3; continue
            out.append((name,l1,l2)); i += 3
        else: i += 1
    if VERBOSE: print(f"[TLE] triplets={len(out)}")
    if not out: raise RuntimeError("No TLE triplets parsed.")
    return out

def get_tles() -> List[Tuple[str,str,str]]:
    now = time.time()
    if now - _tle_cache["fetched_at"] > CACHE_TTL_SECONDS or not _tle_cache["satellites"]:
        if VERBOSE: print("[CACHE] Refreshing TLEs…")
        _tle_cache["satellites"] = fetch_tles(DEFAULT_TLE_URL)
        _tle_cache["fetched_at"] = now
    else:
        if VERBOSE:
            age = int(now - _tle_cache["fetched_at"])
            print(f"[CACHE] Using cached (age={age}s, size={len(_tle_cache['satellites'])})")
    return _tle_cache["satellites"]


NORAD_RE = re.compile(r'''
    ^1\s*                # line number 1, optional spaces
    (\d{5})              # 5‑digit catalog number (the NORAD ID)
''', re.VERBOSE)

def parse_norad(line1: str) -> Optional[int]:
    """
    Extract the NORAD catalog number from a TLE line.
    Returns ``None`` if the line cannot be parsed.
    """
    line = line1.strip()
    # 1️⃣ Try the official column‑based slice (columns 3‑7)
    if len(line) >= 7:
        try:
            return int(line[2:7])
        except ValueError:
            pass    # fall back to regex

    # 2️⃣ Regex fallback – matches even if extra spaces are present
    m = NORAD_RE.match(line)
    if m:
        return int(m.group(1))

    # 3️⃣ Could not parse → signal the problem
    return None

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    lat = math.radians(lat_deg); lon = math.radians(lon_deg); h_km = h_m/1000.0
    sl, cl = math.sin(lat), math.cos(lat); N = WGS84_A_KM/math.sqrt(1.0 - WGS84_E2*sl*sl)
    x = (N + h_km)*cl*math.cos(lon); y = (N + h_km)*cl*math.sin(lon); z = (N*(1.0-WGS84_E2)+h_km)*sl
    return np.array([x,y,z])

def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat); so, co = math.sin(lon), math.cos(lon)
    return np.array([[-so,      co,     0],
                     [-sl*co, -sl*so,  cl],
                     [ cl*co,  cl*so,  sl]])

def gmst_from_julian(jd_ut1: float) -> float:
    T = (jd_ut1 - 2451545.0)/36525.0
    gmst_deg = (280.46061837 + 360.98564736629*(jd_ut1-2451545.0)
                + 0.000387933*T*T - (T**3)/38710000.0)
    return math.radians(gmst_deg % 360.0)

def teme_to_ecef(r_teme_km: np.ndarray, v_teme_km_s: np.ndarray, gmst_rad: float):
    cg, sg = math.cos(gmst_rad), math.sin(gmst_rad)
    R = np.array([[cg, sg, 0],[ -sg, cg, 0],[0,0,1]])
    r_ecef = R @ r_teme_km
    v_ecef = R @ v_teme_km_s - np.cross(np.array([0,0,OMEGA_EARTH]), r_ecef)
    return r_ecef, v_ecef

def los_range_rate(r_rel_ecef: np.ndarray, v_rel_ecef: np.ndarray) -> float:
    rng = np.linalg.norm(r_rel_ecef); 
    if rng == 0.0: return 0.0
    return float(np.dot(v_rel_ecef, r_rel_ecef/rng))

def advance_jd(jd:int, fr:float, dt_seconds:float):
    dt_days = dt_seconds/86400.0; fr2 = fr + dt_days; jd2 = jd
    if fr2 >= 1.0: add = int(fr2); jd2 += add; fr2 -= add
    elif fr2 < 0.0: sub = int((-fr2)+1); jd2 -= sub; fr2 += sub
    return jd2, fr2

# ---------- Worker functions (must be top-level, picklable) ----------
def worker_overhead_chunk(chunk: List[Tuple[str,str,str]], jd:int, fr:float, gmst_rad:float,
                          lat:float, lon:float, alt_m:float, min_el:float,
                          min_alt_km:float, max_alt_km:float) -> List[Dict[str,Any]]:
    obs_ecef = geodetic_to_ecef(lat, lon, alt_m)
    e2enu = ecef_to_enu_matrix(lat, lon)
    out: List[Dict[str,Any]] = []
    for (name,l1,l2) in chunk:
        try:
            s = Satrec.twoline2rv(l1,l2)
            err, r_teme_km, v_teme_km_s = s.sgp4(jd, fr)
            if err != 0: continue
            r_teme = np.array(r_teme_km, float); v_teme = np.array(v_teme_km_s, float)
            r_ecef, v_ecef = teme_to_ecef(r_teme, v_teme, gmst_rad)

            orbital_alt_km = float(np.linalg.norm(r_ecef)) - EARTH_MEAN_RADIUS_KM
            if not (min_alt_km <= orbital_alt_km <= max_alt_km): continue

            r_rel = r_ecef - obs_ecef
            enu = e2enu @ r_rel
            e,n,u = enu.tolist()
            rng_km = float(np.linalg.norm(enu))
            if rng_km == 0.0:
                continue
            az_deg = (math.degrees(math.atan2(e,n)) + 360.0) % 360.0
            el_deg = math.degrees(math.asin(u / rng_km))
            if el_deg < min_el: continue
            rr_km_s = los_range_rate(r_rel, v_ecef)
            out.append({
                "name": name, "norad_id": parse_norad(l1),
                "az_deg": az_deg, "el_deg": el_deg,
                "range_km": rng_km, "range_rate_km_s": rr_km_s,
                "orbital_alt_km": orbital_alt_km,
                "tle_name": name, "tle1": l1, "tle2": l2
            })
        except: 
            continue
    return out
# ----------------------------------------------------------------------
# New worker: produce the same payload as `/upcoming` **but only for the
# satellites that are already overhead right now**.
# ----------------------------------------------------------------------
def worker_overhead_as_upcoming_chunk(
    chunk: List[Tuple[str, str, str]],
    jd: int,
    fr: float,
    gmst_rad: float,
    lat: float,
    lon: float,
    alt_m: float,
    min_el: float,
    min_alt_km: float,
    max_alt_km: float,
    now_dt: datetime,
) -> List[Dict[str, Any]]:
    """
    Same geometry calculation as ``worker_overhead_chunk`` but returns a
    dict that conforms to the ``UpcomingSat`` model.  The current UTC time
    (``now_dt``) is used for both the rise and peak timestamps – we are
    only interested in what is *right now*.
    """
    obs_ecef = geodetic_to_ecef(lat, lon, alt_m)
    e2enu = ecef_to_enu_matrix(lat, lon)
    out: List[Dict[str, Any]] = []
    for (name, l1, l2) in chunk:
        try:
            s = Satrec.twoline2rv(l1, l2)
            err, r_teme_km, v_teme_km_s = s.sgp4(jd, fr)
            if err != 0:
                continue
            r_teme = np.array(r_teme_km, float)
            v_teme = np.array(v_teme_km_s, float)
            r_ecef, v_ecef = teme_to_ecef(r_teme, v_teme, gmst_rad)

            orbital_alt_km = float(np.linalg.norm(r_ecef)) - EARTH_MEAN_RADIUS_KM
            if not (min_alt_km <= orbital_alt_km <= max_alt_km):
                continue

            r_rel = r_ecef - obs_ecef
            enu = e2enu @ r_rel
            e, n, u = enu.tolist()
            rng_km = float(np.linalg.norm(enu))
            if rng_km == 0.0:
                continue

            az_deg = (math.degrees(math.atan2(e, n)) + 360.0) % 360.0
            el_deg = math.degrees(math.asin(u / rng_km))
            if el_deg < min_el:
                # Not high enough to be considered “overhead”.
                continue

            rr_km_s = los_range_rate(r_rel, v_ecef)

            # Build a payload that matches UpcomingSat.
            out.append({
                "name": name,
                "norad_id": parse_norad(l1),
                "rise_time_utc": now_dt.isoformat(),
                "az_at_rise_deg": az_deg,
                "el_at_rise_deg": el_deg,
                # For “now‑overhead” the peak is the current point.
                "peak_time_utc": now_dt.isoformat(),
                "peak_el_deg": el_deg,
                "az_at_peak_deg": az_deg,
                "orbital_alt_km_at_rise": orbital_alt_km,
                "tle_name": name,
                "tle1": l1,
                "tle2": l2,
            })
        except Exception:
            # Any failure on a single satellite should not abort the whole
            # chunk – just skip it.
            continue
    return out

def elevation_triplet(s: Satrec, jd:int, fr:float, lat:float, lon:float, alt_m:float,
                      min_alt_km:float, max_alt_km:float):
    err, r_teme_km, v_teme_km_s = s.sgp4(jd, fr)
    if err != 0: return None
    r_teme = np.array(r_teme_km, float); v_teme = np.array(v_teme_km_s, float)
    gmst = gmst_from_julian(jd + fr)
    r_ecef, v_ecef = teme_to_ecef(r_teme, v_teme, gmst)
    orbital_alt_km = float(np.linalg.norm(r_ecef)) - EARTH_MEAN_RADIUS_KM
    if not (min_alt_km <= orbital_alt_km <= max_alt_km): return None
    obs_ecef = geodetic_to_ecef(lat, lon, alt_m); e2enu = ecef_to_enu_matrix(lat, lon)
    r_rel = r_ecef - obs_ecef
    enu = e2enu @ r_rel; e,n,u = enu.tolist()
    rng_km = float(np.linalg.norm(enu)) 
    if rng_km == 0.0:
        return None
    el = math.degrees(math.asin(u / rng_km))
    az = (math.degrees(math.atan2(e, n)) + 360.0) % 360.0
    return el, az, orbital_alt_km

def worker_upcoming_chunk(chunk: List[Tuple[str,str,str]], jd0:int, fr0:float,
                          lat:float, lon:float, alt_m:float, min_el:float, min_peak_el:float,
                          min_alt_km:float, max_alt_km:float, window_s:int, step_s:int) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    for (name,l1,l2) in chunk:
        try:
            s = Satrec.twoline2rv(l1,l2)
            now_info = elevation_triplet(s, jd0, fr0, lat, lon, alt_m, min_alt_km, max_alt_km)
            if not now_info: continue
            el_now, _, _ = now_info
            if el_now >= min_el:  # already up; skip for "upcoming"
                continue

            # find rise
            prev_el = el_now; jd, fr = jd0, fr0
            rise_time = None; rise_el=rise_az=rise_alt=0.0
            t = 0
            while t <= window_s:
                jd, fr = advance_jd(jd, fr, step_s); t += step_s
                info = elevation_triplet(s, jd, fr, lat, lon, alt_m, min_alt_km, max_alt_km)
                if info is None: continue
                el, az, alt_km = info
                if prev_el < min_el <= el:
                    jd_prev, fr_prev = advance_jd(jd, fr, -step_s)
                    info_prev = elevation_triplet(s, jd_prev, fr_prev, lat, lon, alt_m, min_alt_km, max_alt_km)
                    if info_prev is not None:
                        el_prev, _, _ = info_prev
                        denom = (el - el_prev)
                        frac = 0.0 if denom == 0 else (min_el - el_prev)/denom
                        frac = min(max(frac,0.0),1.0)
                        jd_hit, fr_hit = advance_jd(jd_prev, fr_prev, frac*step_s)
                        info_hit = elevation_triplet(s, jd_hit, fr_hit, lat, lon, alt_m, min_alt_km, max_alt_km) or (el, az, alt_km)
                        el_h, az_h, alt_h = info_hit
                        rise_time = jd_to_datetime_utc(float(jd_hit) + float(fr_hit))
                        rise_el, rise_az, rise_alt = el_h, az_h, alt_h
                    else:
                        rise_time = jd_to_datetime_utc(float(jd_hit) + float(fr_hit))
                        rise_el, rise_az, rise_alt = el, az, alt_km
                    break
                prev_el = el
            if rise_time is None: continue

            # ---------- Find peak (maximum elevation) ----------
            # Initialise with no peak yet – we’ll set it the first time we see a higher elevation.
            peak_el = -1e9
            peak_time: Optional[datetime] = None
            peak_az = 0.0

            jd_peak, fr_peak = jd, fr
            t_after = 0
            while t_after <= window_s:
                info = elevation_triplet(s, jd_peak, fr_peak, lat, lon, alt_m,
                                         min_alt_km, max_alt_km)
                if info is None:
                    # No valid geometry at this step – just advance.
                    jd_peak, fr_peak = advance_jd(jd_peak, fr_peak, step_s)
                    t_after += step_s
                    continue

                el, az, _ = info
                # Once elevation drops below the threshold we’re done with the pass.
                if el < min_el:
                    break

                if el > peak_el:
                    peak_el = el
                    peak_time = jd_to_datetime_utc(float(jd_peak) + float(fr_peak))
                    peak_az = az

                jd_peak, fr_peak = advance_jd(jd_peak, fr_peak, step_s)
                t_after += step_s

            # Discard passes that never reach the required peak elevation.
            if peak_el < min_peak_el or peak_time is None:
                continue

            out.append({
                "name": name,
                "norad_id": parse_norad(l1),
                "rise_time_utc": rise_time.isoformat(),
                "az_at_rise_deg": rise_az,
                "el_at_rise_deg": rise_el,
                # Use the *actual* peak time we just computed.
                "peak_time_utc": peak_time.isoformat(),
                "peak_el_deg": peak_el,
                "az_at_peak_deg": peak_az,
                "orbital_alt_km_at_rise": rise_alt,
                "tle_name": name,
                "tle1": l1,
                "tle2": l2
            })
        except:
            continue
    return out

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "source": DEFAULT_TLE_URL,
            "cache_age_s": int(time.time()-_tle_cache["fetched_at"]),
            "cached": len(_tle_cache["satellites"]),
            "pool_workers": POOL_WORKERS, "chunk_size": CHUNK_SIZE}

@app.get("/overhead", response_model=OverheadResponse)
def overhead(
    lat: float = Query(...), lon: float = Query(...), alt_m: float = Query(0.0),
    min_el: float = Query(0.0), min_orbital_alt_km: float = Query(0.0),
    max_orbital_alt_km: float = Query(2000.0), limit: int = Query(5000)
):
    sats = get_tles()
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond*1e-6)
    gmst = gmst_from_julian(jd+fr)

    # chunk work
    chunks = [sats[i:i+CHUNK_SIZE] for i in range(0, len(sats), CHUNK_SIZE)]
    futures = [ _pool.submit(worker_overhead_chunk, ch, jd, fr, gmst, lat, lon, alt_m,
                             min_el, min_orbital_alt_km, max_orbital_alt_km)
                for ch in chunks ]
    results: List[SatOut] = []
    for f in as_completed(futures):
        for d in f.result():
            results.append(SatOut(**d))
            if len(results) >= limit: break
        if len(results) >= limit: break

    results.sort(key=lambda s: s.el_deg, reverse=True)
    return OverheadResponse(
        count=len(results), epoch_utc=now.isoformat(),
        min_el_deg=min_el, min_orbital_alt_km=min_orbital_alt_km, max_orbital_alt_km=max_orbital_alt_km,
        sats=results)

@app.get("/upcoming", response_model=UpcomingResponse)
def upcoming(
    lat: float = Query(...), lon: float = Query(...), alt_m: float = Query(0.0),
    min_el: float = Query(0.0), min_peak_el: float = Query(45.0),
    min_orbital_alt_km: float = Query(0.0), max_orbital_alt_km: float = Query(2000.0),
    window_min: int = Query(60), step_s: int = Query(30), limit: int = Query(200)
):
    sats = get_tles()
    now = datetime.now(timezone.utc)
    jd0, fr0 = jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond*1e-6)
    window_s = max(1, window_min*60); step_s = max(1, step_s)

    chunks = [sats[i:i+CHUNK_SIZE] for i in range(0, len(sats), CHUNK_SIZE)]
    futures = [ _pool.submit(worker_upcoming_chunk, ch, jd0, fr0, lat, lon, alt_m, min_el, min_peak_el,
                             min_orbital_alt_km, max_orbital_alt_km, window_s, step_s)
                for ch in chunks ]
    flat: List[UpcomingSat] = []
    for f in as_completed(futures):
        for d in f.result():
            flat.append(UpcomingSat(**d))
            if len(flat) >= limit: break
        if len(flat) >= limit: break

    flat.sort(key=lambda s: s.rise_time_utc)
    return UpcomingResponse(
        count=len(flat), now_utc=now.isoformat(),
        window_min=window_min, step_s=step_s,
        min_el_deg=min_el, min_orbital_alt_km=min_orbital_alt_km, max_orbital_alt_km=max_orbital_alt_km,
        sats=flat)
# ----------------------------------------------------------------------
# New endpoint: `/overhead_now`
# Returns the same JSON structure as `/upcoming` but only for satellites
# that are currently overhead (elevation ≥ min_el).  No future‑pass search.
# ----------------------------------------------------------------------
@app.get("/overhead_now", response_model=UpcomingResponse)
def overhead_now(
    lat: float = Query(...),
    lon: float = Query(...),
    alt_m: float = Query(0.0),
    min_el: float = Query(0.0),
    min_orbital_alt_km: float = Query(0.0),
    max_orbital_alt_km: float = Query(2000.0),
    limit: int = Query(5000),
):
    """
    Returns satellites that are **already** above the observer at the
    moment of the request.  The response format is identical to the
    ``/upcoming`` endpoint (i.e. a list of ``UpcomingSat`` objects) – the
    current UTC time is used for both the rise and peak timestamps.
    """
    sats = get_tles()
    now_dt = datetime.now(timezone.utc)
    jd, fr = jday(
        now_dt.year,
        now_dt.month,
        now_dt.day,
        now_dt.hour,
        now_dt.minute,
        now_dt.second + now_dt.microsecond * 1e-6,
    )
    gmst = gmst_from_julian(jd + fr)

    # Parallel work – identical chunking strategy to the other endpoints.
    chunks = [sats[i : i + CHUNK_SIZE] for i in range(0, len(sats), CHUNK_SIZE)]
    futures = [
        _pool.submit(
            worker_overhead_as_upcoming_chunk,
            ch,
            jd,
            fr,
            gmst,
            lat,
            lon,
            alt_m,
            min_el,
            min_orbital_alt_km,
            max_orbital_alt_km,
            now_dt,
        )
        for ch in chunks
    ]

    flat: List[UpcomingSat] = []
    for f in as_completed(futures):
        for d in f.result():
            flat.append(UpcomingSat(**d))
            if len(flat) >= limit:
                break
        if len(flat) >= limit:
            break

    # Sort by the (current) rise time – all entries have the same timestamp,
    # but keeping a deterministic order is nice.
    flat.sort(key=lambda s: s.rise_time_utc)

    return UpcomingResponse(
        count=len(flat),
        now_utc=now_dt.isoformat(),
        window_min=0,               # not used for this endpoint
        step_s=0,                   # not used for this endpoint
        min_el_deg=min_el,
        min_orbital_alt_km=min_orbital_alt_km,
        max_orbital_alt_km=max_orbital_alt_km,
        sats=flat,
    )