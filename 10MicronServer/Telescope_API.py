import socket
import time
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
from typing import Optional

# --- Configuration ---
TELESCOPE_HOST = "192.168.1.99"
TELESCOPE_PORT = 3492
CELESTRAK_URL_BASE = "https://celestrak.org/NORAD/elements/gp.php"
SEARCH_DAYS = 4

# --- Helper Functions (Time Conversion & TLE Fetching) ---

def datetime_to_jd(dt_obj):
    """Converts a Python datetime object to a Julian Date."""
    return dt_obj.timestamp() / 86400.0 + 2440587.5

def jd_to_datetime(jd):
    """Converts a Julian Date to a Python datetime object in UTC."""
    unix_timestamp = (jd - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

def fetch_tle(satellite_name):
    """Fetches and sanitizes the latest TLE data for a given satellite from Celestrak."""
    params = {'NAME': satellite_name, 'FORMAT': 'tle'}
    try:
        response = requests.get(CELESTRAK_URL_BASE, params=params, timeout=10)
        response.raise_for_status()
        raw_tle = response.text
        if "No TLE found" in raw_tle or not raw_tle.strip():
            return None
        sanitized_chars = [c if 32 <= ord(c) <= 126 or c in ('\n', '\r') else ' ' for c in raw_tle]
        return "".join(sanitized_chars).strip().replace('\r\n', '\n')
    except requests.exceptions.RequestException:
        return None

# --- Telescope Controller Class ---
# This class contains all the low-level communication logic with the mount.

class TelescopeController:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.sock, self.is_connected = None, False
        self.buffer = b""
        self.default_timeout = 15

    def connect(self):
        try:
            print(f"Connecting to telescope at {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.default_timeout)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            print("Connection successful. Clearing buffer and getting info...")
            self.clear_buffer()
            self.get_mount_info() # Initial info query on connect
        except socket.error as e:
            self.is_connected = False
            print(f"Error: Could not connect. {e}")
            raise

    def disconnect(self):
        if self.sock:
            print("Disconnecting from telescope.")
            self.sock.close()
            self.sock = None

    def send_command(self, command, timeout=2.0):
        """
        Sends a command and waits for a response.
        Handles responses with and without a '#' terminator by using a timeout.
        """
        if not self.is_connected:
            raise HTTPException(status_code=503, detail="Telescope not connected")
        
        original_timeout = self.sock.gettimeout()
        
        try:
            # Clear any stale data from the buffer before sending
            self.sock.settimeout(0.1)
            try:
                while self.sock.recv(1024): pass
            except socket.timeout:
                pass # Buffer is now clear, which is expected

            # Set the actual desired timeout for the command response
            self.sock.settimeout(timeout)
            
            # Send the command
            self.sock.sendall(command.encode('ascii'))
            
            # Receive the response
            buffer = b""
            while True:
                try:
                    chunk = self.sock.recv(1024)
                    if not chunk:
                        break # Connection closed
                    buffer += chunk
                    if b'#' in chunk:
                        break # Stop early if we find the terminator
                except socket.timeout:
                    # Timeout is an acceptable way to end for commands without '#'
                    break
            
            return buffer.decode('ascii', errors='ignore').strip()

        except socket.error as e:
            raise HTTPException(status_code=504, detail=f"Telescope communication error: {e}")
        finally:
            # Always restore the original default timeout
            self.sock.settimeout(original_timeout)
    
    def send_fire_and_forget(self, command):
        if not self.is_connected:
            raise HTTPException(status_code=503, detail="Telescope not connected")
        try:
            self.sock.sendall(command.encode('ascii'))
        except socket.error as e:
            raise HTTPException(status_code=504, detail=f"Telescope communication error: {e}")

    def clear_buffer(self):
        self.send_fire_and_forget("#")
        time.sleep(0.5)

    # --- API Methods ---
    def get_mount_info(self):
        fw = self.send_command(":GVN#").strip('#')
        uid = self.send_command(":GETID#").strip('#')
        return {"firmware_version": fw, "unique_id": uid}
        
    def get_status(self):
        status_map = {
            "0": "Tracking", "1": "Stopped (user)", "2": "Slewing to Park",
            "3": "Unparking", "4": "Slewing to Home", "5": "Parked",
            "6": "Slewing", "7": "Tracking Off", "8": "Low Temp Inhibit",
            "9": "Outside Limits", "10": "Tracking Satellite", "11": "User Intervention Required"
        }
        code = self.send_command(":Gstat#").strip('#')
        return {"status_code": code, "status_description": status_map.get(code, "Unknown")}

    def get_radec(self):
        ra = self.send_command(":GR#").strip('#')
        dec = self.send_command(":GD#").strip('#')
        return {"ra": ra, "dec": dec}
    def slew_to_radec(self, ra_str: str, dec_str: str, rapid = True):
        """Sets target RA/Dec and initiates a slew using fire-and-forget for the slew command."""
        self.send_fire_and_forget("#") #clear command buffer
        status = self.get_status()
        if status.get("status_description") == "Parked":
            self.unpark()
                
        # Set target Right Ascension using :Sr command. This is a quick command.
        # It's important to wait for the response to know the coordinates are valid.
        if not rapid: 
            # Set tracking to sidereal rate before slewing. This is a quick command.
            self.send_command(":TQ#")
            ra_response = self.send_command(f":Sr{ra_str}#")
            if ra_response.strip('#') != '1':
                raise HTTPException(status_code=400, detail=f"Mount rejected RA value. Response: {ra_response}")

            # Set target Declination using :Sd command. This is a quick command.
            dec_response = self.send_command(f":Sd{dec_str}#")
            if dec_response.strip('#') != '1':
                raise HTTPException(status_code=400, detail=f"Mount rejected Dec value. Response: {dec_response}")
            
            # Initiate Slew using :MS# command with fire-and-forget to avoid timeouts.
            # We lose the ability to check the return code, but this prevents the API from hanging.
            self.send_fire_and_forget(":MS#")
            
            return {"message": f"Slew command sent to RA: {ra_str}, Dec: {dec_str}. Tracking should be active at sidereal rate."}
        else: 
            # Set tracking to sidereal rate before slewing. This is a quick command.
            self.send_fire_and_forget(":TQ#")
            self.send_fire_and_forget(f":Sr{ra_str}#")
            self.send_fire_and_forget(f":Sd{dec_str}#")
            # Initiate Slew using :MS# command with fire-and-forget to avoid timeouts.
            # We lose the ability to check the return code, but this prevents the API from hanging.
            self.send_fire_and_forget(":MS#")
            
            return {"message": f"Rapid Slew command sent to RA: {ra_str}, Dec: {dec_str}. Tracking should be active at sidereal rate."}

    def get_altaz(self):
        alt = self.send_command(":GA#").strip('#')
        az = self.send_command(":GZ#").strip('#')
        return {"alt": alt, "az": az}

    def park(self):
        self.send_fire_and_forget(":hP#")
        return {"message": "Park command sent."}

    def unpark(self):
        self.send_fire_and_forget(":PO#")
        time.sleep(5) # Unparking can take time
        return {"message": "Unpark command sent."}

    def _escape_tle(self, tle_string):
        escaped = []
        for char in tle_string:
            if char == '$': escaped.append('$$')
            elif char == '#': escaped.append('$23')
            elif char == ',': escaped.append('$2C')
            elif ord(char) < 32 or ord(char) > 126: escaped.append(f'${ord(char):02X}')
            else: escaped.append(char)
        return "".join(escaped)

    def load_tle(self, tle_string):
        command = f":TLEL0{self._escape_tle(tle_string)}#"
        response = self.send_command(command, timeout=30)
        if response != "V#":
            raise HTTPException(status_code=400, detail=f"Mount rejected TLE. Response: {response}")
        return {"message": "TLE loaded successfully."}

    def get_next_pass(self, start_jd):
        response = self.send_command(f":TLEP{start_jd:.8f},1440#")
        return response

# --- FastAPI Application ---

app = FastAPI(title="Telescope Control API", description="An API to control a 10micron telescope mount.")
controller = TelescopeController(TELESCOPE_HOST, TELESCOPE_PORT)

@app.on_event("startup")
async def startup_event():
    controller.connect()

@app.on_event("shutdown")
async def shutdown_event():
    controller.disconnect()

# --- Pydantic Models for API ---
class StatusResponse(BaseModel):
    status_code: str
    status_description: str

class VersionResponse(BaseModel):
    firmware_version: str
    unique_id: str
    
class PositionResponse(BaseModel):
    ra: Optional[str] = None
    dec: Optional[str] = None
    alt: Optional[str] = None
    az: Optional[str] = None

class MessageResponse(BaseModel):
    message: str
    
class PassResponse(BaseModel):
    message: str
    pass_found: bool
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None

# NEW: Pydantic model for the RA/Dec slew request body
class SlewRaDecBody(BaseModel):
    ra: str = Field(..., description="Target Right Ascension", example="18:36:56.3")
    dec: str = Field(..., description="Target Declination", example="+38*47:01")

# --- API Endpoints ---
@app.get("/status", response_model=StatusResponse, summary="Get current mount status")
def get_status():
    return controller.get_status()

@app.get("/version", response_model=VersionResponse, summary="Get mount firmware and ID")
def get_version():
    return controller.get_mount_info()

@app.get("/position/radec", response_model=PositionResponse, summary="Get current RA/Dec coordinates")
def get_position_radec():
    return controller.get_radec()

@app.get("/position/altaz", response_model=PositionResponse, summary="Get current Alt/Az coordinates")
def get_position_altaz():
    return controller.get_altaz()

@app.post("/park", response_model=MessageResponse, summary="Park the telescope")
def park_scope():
    return controller.park()

@app.post("/unpark", response_model=MessageResponse, summary="Unpark the telescope")
def unpark_scope():
    return controller.unpark()
# NEW: API endpoint to slew to specific RA/Dec coordinates
@app.post("/slew/radec", response_model=MessageResponse, summary="Slew to specific RA/Dec coordinates")
def slew_to_radec(coords: SlewRaDecBody):
    """
    Slews the telescope to the specified Right Ascension and Declination.
    
    - **ra**: Target RA in `HH:MM:SS.S` format (or other formats accepted by the mount).
    - **dec**: Target Dec in `sDD*MM:SS` format (e.g., `+25*30:15` or `-05*10:00`).
    """
    return controller.slew_to_radec(coords.ra, coords.dec)

@app.post("/tle/{satellite_name}", response_model=MessageResponse, summary="Fetch and load TLE for a satellite")
def load_satellite_tle(satellite_name: str):
    tle_data = fetch_tle(satellite_name)
    if not tle_data:
        raise HTTPException(status_code=404, detail="Satellite not found on Celestrak")
    return controller.load_tle(tle_data)

@app.get("/passes/{satellite_name}", response_model=PassResponse, summary="Find the next pass for a satellite")
def get_passes(satellite_name: str):
    # Ensure TLE is loaded first
    load_satellite_tle(satellite_name)
    start_jd = datetime_to_jd(datetime.now(timezone.utc))
    for _ in range(SEARCH_DAYS):
        pass_info = controller.get_next_pass(start_jd)
        if pass_info and pass_info not in ["N#", "E#"]:
            parts = pass_info[:-1].split(',')
            start_pass_time = jd_to_datetime(float(parts[0]))
            end_pass_time = jd_to_datetime(float(parts[1]))
            return {
                "message": f"Next pass for '{satellite_name}' found.",
                "pass_found": True,
                "start_time_utc": start_pass_time.isoformat(),
                "end_time_utc": end_pass_time.isoformat()
            }
        start_jd += 1.0 # Advance to the next day
    return {"message": "No pass found in the next 4 days.", "pass_found": False}

@app.post("/slew/satellite/{satellite_name}", response_model=MessageResponse, summary="Slew to an imminent satellite pass")
def slew_to_satellite(satellite_name: str):
    status = controller.get_status()
    if status.get("status_description") == "Parked":
        controller.unpark()
    
    pass_data = get_passes(satellite_name)
    if not pass_data["pass_found"]:
        raise HTTPException(status_code=404, detail="No upcoming pass found to slew to.")
    
    controller.send_command(":TLES#")
    return {"message": f"Slew command sent for {satellite_name}."}
# ----- Health endpoint -------------------------------------------------
@app.get("/healthz", summary="Simple health check")
def health_check():
    """
    Returns a minimal JSON indicating that the FastAPI process is running
    and (optionally) whether the telescope controller is currently connected.
    """
    # Basic “process is alive” response
    result = {"service": "telescope_api", "status": "ok"}

    # Optional: include mount‑connection status if you want to surface that too
    if controller.is_connected:
        result["mount"] = {"connected": True}
    else:
        result["mount"] = {"connected": False, "detail": "Not connected to mount"}

    return result
# --- To run this API ---
# 1. Save the file as telescope_api.py
# 2. Run in your terminal: uvicorn telescope_api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

