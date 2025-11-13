#!/usr/bin/env python3
"""
Diagnostic script to check what's wrong with the Polar Alignment App
"""

import os
import sys

print("=" * 70)
print(" POLAR ALIGNMENT APP - DIAGNOSTIC CHECK")
print("=" * 70)
print()

issues_found = []
warnings_found = []

# Check 1: Files exist
print("[1] Checking files...")
required_files = ["polar_align.py", "config.ini", "templates/index.html"]
for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ {file} exists")
    else:
        print(f"  ✗ {file} MISSING!")
        issues_found.append(f"Missing file: {file}")
print()

# Check 2: Check Python version
print("[2] Checking Python version...")
print(f"  Python: {sys.version}")
if sys.version_info < (3, 7):
    issues_found.append("Python version too old (need 3.7+)")
    print("  ✗ Python version is too old!")
else:
    print("  ✓ Python version OK")
print()

# Check 3: Check imports
print("[3] Checking required modules...")
required_modules = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "astropy": "astropy",
    "flask": "flask"
}

for module, package in required_modules.items():
    try:
        __import__(module)
        print(f"  ✓ {module} installed")
    except ImportError:
        print(f"  ✗ {module} NOT installed (install with: pip install {package})")
        issues_found.append(f"Missing module: {module}")
print()

# Check 4: Check config.ini
print("[4] Checking config.ini...")
if os.path.exists("config.ini"):
    import configparser
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    
    if config.has_section("GPS"):
        lat = config.get("GPS", "latitude", fallback="NOT SET")
        lon = config.get("GPS", "longitude", fallback="NOT SET")
        print(f"  GPS Latitude: {lat}")
        print(f"  GPS Longitude: {lon}")
        if lat == "NOT SET" or lon == "NOT SET":
            warnings_found.append("GPS coordinates not set properly")
    
    if config.has_section("CAMERA"):
        camera_index = config.get("CAMERA", "index", fallback="NOT SET")
        print(f"  Camera Index: {camera_index}")
    
    print("  ✓ Config file OK")
else:
    print("  ✗ config.ini not found!")
    issues_found.append("config.ini missing")
print()

# Check 5: Check log file
print("[5] Checking log file...")
if os.path.exists("polar_align.log"):
    with open("polar_align.log", "r", encoding="utf-8") as f:
        lines = f.readlines()
        if lines:
            print(f"  Log file has {len(lines)} lines")
            # Check last 10 lines for errors
            last_lines = lines[-10:]
            errors = [line for line in last_lines if "ERROR" in line or "error" in line]
            if errors:
                print("  ⚠ Recent errors found in log:")
                for err in errors[:3]:  # Show max 3
                    print(f"    {err.strip()}")
                warnings_found.append("Errors in log file")
            
            # Check for Polaris calculations
            polaris_calcs = [line for line in lines if "Polaris at" in line]
            if polaris_calcs:
                print(f"  ✓ Found {len(polaris_calcs)} Polaris calculations in log")
                print(f"    Last: {polaris_calcs[-1].strip()}")
            else:
                print("  ⚠ No Polaris calculations found in log")
                warnings_found.append("No Polaris calculations in log - app may not be running")
        else:
            print("  ⚠ Log file is empty")
    print("  ✓ Log file exists")
else:
    print("  ⚠ No log file yet (app hasn't run)")
    warnings_found.append("App hasn't been run yet")
print()

# Check 6: Check if app is running
print("[6] Checking if app is running...")
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', 5000))
    sock.close()
    if result == 0:
        print("  ✓ App is running on port 5000")
        print("    Open: http://localhost:5000")
    else:
        print("  ✗ App is NOT running on port 5000")
        warnings_found.append("App not running - start it with: python polar_align.py")
except:
    print("  ✗ Could not check if app is running")
print()

# Check 7: Test Polaris calculation
print("[7] Testing Polaris calculation...")
try:
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    import astropy.units as u
    import datetime
    import math
    
    location = EarthLocation(lat=50.51667 * u.deg, lon=2.86667 * u.deg, height=25 * u.m)
    utc_time = Time(datetime.datetime.now(datetime.timezone.utc), scale="utc")
    polaris = SkyCoord(ra=2.53030102 * u.hourangle, dec=89.26413805 * u.deg, frame="icrs")
    altaz_frame = AltAz(obstime=utc_time, location=location)
    polaris_altaz = polaris.transform_to(altaz_frame)
    
    print(f"  Polaris Altitude: {polaris_altaz.alt.degree:.2f}°")
    print(f"  Polaris Azimuth: {polaris_altaz.az.degree:.2f}°")
    print("  ✓ Polaris calculation WORKS!")
except Exception as e:
    print(f"  ✗ Polaris calculation FAILED: {e}")
    issues_found.append(f"Polaris calculation error: {e}")
print()

# Summary
print("=" * 70)
print(" SUMMARY")
print("=" * 70)

if not issues_found and not warnings_found:
    print("✓ Everything looks good!")
    print()
    print("If you still don't see:")
    print("  - Polaris red dot on video feed")
    print("  - Hour angle in interface")
    print("  - Zoom controls")
    print()
    print("Then you need to RESTART the app:")
    print("  1. Stop the app (Ctrl+C in terminal)")
    print("  2. Run: python polar_align.py")
    print("  3. Refresh browser (Ctrl+F5)")
elif issues_found:
    print(f"✗ Found {len(issues_found)} critical issue(s):")
    for issue in issues_found:
        print(f"  - {issue}")
    print()
    print("Fix these issues before running the app!")
else:
    print(f"⚠ Found {len(warnings_found)} warning(s):")
    for warning in warnings_found:
        print(f"  - {warning}")
    print()
    print("The app might work, but check these warnings.")

print()
print("=" * 70)

# Provide next steps
if "App not running" in str(warnings_found):
    print()
    print("NEXT STEPS:")
    print("  1. Start the app: python polar_align.py")
    print("  2. Open browser: http://localhost:5000")
    print("  3. Check the interface")

