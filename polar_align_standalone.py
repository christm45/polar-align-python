"""
Polar Align Assistant - Version Standalone (Interface Native)
Interface native sans navigateur web externe
"""

import cv2
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
import math
import datetime
import threading
from flask import Flask, render_template, jsonify, Response, request
import time
import configparser
import os
import logging
import json
import webview  # Interface native Windows
import sys

# Importer le module principal
from polar_align import (
    PolarAlignAssistant,
    load_config
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("polar_align.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("PolarAlign-Standalone")

# Flask app
config = load_config()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

assistant = PolarAlignAssistant(config)


def generate_frames():
    """Video stream generator for multipart/x-mixed-replace."""
    boundary = b"--frame"
    while True:
        frame = assistant.get_frame()
        if frame:
            yield boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(1.0 / assistant.camera_fps)


def generate_polaris_display_frames():
    """Polaris display stream generator for multipart/x-mixed-replace."""
    boundary = b"--frame"
    while True:
        frame = assistant.get_polaris_display_frame()
        if frame:
            yield boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(1.0 / assistant.camera_fps)

# Routes Flask (same as before)
@app.route("/")
def index():
    try:
        config = load_config()
        return render_template("index.html", assistant=assistant, config=config._sections)
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        return "Template rendering error.", 500

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/polaris_display")
def polaris_display():
    return Response(generate_polaris_display_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify(assistant.get_status())

@app.route("/recalibrate")
def recalibrate():
    assistant.recalibrate()
    return jsonify({"status": "recalibration_started"})

@app.route("/zoom", methods=['POST'])
def set_zoom():
    try:
        data = request.get_json()
        zoom_level = float(data.get('zoom_level', 1.0))
        assistant.set_zoom(zoom_level)
        return jsonify({"status": "success", "zoom_level": assistant.zoom_level})
    except Exception as e:
        logger.error(f"Error setting zoom: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/zoom", methods=['GET'])
def get_zoom():
    return jsonify({"zoom_level": assistant.zoom_level})

@app.route("/config")
def get_config():
    return jsonify({
        "latitude": assistant.latitude,
        "longitude": assistant.longitude,
        "camera_index": assistant.camera_index,
        "camera_width": assistant.camera_width,
        "camera_height": assistant.camera_height,
        "camera_fps": assistant.camera_fps,
        "min_radius": assistant.min_radius,
        "max_radius": assistant.max_radius,
        "detection_interval": assistant.detection_interval
    })

@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify({
        'camera': {
            'width': assistant.camera_width,
            'height': assistant.camera_height,
            'fps': assistant.camera_fps
        },
        'calibration': {
            'min_radius': assistant.min_radius,
            'max_radius': assistant.max_radius,
            'detection_interval': assistant.detection_interval,
            'three_circle_mode': assistant.three_circle_mode,
            'circle_cluster_threshold': assistant.circle_cluster_threshold
        },
        'location': {
            'latitude': assistant.latitude,
            'longitude': assistant.longitude,
            'altitude': assistant.altitude
        }
    })

@app.route('/settings/camera', methods=['POST'])
def update_camera_settings():
    try:
        data = request.get_json()
        if assistant.update_camera_settings(
            data.get('width', 1280),
            data.get('height', 720),
            data.get('fps', 30)
        ):
            return jsonify({'status': 'success', 'message': 'Camera settings updated'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update camera settings'}), 400
    except Exception as e:
        logger.error(f"Error in update_camera_settings: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/settings/calibration', methods=['POST'])
def update_calibration_settings():
    try:
        data = request.get_json()
        if assistant.update_calibration_settings(
            data.get('min_radius', 25),
            data.get('max_radius', 150),
            data.get('detection_interval', 5),
            data.get('three_circle_mode'),
            data.get('circle_cluster_threshold')
        ):
            return jsonify({'status': 'success', 'message': 'Calibration settings updated'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update calibration settings'}), 400
    except Exception as e:
        logger.error(f"Error in update_calibration_settings: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/settings/location', methods=['POST'])
def update_location_settings():
    try:
        data = request.get_json()
        if assistant.update_location_settings(
            data.get('latitude', assistant.latitude),
            data.get('longitude', assistant.longitude),
            data.get('altitude', assistant.altitude)
        ):
            return jsonify({'status': 'success', 'message': 'Location settings updated'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update location settings'}), 400
    except Exception as e:
        logger.error(f"Error in update_location_settings: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def start_flask():
    """Start Flask server in background thread"""
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)


def main():
    """Main function - Start Flask and open native window"""
    logger.info("Starting Polar Align Assistant - Standalone Mode")
    
    # Start image processing
    assistant.run()
    
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to start
    time.sleep(2)
    
    logger.info("Opening native window interface...")
    
    # Create native window with pywebview
    window = webview.create_window(
        title='Polar Align Assistant - Interface Native',
        url='http://127.0.0.1:5000',
        width=1400,
        height=900,
        resizable=True,
        fullscreen=False,
        min_size=(1024, 768),
        background_color='#000000'
    )
    
    # Start webview (blocking call)
    webview.start()
    
    logger.info("Application closed")


if __name__ == "__main__":
    main()

