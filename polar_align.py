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

# Logging configuration (UTF-8 safe)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("polar_align.log", encoding="utf-8"),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger("PolarAlign")


def load_config():
    # """Load configuration from config.ini or create a default one."""
    config = configparser.ConfigParser()

    if os.path.exists("config.ini"):
        config.read("config.ini", encoding="utf-8")
    else:
        # Default configuration
        config["GPS"] = {
            "latitude": "50.51783",
            "longitude": "2.86613",
            "altitude": "25"
        }
        config["CAMERA"] = {
            "index": "0",
            "width": "1280",
            "height": "720",
            "fps": "30"
        }
        config["CALIBRATION"] = {
            "min_radius": "25",
            "max_radius": "150",
            "detection_interval": "5",
            "three_circle_mode": "true",
            "circle_cluster_threshold": "30"
        }

        with open("config.ini", "w", encoding="utf-8") as configfile:
            config.write(configfile)

    return config


class PolarAlignAssistant:
    def __init__(self, config):
        # GPS configuration
        self.latitude = float(config["GPS"]["latitude"])
        self.longitude = float(config["GPS"]["longitude"])
        self.altitude = float(config["GPS"].get("altitude", "0"))

        # Camera configuration
        self.camera_index = int(config["CAMERA"]["index"])
        self.camera_width = int(config["CAMERA"]["width"])
        self.camera_height = int(config["CAMERA"]["height"])
        self.camera_fps = int(config["CAMERA"]["fps"])

        # Calibration configuration
        self.min_radius = int(config["CALIBRATION"]["min_radius"])
        self.max_radius = int(config["CALIBRATION"]["max_radius"])
        self.detection_interval = int(config["CALIBRATION"]["detection_interval"])
        self.three_circle_mode = config["CALIBRATION"].getboolean("three_circle_mode", True)
        self.circle_cluster_threshold = int(config["CALIBRATION"].get("circle_cluster_threshold", "30"))

        # Astropy location
        self.location = EarthLocation(
            lat=self.latitude * u.deg,
            lon=self.longitude * u.deg,
            height=self.altitude * u.m
        )

        # Camera initialization - try multiple backends for ASI camera compatibility
        self.cap = None
        self.cap = self._initialize_camera(self.camera_index)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        # State variables
        self.reticle_center = None
        self.reticle_radius = None
        self.direction = "UNKNOWN"
        self.offset = (0, 0)
        self.current_frame = None
        self.polaris_display_frame = None
        self.lock = threading.Lock()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.polaris_data = None
        self.reticle_locked = False  # New flag to lock reticle position
        
        # Zoom functionality
        self.zoom_level = 1.0  # 1.0 = no zoom, 2.0 = 2x zoom, etc.
        self.zoom_center = None  # (x, y) or None for center of frame
        
        # Reticle detection improvements
        self.reticle_position_history = []  # For temporal filtering
        self.max_history_size = 10
        self.detection_confidence = 0.0  # 0.0 to 1.0
        self.detection_attempts = 0  # Counter for detection attempts
        self.max_detection_attempts = 10  # Limit detection attempts

        # Load camera calibration if available
        self.load_camera_calibration()

        logger.info(f"Polar alignment assistant initialized with lat={self.latitude}, lon={self.longitude}")

    def _initialize_camera(self, camera_index):
        """Initialize camera with multiple backend attempts for ASI camera compatibility."""
        # Try different backends in order of preference for ASI cameras
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
            (cv2.CAP_ANY, "Any available")
        ]
        
        for backend, name in backends:
            logger.info(f"Attempting to open camera {camera_index} with {name} backend...")
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                # Try to read a test frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"Successfully opened camera {camera_index} with {name} backend")
                    return cap
                else:
                    logger.warning(f"Camera opened but cannot read frames with {name} backend")
                    cap.release()
            else:
                logger.warning(f"Could not open camera with {name} backend")
        
        # If all backends fail, raise error
        logger.error(f"Unable to open camera at index {camera_index} with any backend")
        raise ValueError(f"Unable to open camera at index {camera_index}. Please check camera connection and drivers.")

    def load_camera_calibration(self):
        """Load camera calibration parameters if available."""
        try:
            if os.path.exists("camera_calibration.npz"):
                with np.load("camera_calibration.npz") as data:
                    self.camera_matrix = data.get("camera_matrix", None)
                    self.dist_coeffs = data.get("dist_coeffs", None)
                logger.info("Camera calibration loaded successfully")
            else:
                logger.info("No camera calibration file found")
        except Exception as e:
            logger.error(f"Error loading camera calibration: {e}")

    def undistort_frame(self, frame):
        """Correct lens distortion if calibration is available."""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return frame
    
    def apply_zoom(self, frame):
        """Apply digital zoom to frame."""
        if self.zoom_level <= 1.0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Determine zoom center
        if self.zoom_center:
            center_x, center_y = self.zoom_center
        elif self.reticle_center:
            # Zoom to reticle center if detected
            center_x, center_y = self.reticle_center
        else:
            # Default to frame center
            center_x, center_y = w // 2, h // 2
        
        # Calculate crop region
        crop_w = int(w / self.zoom_level)
        crop_h = int(h / self.zoom_level)
        
        # Ensure crop region is within bounds
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        # Adjust if at edge
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Note: Don't modify reticle_center/radius here as it's used in other parts
        # The overlay drawing will handle the zoom adjustment separately
        
        return zoomed
    
    def set_zoom(self, zoom_level):
        """Set zoom level (1.0 = no zoom, 2.0 = 2x, etc.)"""
        self.zoom_level = max(1.0, min(4.0, zoom_level))  # Limit zoom to 1x-4x
        logger.info(f"Zoom level set to {self.zoom_level}x")

    def update_camera_settings(self, width, height, fps):
        """Update camera settings"""
        try:
            self.camera_width = int(width)
            self.camera_height = int(height)
            self.camera_fps = int(fps)

            # Update camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

            # Update config
            config = load_config()
            config["CAMERA"]["width"] = str(self.camera_width)
            config["CAMERA"]["height"] = str(self.camera_height)
            config["CAMERA"]["fps"] = str(self.camera_fps)

            with open("config.ini", "w", encoding="utf-8") as configfile:
                config.write(configfile)

            logger.info(f"Camera settings updated: {width}x{height} @ {fps}fps")
            return True
        except Exception as e:
            logger.error(f"Error updating camera settings: {e}")
            return False

    def update_calibration_settings(self, min_radius, max_radius, detection_interval, three_circle_mode=None, circle_cluster_threshold=None):
        """Update calibration settings"""
        try:
            self.min_radius = int(min_radius)
            self.max_radius = int(max_radius)
            self.detection_interval = int(detection_interval)
            
            if three_circle_mode is not None:
                self.three_circle_mode = bool(three_circle_mode)
            if circle_cluster_threshold is not None:
                self.circle_cluster_threshold = int(circle_cluster_threshold)

            # Update config
            config = load_config()
            config["CALIBRATION"]["min_radius"] = str(self.min_radius)
            config["CALIBRATION"]["max_radius"] = str(self.max_radius)
            config["CALIBRATION"]["detection_interval"] = str(self.detection_interval)
            config["CALIBRATION"]["three_circle_mode"] = str(self.three_circle_mode).lower()
            config["CALIBRATION"]["circle_cluster_threshold"] = str(self.circle_cluster_threshold)

            with open("config.ini", "w", encoding="utf-8") as configfile:
                config.write(configfile)

            logger.info(f"Calibration settings updated: min={min_radius}, max={max_radius}, interval={detection_interval}, 3-circle={self.three_circle_mode}")
            return True
        except Exception as e:
            logger.error(f"Error updating calibration settings: {e}")
            return False

    def update_location_settings(self, latitude, longitude, altitude):
        """Update location settings"""
        try:
            self.latitude = float(latitude)
            self.longitude = float(longitude)
            self.altitude = float(altitude)

            # Update astropy location
            self.location = EarthLocation(
                lat=self.latitude * u.deg,
                lon=self.longitude * u.deg,
                height=self.altitude * u.m
            )

            # Update config
            config = load_config()
            config["GPS"]["latitude"] = str(self.latitude)
            config["GPS"]["longitude"] = str(self.longitude)
            config["GPS"]["altitude"] = str(self.altitude)

            with open("config.ini", "w", encoding="utf-8") as configfile:
                config.write(configfile)

            logger.info(f"Location updated: lat={latitude}, lon={longitude}, alt={altitude}")
            return True
        except Exception as e:
            logger.error(f"Error updating location settings: {e}")
            return False

    def detect_reticle(self, frame):
        """Detect reticle using 3-circle detection system similar to SkyWatcher reticle v2 with improved precision."""
        # Continue to track even if locked (for position updates)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate image quality metrics for adaptive parameters
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Apply multiple preprocessing techniques for better edge detection
        # Adaptive blur based on image quality
        blur_size = 5 if std_brightness > 30 else 3
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 1.5)
        
        # Adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Also try Otsu thresholding as backup
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Enhanced Canny with adaptive thresholds
        canny_low = max(20, int(mean_brightness * 0.3))
        canny_high = min(200, int(mean_brightness * 1.5))
        canny_edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Try tracking if we already have a reticle
        if self.reticle_center and self.reticle_radius:
            x, y = int(self.reticle_center[0]), int(self.reticle_center[1])
            r = int(self.reticle_radius)

            # Define larger ROI around the previous reticle for better tracking
            margin = int(max(80, r * 1.5))
            y1 = max(0, int(y - margin))
            y2 = min(frame.shape[0], int(y + margin))
            x1 = max(0, int(x - margin))
            x2 = min(frame.shape[1], int(x + margin))
            roi = adaptive_thresh[y1:y2, x1:x2]

            # Detect circles in ROI with optimized parameters for tracking
            circles = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(20, r//3),
                param1=40,  # Lower threshold for tracking
                param2=25,  # Lower threshold for tracking
                minRadius=max(5, int(r * 0.8)),
                maxRadius=int(r * 1.2)
            )

            if circles is not None and len(circles) > 0:
                circles_raw = circles[0, :]  # Keep float precision
                # Adjust coordinates to full image
                circles_raw[:, 0] += x1
                circles_raw[:, 1] += y1
                
                # Pick the closest circle to previous position
                closest_idx = np.argmin([math.sqrt((c[0] - x) ** 2 + (c[1] - y) ** 2) for c in circles_raw])
                closest_circle = circles_raw[closest_idx]
                
                # Apply sub-pixel refinement
                refined_center, refined_radius = self._refine_circle_subpixel(
                    gray, (closest_circle[0], closest_circle[1]), closest_circle[2], blur_size
                )
                
                # Apply temporal filtering
                filtered_center, filtered_radius = self._apply_temporal_filter(refined_center, refined_radius)
                
                # Convert to integers for OpenCV
                self.reticle_center = (int(filtered_center[0]), int(filtered_center[1]))
                self.reticle_radius = int(filtered_radius)
                self.detection_confidence = 0.95  # High confidence for tracking
                return True

        # Full detection with 3-circle system similar to SkyWatcher reticle v2
        # Try multiple detection approaches with adaptive parameters
        detection_methods = [
            # Method 1: Adaptive thresholding
            (adaptive_thresh, 50, 30),
            # Method 2: Otsu thresholding  
            (otsu_thresh, 45, 25),
            # Method 3: Enhanced Canny edge detection
            (canny_edges, 40, 20)
        ]
        
        all_circles = []
        
        for method_img, param1, param2 in detection_methods:
            # Detect circles with different parameters
            circles = cv2.HoughCircles(
                method_img,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(frame.shape[0], frame.shape[1]) // 4,
                param1=param1,
                param2=param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
            
            if circles is not None:
                circles_raw = circles[0, :]  # Keep float precision
                all_circles.extend(circles_raw)

        if all_circles:
            # Find the best circle cluster (3-circle pattern) with confidence scoring
            best_center, best_radius, confidence = self._find_best_circle_cluster_improved(
                all_circles, frame.shape
            )
            
            if best_center and best_radius and confidence > 0.5:
                # Apply sub-pixel refinement
                refined_center, refined_radius = self._refine_circle_subpixel(
                    gray, best_center, best_radius, blur_size
                )
                
                # Apply temporal filtering
                filtered_center, filtered_radius = self._apply_temporal_filter(refined_center, refined_radius)
                
                # Convert to integers for OpenCV
                self.reticle_center = (int(filtered_center[0]), int(filtered_center[1]))
                self.reticle_radius = int(filtered_radius)
                self.detection_confidence = confidence
                logger.info(f"3-circle reticle detected: center={self.reticle_center}, radius={self.reticle_radius}, confidence={confidence:.2f}")
                return True

        return False

    def _refine_circle_subpixel(self, gray, center, radius, blur_size):
        """Refine circle detection to sub-pixel accuracy using moments."""
        try:
            x, y = int(center[0]), int(center[1])
            r = int(radius)
            
            # Create ROI around the circle
            margin = 20
            y1 = max(0, y - r - margin)
            y2 = min(gray.shape[0], y + r + margin)
            x1 = max(0, x - r - margin)
            x2 = min(gray.shape[1], x + r + margin)
            
            if x2 <= x1 or y2 <= y1:
                return center, radius
                
            roi = gray[y1:y2, x1:x2]
            
            # Create binary mask for the circle
            h, w = roi.shape
            y_center = y - y1
            x_center = x - x1
            y_grid, x_grid = np.ogrid[:h, :w]
            mask = (x_grid - x_center)**2 + (y_grid - y_center)**2 <= r**2
            
            # Find edges in the ROI
            edges = cv2.Canny(roi, 50, 150)
            edge_pixels = np.column_stack(np.where(edges > 0))
            
            if len(edge_pixels) < 10:
                return center, radius
            
            # Filter edge pixels near the circle perimeter
            distances = np.sqrt((edge_pixels[:, 1] - x_center)**2 + (edge_pixels[:, 0] - y_center)**2)
            valid_indices = np.abs(distances - r) < 5
            edge_pixels = edge_pixels[valid_indices]
            
            if len(edge_pixels) < 10:
                return center, radius
            
            # Use least squares to find better circle fit
            # For a circle: (x - cx)^2 + (y - cy)^2 = r^2
            # Rearranging: x^2 + y^2 = 2*cx*x + 2*cy*y - (cx^2 + cy^2 - r^2)
            xy = edge_pixels[:, 1]  # x coordinates
            yy = edge_pixels[:, 0]  # y coordinates
            
            A = np.vstack([2*xy, 2*yy, np.ones(len(xy))]).T
            b = xy**2 + yy**2
            
            try:
                coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
                cx_refined = coeffs[0] + x1
                cy_refined = coeffs[1] + y1
                r_refined = np.sqrt(coeffs[0]**2 + coeffs[1]**2 - coeffs[2])
                
                # Validate refinement
                if abs(cx_refined - center[0]) < 30 and abs(cy_refined - center[1]) < 30 and abs(r_refined - radius) < 30:
                    return (cx_refined, cy_refined), r_refined
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Sub-pixel refinement failed: {e}")
        
        return center, radius
    
    def _apply_temporal_filter(self, center, radius):
        """Apply temporal filtering to smooth reticle position over time."""
        # Convert to float to ensure consistent types
        if isinstance(center, (tuple, list, np.ndarray)) and len(center) >= 2:
            center = (float(center[0]), float(center[1]))
        radius = float(radius)
        
        # Add current detection to history
        self.reticle_position_history.append((center, radius))
        
        # Keep only recent history
        if len(self.reticle_position_history) > self.max_history_size:
            self.reticle_position_history.pop(0)
        
        # Calculate weighted average (recent detections have more weight)
        if len(self.reticle_position_history) == 1:
            return center, radius
        
        # Use exponential weighting: more recent = higher weight
        weights = np.exp(np.linspace(-2, 0, len(self.reticle_position_history)))
        weights = weights / np.sum(weights)
        
        # Handle both tuple and scalar cases for center
        centers_x = []
        centers_y = []
        radii = []
        for c, r in self.reticle_position_history:
            if isinstance(c, (tuple, list, np.ndarray)) and len(c) >= 2:
                centers_x.append(float(c[0]))
                centers_y.append(float(c[1]))
            else:
                centers_x.append(float(c))
                centers_y.append(float(c))
            radii.append(float(r))
        
        centers_x = np.array(centers_x)
        centers_y = np.array(centers_y)
        radii = np.array(radii)
        
        # Weighted average
        filtered_center_x = np.average(centers_x, weights=weights)
        filtered_center_y = np.average(centers_y, weights=weights)
        filtered_radius = np.average(radii, weights=weights)
        
        return (float(filtered_center_x), float(filtered_center_y)), float(filtered_radius)
    
    def _find_best_circle_cluster_improved(self, circles, frame_shape):
        """Find the best circle cluster with improved confidence scoring."""
        if len(circles) < 1:
            return None, None, 0.0
        
        # Keep float precision
        circles_float = [(float(c[0]), float(c[1]), float(c[2])) for c in circles]
        
        # Group circles by proximity (similar centers)
        clusters = []
        used = set()
        
        for i, (x1, y1, r1) in enumerate(circles_float):
            if i in used:
                continue
                
            cluster = [(x1, y1, r1)]
            used.add(i)
            
            # Find nearby circles
            for j, (x2, y2, r2) in enumerate(circles_float):
                if j in used:
                    continue
                    
                # Check if circles are close (within configurable threshold)
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance < self.circle_cluster_threshold:
                    cluster.append((x2, y2, r2))
                    used.add(j)
            
            clusters.append(cluster)
        
        # Score each cluster based on multiple criteria
        best_cluster = None
        best_score = -1
        
        for cluster in clusters:
            if len(cluster) < 1:
                continue
                
            # Calculate average center and radius
            avg_x = sum(c[0] for c in cluster) / len(cluster)
            avg_y = sum(c[1] for c in cluster) / len(cluster)
            avg_r = sum(c[2] for c in cluster) / len(cluster)
            
            # Score based on multiple criteria
            score = 0.0
            max_score = 0.0
            
            # 1. Prefer clusters closer to image center
            center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2
            distance_from_center = math.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)
            center_score = max(0.0, 100.0 - distance_from_center / 5.0)
            score += center_score
            max_score += 100.0
            
            # 2. Prefer reasonable radius sizes
            if self.min_radius <= avg_r <= self.max_radius:
                radius_score = 100.0 - abs(avg_r - 75.0) / 2.0
                score += radius_score
            max_score += 100.0
            
            # 3. Bonus for multiple circles in cluster (3-circle pattern)
            if len(cluster) >= 2:
                score += 50.0
            if len(cluster) >= 3:
                score += 100.0
            max_score += 150.0
            
            # 4. Prefer circles with consistent radii (concentric circles)
            if len(cluster) > 1:
                radii = [c[2] for c in cluster]
                radius_variance = np.var(radii)
                consistency_score = max(0.0, 50.0 - radius_variance / 10.0)
                score += consistency_score
            max_score += 50.0
            
            # 5. Prefer circles that are well within frame bounds
            margin = 50
            if (margin < avg_x < frame_shape[1] - margin and 
                margin < avg_y < frame_shape[0] - margin):
                score += 25.0
            max_score += 25.0
            
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        # Calculate confidence (normalized score)
        if best_cluster and best_score > 50:
            confidence = best_score / max_score if max_score > 0 else 0.0
            # Return the largest circle from the best cluster
            best_circle = max(best_cluster, key=lambda c: c[2])
            return (float(best_circle[0]), float(best_circle[1])), float(best_circle[2]), confidence
        
        # Fallback: return the circle closest to image center
        if circles_float:
            center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2
            closest = min(circles_float, key=lambda c: math.sqrt((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2))
            return (float(closest[0]), float(closest[1])), float(closest[2]), 0.3
        
        return None, None, 0.0
    
    def _find_best_circle_cluster(self, circles, frame_shape):
        """Find the best circle cluster that matches a 3-circle reticle pattern."""
        if len(circles) < 1:
            return None, None
        
        # Convert to list of (x, y, radius) tuples
        circles = [(int(c[0]), int(c[1]), int(c[2])) for c in circles]
        
        # Group circles by proximity (similar centers)
        clusters = []
        used = set()
        
        for i, (x1, y1, r1) in enumerate(circles):
            if i in used:
                continue
                
            cluster = [(x1, y1, r1)]
            used.add(i)
            
            # Find nearby circles
            for j, (x2, y2, r2) in enumerate(circles):
                if j in used:
                    continue
                    
                # Check if circles are close (within configurable threshold)
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance < self.circle_cluster_threshold:
                    cluster.append((x2, y2, r2))
                    used.add(j)
            
            clusters.append(cluster)
        
        # Score each cluster based on SkyWatcher reticle v2 characteristics
        best_cluster = None
        best_score = -1
        
        for cluster in clusters:
            if len(cluster) < 1:
                continue
                
            # Calculate average center and radius
            avg_x = sum(c[0] for c in cluster) / len(cluster)
            avg_y = sum(c[1] for c in cluster) / len(cluster)
            avg_r = sum(c[2] for c in cluster) / len(cluster)
            
            # Score based on multiple criteria
            score = 0
            
            # 1. Prefer clusters closer to image center (but not exactly center)
            center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2
            distance_from_center = math.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)
            center_score = max(0, 100 - distance_from_center / 5)  # Prefer circles not too far from center
            score += center_score
            
            # 2. Prefer reasonable radius sizes (25-150 pixels)
            if self.min_radius <= avg_r <= self.max_radius:
                radius_score = 100 - abs(avg_r - 75) / 2  # Prefer radius around 75 pixels
                score += radius_score
            
            # 3. Bonus for multiple circles in cluster (3-circle pattern)
            if len(cluster) >= 2:
                score += 50
            if len(cluster) >= 3:
                score += 100
            
            # 4. Prefer circles with consistent radii (concentric circles)
            if len(cluster) > 1:
                radii = [c[2] for c in cluster]
                radius_variance = np.var(radii)
                consistency_score = max(0, 50 - radius_variance / 10)
                score += consistency_score
            
            # 5. Prefer circles that are well within frame bounds
            margin = 50
            if (margin < avg_x < frame_shape[1] - margin and 
                margin < avg_y < frame_shape[0] - margin):
                score += 25
            
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        if best_cluster and best_score > 50:  # Minimum threshold
            # Return the largest circle from the best cluster
            best_circle = max(best_cluster, key=lambda c: c[2])
            return (best_circle[0], best_circle[1]), best_circle[2]
        
        # Fallback: return the circle closest to image center
        if circles:
            center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2
            closest = min(circles, key=lambda c: math.sqrt((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2))
            return (closest[0], closest[1]), closest[2]
        
        return None, None

    def calculate_polaris_position(self):
        """Compute Polaris position relative to the reticle with correct astronomical time."""
        try:
            # Current UTC time
            utc_time = Time(datetime.datetime.now(datetime.timezone.utc), scale="utc")

            # Accurate coordinates of Polaris (J2000)
            polaris = SkyCoord(ra=2.53030102 * u.hourangle, dec=89.26413805 * u.deg, frame="icrs")

            # Local sidereal time and hour angle calculation
            # This properly accounts for:
            # 1. GPS position (latitude, longitude, altitude)
            # 2. UTC time
            # 3. Local sidereal time based on longitude
            lst = utc_time.sidereal_time("apparent", self.location.lon)
            ha = lst - polaris.ra
            
            # Calculate Hour Angle using LOCAL SIDEREAL TIME (LST)
            # This is the CORRECT astronomical method for polar alignment
            # Hour Angle = LST - RA (both in hours)
            # This makes Polaris rotate COUNTERCLOCKWISE as it should
            
            # Get AltAz coordinates for distance calculation
            altaz_frame = AltAz(obstime=utc_time, location=self.location)
            polaris_altaz = polaris.transform_to(altaz_frame)
            
            # Calculate Hour Angle: HA = LST - RA
            ha_hours = lst.hour - polaris.ra.hour
            
            # Normalize to 0-24 hours
            if ha_hours < 0:
                ha_hours += 24
            if ha_hours >= 24:
                ha_hours -= 24
            
            # Convert to degrees for angle calculations
            ha_deg = ha_hours * 15.0  # 1 hour = 15 degrees
            
            # Distance from celestial pole (90° - altitude)
            distance_from_pole = 90.0 - polaris_altaz.alt.degree

            # Convert to reticle coordinates - use fallback center if no reticle detected
            if self.reticle_center and self.reticle_radius:
                center_x, center_y = self.reticle_center
                radius = self.reticle_radius
            else:
                # Fallback: use image center and default radius for Polaris calculation
                center_x, center_y = 320, 240  # Default center for 640x480
                radius = 100  # Default radius
                logger.info("Using fallback center for Polaris calculation - reticle not detected")
            
            # For polar scope reticle, Polaris should be on a FIXED circle
            # Not scaled by actual distance from pole
            # Use the middle circle of the 3-circle reticle pattern (97% of outer radius)
            polaris_radius = radius * 0.97  # Middle circle of reticle
            
            # CORRECT METHOD for polar scope with specific orientation
            # Formula fine-tuned to match professional polar align app:
            # ha_12h = ((24 - HA) / 2 + 6.3) % 12
            # Offset 6.3 verified to match professional app reading (7h18)
            azimuth_deg = polaris_altaz.az.degree
            ha_12h = ((24 - ha_hours) / 2.0 + 6.3) % 12
            
            # Calculate visual angle for drawing
            # Use the hour angle converted to degrees for position on reticle
            # The position rotates as: HA increases → Polaris moves westward (right)
            visual_angle_deg = ((24 - ha_hours) / 2.0 + 6.3) * 30.0  # Convert 12h clock to degrees
            angle_rad = math.radians(visual_angle_deg)

            # Calculate position on reticle using the fixed Polaris circle radius
            # Standard orientation: 0° = North (top), angles increase clockwise
            # X-axis: sin(angle) for East-West
            # Y-axis: -cos(angle) for North-South (negative for image coordinates)
            target_x = int(center_x + polaris_radius * math.sin(angle_rad))
            target_y = int(center_y - polaris_radius * math.cos(angle_rad))

            # Log the calculation details for debugging
            logger.info(f"UTC: {utc_time.iso}")
            logger.info(f"LST: {lst.hour:.3f}h")
            logger.info(f"Polaris RA: {polaris.ra.hour:.3f}h")
            logger.info(f"HA (hours): {ha_hours:.3f}h")
            logger.info(f"HA (deg): {ha_deg:.1f}°")
            logger.info(f"Azimuth: {azimuth_deg:.2f}°")
            logger.info(f"Polaris Clock Position: {ha_12h:.3f}h (12h format, from azimuth)")

            return (target_x, target_y, ha_deg, ha_12h, distance_from_pole, polaris_altaz.alt.degree, polaris_altaz.az.degree)
        except Exception as e:
            logger.error(f"Error computing Polaris position: {e}")

        return None

    def calculate_direction(self, target_pos):
        """Compute movement direction and offset from reticle center to target position."""
        if not target_pos:
            return "UNKNOWN", (0, 0)

        # Use reticle center if available, otherwise use fallback center
        if self.reticle_center:
            center_x, center_y = self.reticle_center
        else:
            # Fallback center (same as in calculate_polaris_position)
            center_x, center_y = 320, 240

        dx = target_pos[0] - center_x
        dy = target_pos[1] - center_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 12:
            return "CENTER", (0, 0)

        # Determine primary direction based on largest displacement
        if abs(dx) > abs(dy):
            direction = "WEST" if dx > 0 else "EAST"
        else:
            direction = "SOUTH" if dy > 0 else "NORTH"

        return direction, (dx, dy)

    def draw_overlay(self, frame, target_pos, ha_deg, ha_12h, distance_from_pole, direction):
        """Draw reticle, markings, and info overlay on frame with 3-circle detection indicators."""
        # Store local copies to avoid threading issues
        reticle_center = self.reticle_center
        reticle_radius = self.reticle_radius
        
        if reticle_center and reticle_radius:
            # Main reticle (green) - SkyWatcher style with VERY CLOSE circles like SkyWatcher reticle v2
            cv2.circle(frame, reticle_center, reticle_radius, (0, 255, 0), 1)
            
            # Draw 3 concentric circles VERY CLOSE together to match SkyWatcher reticle v2
            # SkyWatcher reticles have circles very close together (within a few pixels)
            # Inner circle - just slightly smaller
            inner_radius = int(reticle_radius * 0.95)  # Very close to outer
            cv2.circle(frame, reticle_center, inner_radius, (0, 220, 0), 1)
            
            # Middle circle - between inner and outer
            middle_radius = int(reticle_radius * 0.97)  # Very close spacing
            cv2.circle(frame, reticle_center, middle_radius, (0, 200, 0), 1)
            
            # Outer circle (main reticle already drawn)
            
            # Center cross - enhanced visibility
            cross_size = 15
            cv2.line(frame,
                     (reticle_center[0] - cross_size, reticle_center[1]),
                     (reticle_center[0] + cross_size, reticle_center[1]),
                     (0, 255, 0), 2)
            cv2.line(frame,
                     (reticle_center[0], reticle_center[1] - cross_size),
                     (reticle_center[0], reticle_center[1] + cross_size),
                     (0, 255, 0), 2)
            
            # Center dot
            cv2.circle(frame, reticle_center, 3, (0, 255, 0), -1)

            # Graduations de 10 minutes (marques fines)
            for minute in range(0, 60, 10):
                # Angle: 12h = haut (0°), rotation horaire
                angle_deg = minute * 6 - 90  # 6° par 10 minutes, -90 pour aligner 12h en haut
                angle_rad = math.radians(angle_deg)
                
                # Sauter les positions des heures (seront dessinées après)
                if minute % 60 == 0:
                    continue
                
                # Marque courte pour les 10 minutes
                tick_length = 8
                x1 = int(reticle_center[0] + (reticle_radius - tick_length) * math.cos(angle_rad))
                y1 = int(reticle_center[1] + (reticle_radius - tick_length) * math.sin(angle_rad))
                x2 = int(reticle_center[0] + reticle_radius * math.cos(angle_rad))
                y2 = int(reticle_center[1] + reticle_radius * math.sin(angle_rad))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)  # Ligne plus fine et plus claire
            
            # Marques d'heures complètes (1H à 12H)
            for hour in range(1, 13):
                # Angle: 12h = haut (0°), rotation horaire
                angle_deg = hour * 30 - 90  # 30° par heure, -90 pour aligner 12h en haut
                angle_rad = math.radians(angle_deg)
                
                # Marque longue pour les heures
                tick_length = 20
                x1 = int(reticle_center[0] + (reticle_radius - tick_length) * math.cos(angle_rad))
                y1 = int(reticle_center[1] + (reticle_radius - tick_length) * math.sin(angle_rad))
                x2 = int(reticle_center[0] + reticle_radius * math.cos(angle_rad))
                y2 = int(reticle_center[1] + reticle_radius * math.sin(angle_rad))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label de l'heure
                label = f"{hour}H"
                label_x = int(reticle_center[0] + (reticle_radius + 25) * math.cos(angle_rad))
                label_y = int(reticle_center[1] + (reticle_radius + 25) * math.sin(angle_rad))
                
                # Ajuster la position verticale du texte pour centrage
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_x -= text_size[0] // 2
                label_y += text_size[1] // 2
                
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add detection status indicator with confidence
            status_color = (0, 255, 0) if self.reticle_locked else (0, 255, 255)
            confidence_text = f"CONF: {self.detection_confidence:.2f}" if self.detection_confidence > 0 else "SCANNING"
            status_text = "3-CIRCLE DETECTED" if self.reticle_locked else "3-CIRCLE DETECTING"
            cv2.putText(frame, status_text, (reticle_center[0] - 80, reticle_center[1] - reticle_radius - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            cv2.putText(frame, confidence_text, (reticle_center[0] - 50, reticle_center[1] - reticle_radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Always show Polaris position if calculated, even if reticle not detected
        if target_pos:
            # CERCLE ROUGE VIDE pour aligner Polaris visuellement
            # Cercle principal vide (diamètre ~30px) pour voir l'étoile à travers
            cv2.circle(frame, target_pos, 15, (0, 0, 255), 2)  # Cercle rouge vide (ligne épaisse)
            
            # Cercle intérieur plus petit pour aide au centrage
            cv2.circle(frame, target_pos, 8, (0, 0, 255), 1)   # Cercle intérieur fin
            
            # Petite croix centrale fine pour centrage précis
            cross_size = 4
            cv2.line(frame, (target_pos[0] - cross_size, target_pos[1]), 
                     (target_pos[0] + cross_size, target_pos[1]), (0, 0, 255), 1)
            cv2.line(frame, (target_pos[0], target_pos[1] - cross_size), 
                     (target_pos[0], target_pos[1] + cross_size), (0, 0, 255), 1)
            
            # Label "POLARIS" à côté du cercle
            label_pos = (target_pos[0] + 20, target_pos[1] - 15)
            text_size = cv2.getTextSize("POLARIS", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, 
                         (label_pos[0] - 3, label_pos[1] - text_size[1] - 3),
                         (label_pos[0] + text_size[0] + 3, label_pos[1] + 3),
                         (0, 0, 0), -1)
            cv2.putText(frame, "POLARIS", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Show "POLARIS NOT VISIBLE" if no reticle detected
            if not self.reticle_center:
                cv2.putText(frame, "POLARIS NOT VISIBLE - NO RETICLE DETECTED", (20, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Compact info overlay in top-right corner (less obstructive)
        overlay_height = 100
        overlay_width = 250
        overlay_x = frame.shape[1] - overlay_width - 10  # Top-right corner
        overlay_y = 10
        
        # Semi-transparent dark background
        overlay_region = frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width].copy()
        cv2.rectangle(frame, (overlay_x, overlay_y), (overlay_x+overlay_width, overlay_y+overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_region, 0.3, 
                       frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width], 0.7, 0,
                       frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width])
        
        # Compact direction info
        direction_color = (0, 255, 0) if direction == "CENTER" else (255, 255, 0)
        cv2.putText(frame, f"Dir: {direction}", (overlay_x + 5, overlay_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_color, 1)
        
        # Compact time and status display - ALWAYS SHOW
        if ha_12h is not None:
            # Use pre-calculated 12-hour clock position
            h_12 = int(ha_12h)
            m_12 = int((ha_12h - h_12) * 60)
            time_12_str = f"{h_12:02d}:{m_12:02d}"
            
            cv2.putText(frame, f"Time: {time_12_str}", (overlay_x + 5, overlay_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"HA: {ha_deg:.1f}°" if ha_deg else "HA: --", (overlay_x + 5, overlay_y + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Show that calculation is happening
            cv2.putText(frame, "Time: --:--", (overlay_x + 5, overlay_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, "HA: calculating...", (overlay_x + 5, overlay_y + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Reticle detection status - compact
        reticle_status = "LOCK" if self.reticle_locked else "SCAN" if self.reticle_center else "NO RETICLE"
        reticle_color = (0, 255, 0) if self.reticle_locked else (255, 255, 0) if self.reticle_center else (0, 0, 255)
        cv2.putText(frame, reticle_status, (overlay_x + 5, overlay_y + 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, reticle_color, 1)
        
        # Show if Polaris is visible
        if target_pos:
            cv2.putText(frame, "Polaris: VISIBLE", (overlay_x + 5, overlay_y + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Polaris: CALC...", (overlay_x + 5, overlay_y + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def create_polaris_display(self, ha_deg, ha_12h, distance_from_pole, polaris_alt, polaris_az):
        """Create a separate SkyWatcher-style reticle display with 3-circle pattern and Polaris position."""
        # Create a black image for the polaris display (350x350 pixels - larger for better visibility)
        display_size = 350
        display_frame = np.zeros((display_size, display_size, 3), dtype=np.uint8)

        center = (display_size // 2, display_size // 2)
        radius = display_size // 2 - 30

        # Draw 3 concentric circles - SkyWatcher reticle v2 style
        # Outer circle (main reticle)
        cv2.circle(display_frame, center, radius, (0, 255, 0), 2)
        
        # Middle circle
        middle_radius = int(radius * 0.75)
        cv2.circle(display_frame, center, middle_radius, (0, 200, 0), 1)
        
        # Inner circle
        inner_radius = int(radius * 0.5)
        cv2.circle(display_frame, center, inner_radius, (0, 150, 0), 1)

        # Draw center cross - enhanced
        cross_size = 15
        cv2.line(display_frame, (center[0] - cross_size, center[1]), (center[0] + cross_size, center[1]), (0, 255, 0), 2)
        cv2.line(display_frame, (center[0], center[1] - cross_size), (center[0], center[1] + cross_size), (0, 255, 0), 2)
        
        # Center dot
        cv2.circle(display_frame, center, 3, (0, 255, 0), -1)

        # Draw cardinal directions
        directions = [
            (0, "S", (0, 1)),     # South (bottom)
            (90, "W", (-1, 0)),   # West (left)
            (180, "N", (0, -1)),  # North (top)
            (270, "E", (1, 0))    # East (right)
        ]

        for angle, label, (dx_sign, dy_sign) in directions:
            x1 = int(center[0] + (radius - 8) * dx_sign)
            y1 = int(center[1] + (radius - 8) * dy_sign)
            x2 = int(center[0] + radius * dx_sign)
            y2 = int(center[1] + radius * dy_sign)
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            label_x = int(center[0] + (radius + 10) * dx_sign)
            label_y = int(center[1] + (radius + 10) * dy_sign) + 5
            cv2.putText(display_frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw hour markings (every 2 hours) - FIXED: Polar scope orientation
        for i in range(12):
            hour = i * 2
            angle_deg = hour * 15 - 90  # 15 degrees per hour, -90 to align 0h with top (North)
            angle_rad = math.radians(angle_deg)

            tick_length = 8 if hour % 4 == 0 else 6
            x1 = int(center[0] + (radius - tick_length) * math.cos(angle_rad))
            y1 = int(center[1] + (radius - tick_length) * math.sin(angle_rad))
            x2 = int(center[0] + radius * math.cos(angle_rad))
            y2 = int(center[1] + radius * math.sin(angle_rad))
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Hour labels for main hours
            if hour % 4 == 0:
                label_x = int(center[0] + (radius + 15) * math.cos(angle_rad))
                label_y = int(center[1] + (radius + 15) * math.sin(angle_rad)) + 5
                cv2.putText(display_frame, f"{hour:02d}h", (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1)

        # Calculate Polaris position on the display reticle
        if ha_12h is not None and distance_from_pole is not None:
            # Normalize distance (0-45 degrees -> 0-1)
            radius_norm = min(distance_from_pole / 45.0, 1.0)
            display_radius = int(radius * radius_norm)

            # Calculate position using the corrected 12-hour clock position
            # Convert ha_12h back to degrees (each hour = 30°)
            # Orientation: 0h/12h at top (North), rotating clockwise
            visual_angle_deg = ha_12h * 30.0  # Convert 12h clock to degrees
            angle_rad = math.radians(visual_angle_deg - 90)  # -90 to align 0h with top
            polaris_x = int(center[0] + display_radius * math.cos(angle_rad))
            polaris_y = int(center[1] + display_radius * math.sin(angle_rad))

            # Draw Polaris marker - CERCLE ROUGE VIDE pour alignement visuel
            # Cercle principal vide pour voir l'étoile à travers
            cv2.circle(display_frame, (polaris_x, polaris_y), 12, (0, 0, 255), 2)
            
            # Cercle intérieur plus petit pour aide au centrage
            cv2.circle(display_frame, (polaris_x, polaris_y), 6, (0, 0, 255), 1)

            # Petite croix centrale pour centrage précis
            cv2.line(display_frame, (polaris_x - 3, polaris_y), (polaris_x + 3, polaris_y), (0, 0, 255), 1)
            cv2.line(display_frame, (polaris_x, polaris_y - 3), (polaris_x, polaris_y + 3), (0, 0, 255), 1)

            # Add Polaris label
            cv2.putText(display_frame, "POLARIS", (polaris_x + 15, polaris_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Add title and information
        cv2.putText(display_frame, "POLARIS POSITION", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if ha_12h is not None:
            # Use pre-calculated 12-hour clock position
            h = int(ha_12h)
            m = int((ha_12h - h) * 60)
            s = int(((ha_12h - h) * 60 - m) * 60)
            time_str = f"{h:02d}h{m:02d}m{s:02d}s"
            
            # Also show the clock position
            clock_hour = int(ha_12h)
            if clock_hour == 0:
                clock_str = "12 o'clock"
            else:
                clock_str = f"{clock_hour} o'clock"
            
            cv2.putText(display_frame, f"Time: {time_str}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Position: {clock_str}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, f"HA: {ha_deg:.1f}°" if ha_deg else "HA: --", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if distance_from_pole is not None:
            cv2.putText(display_frame, f"Dist: {distance_from_pole:.1f}°", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if polaris_alt is not None:
            cv2.putText(display_frame, f"Alt: {polaris_alt:.1f}°", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if polaris_az is not None:
            cv2.putText(display_frame, f"Az: {polaris_az:.1f}°", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return display_frame

    def process_frame(self):
        """Main loop: read frames, detect reticle, compute Polaris, draw overlay."""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.03)
                continue

            # Apply distortion correction if available
            frame = self.undistort_frame(frame)
            
            # Apply zoom if enabled
            frame = self.apply_zoom(frame)
            frame_count += 1

            # Try to track or detect reticle
            if self.reticle_locked:
                # Track reticle every frame once locked (fast ROI-based tracking)
                if frame_count % 2 == 0:  # Track every 2nd frame for performance
                    self.detect_reticle(frame)  # This will use fast tracking
            elif self.detection_attempts < self.max_detection_attempts:
                # Try detection every few frames for better performance
                if frame_count % 3 == 0:  # Every 3rd frame instead of every frame
                    if self.detect_reticle(frame):
                        # Lock the reticle once detected
                        self.reticle_locked = True
                        logger.info("Reticle locked successfully")
                    else:
                        self.detection_attempts += 1
                        if self.detection_attempts >= self.max_detection_attempts:
                            logger.warning("Max reticle detection attempts reached. Please check camera alignment.")

            # Compute Polaris position - always calculate, even without reticle
            polaris_data = self.calculate_polaris_position()
            if polaris_data:
                target_x, target_y, ha_deg, ha_12h, distance_from_pole, polaris_alt, polaris_az = polaris_data
                target_pos = (target_x, target_y)
                direction, offset = self.calculate_direction(target_pos)
                # Log every 30 frames for debugging
                if frame_count % 30 == 0:
                    logger.info(f"Polaris at ({target_x}, {target_y}), HA={ha_deg:.1f}°, Clock={ha_12h:.2f}h, Distance={distance_from_pole:.1f}°")
            else:
                target_pos = None
                ha_deg = None
                ha_12h = None
                distance_from_pole = None
                polaris_alt = None
                polaris_az = None
                direction, offset = "UNKNOWN", (0, 0)
                # Log error
                if frame_count % 30 == 0:
                    logger.warning("Polaris calculation returned None!")

            # Draw overlay elements on main frame (always draw, even if data is None)
            self.draw_overlay(frame, target_pos, ha_deg, ha_12h, distance_from_pole, direction)

            # Create separate Polaris display
            polaris_display = self.create_polaris_display(ha_deg, ha_12h, distance_from_pole, polaris_alt, polaris_az)

            # Update state (thread-safe)
            with self.lock:
                self.current_frame = frame.copy()
                self.polaris_display_frame = polaris_display
                self.direction = direction
                self.offset = offset
                self.polaris_data = {
                    "ha": ha_deg,
                    "distance": distance_from_pole,
                    "target_pos": target_pos,
                    "altitude": polaris_alt,
                    "azimuth": polaris_az
                }

            time.sleep(1.0 / self.camera_fps)

    def get_frame(self):
        """Return current JPEG-encoded frame bytes, or None."""
        with self.lock:
            if self.current_frame is not None:
                ret, buffer = cv2.imencode(".jpg", self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    return buffer.tobytes()
            return None

    def get_polaris_display_frame(self):
        """Return current Polaris display frame bytes, or None."""
        with self.lock:
            if self.polaris_display_frame is not None:
                ret, buffer = cv2.imencode(".jpg", self.polaris_display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buffer.tobytes()
            return None

    def get_status(self):
        """Return current status as a dict."""
        with self.lock:
            # Get current time and location info
            utc_time = Time(datetime.datetime.now(datetime.timezone.utc), scale="utc")
            lst = utc_time.sidereal_time("apparent", self.location.lon)
            
            status = {
                "direction": self.direction,
                "offset_x": int(self.offset[0]),
                "offset_y": int(self.offset[1]),
                "reticle_detected": self.reticle_center is not None,
                "reticle_locked": self.reticle_locked,
                "detection_confidence": float(self.detection_confidence),
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
                "utc_time": utc_time.iso,
                "local_sidereal_time": f"{lst.hour:.3f}h",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            if self.polaris_data is not None:
                status.update(self.polaris_data)
            return status

    def recalibrate(self):
        """Reset reticle detection to force recalibration."""
        with self.lock:
            self.reticle_center = None
            self.reticle_radius = None
            self.reticle_locked = False  # Unlock reticle for recalibration
            self.reticle_position_history = []  # Clear temporal history
            self.detection_confidence = 0.0  # Reset confidence
            # IMPORTANT: Reset detection attempts counter so detection can restart
            self.detection_attempts = 0
        logger.info("Recalibration requested - reticle detection will restart")

    def run(self):
        """Start processing thread."""
        processing_thread = threading.Thread(target=self.process_frame, name="FrameProcessor", daemon=True)
        processing_thread.start()
        logger.info("Image processing started")


# Flask app initialization
config = load_config()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Important pour l'UTF-8
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


@app.route("/")
def index():
    try:
        config = load_config()
        return render_template("index.html", assistant=assistant, config=config._sections)
    except UnicodeDecodeError as e:
        logger.error(f"Template encoding error: {e}")
        return "Template encoding error. Please save index.html as UTF-8.", 500
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
    """Set zoom level"""
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
    """Get current zoom level"""
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


# Nouvelles routes pour les paramètres
@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
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
    """Update camera settings"""
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
    """Update calibration settings"""
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
    """Update location settings"""
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


if __name__ == "__main__":
    assistant.run()
    logger.info("Web server started at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)