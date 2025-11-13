#!/usr/bin/env python3
"""
Camera Detection Utility
Scans all available camera indices and displays information about each camera
"""

import cv2
import sys

def test_camera(index):
    """Test if a camera exists at the given index and get its properties."""
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        return None
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    backend = cap.getBackendName()
    
    # Get additional properties if available
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    
    cap.release()
    
    return {
        'index': index,
        'width': width,
        'height': height,
        'fps': fps,
        'backend': backend,
        'brightness': brightness,
        'contrast': contrast,
        'exposure': exposure,
        'gain': gain,
        'frame_shape': frame.shape
    }

def detect_all_cameras(max_index=10):
    """Detect all available cameras up to max_index."""
    print("=" * 70)
    print("CAMERA DETECTION UTILITY")
    print("=" * 70)
    print()
    
    cameras = []
    
    for i in range(max_index):
        print(f"Testing camera index {i}...", end=" ")
        sys.stdout.flush()
        
        info = test_camera(i)
        
        if info:
            cameras.append(info)
            print("[FOUND]")
            print(f"   Resolution: {info['width']}x{info['height']}")
            print(f"   FPS: {info['fps']}")
            print(f"   Backend: {info['backend']}")
            print(f"   Frame Shape: {info['frame_shape']}")
            
            # Try to identify camera type
            if info['width'] >= 1920 or info['height'] >= 1080:
                print(f"   >> Likely: High-resolution camera (possibly ASI camera)")
            elif info['width'] == 640 and info['height'] == 480:
                print(f"   >> Likely: Standard webcam")
            
            print()
        else:
            print("[Not available]")
    
    print("=" * 70)
    print(f"SUMMARY: Found {len(cameras)} camera(s)")
    print("=" * 70)
    print()
    
    if cameras:
        print("Available cameras:")
        for cam in cameras:
            print(f"  - Index {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps ({cam['backend']})")
        
        print()
        print("INSTRUCTIONS:")
        print("1. Identify which camera is your ASI camera from the list above")
        print("   (ASI cameras typically have higher resolution capabilities)")
        print()
        print("2. Update config.ini with the correct camera index:")
        print("   [CAMERA]")
        print("   index = X  # Replace X with your ASI camera index")
        print()
        print("3. Or run the app with: python polar_align.py --camera-index X")
        print()
        
        # Recommend most likely ASI camera
        high_res_cameras = [c for c in cameras if c['width'] >= 1280]
        if high_res_cameras:
            best_candidate = max(high_res_cameras, key=lambda c: c['width'] * c['height'])
            print(f">> RECOMMENDATION: Camera index {best_candidate['index']} looks like your ASI camera")
            print(f"   ({best_candidate['width']}x{best_candidate['height']})")
    else:
        print("No cameras detected!")
        print("   Please check:")
        print("   - Camera is connected via USB")
        print("   - Camera drivers are installed")
        print("   - Camera is powered on")
    
    print()
    return cameras

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect available cameras')
    parser.add_argument('--max-index', type=int, default=10, 
                       help='Maximum camera index to test (default: 10)')
    parser.add_argument('--test-index', type=int, 
                       help='Test a specific camera index in detail')
    
    args = parser.parse_args()
    
    if args.test_index is not None:
        print(f"Testing camera at index {args.test_index}...")
        cap = cv2.VideoCapture(args.test_index)
        if cap.isOpened():
            print("[OK] Camera opened successfully!")
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Frame captured: {frame.shape}")
                # Save test image
                cv2.imwrite(f"test_camera_{args.test_index}.jpg", frame)
                print(f"[OK] Test image saved as: test_camera_{args.test_index}.jpg")
            else:
                print("[ERROR] Could not capture frame")
            cap.release()
        else:
            print("[ERROR] Could not open camera")
    else:
        detect_all_cameras(args.max_index)

