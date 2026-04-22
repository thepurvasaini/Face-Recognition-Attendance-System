"""
╔══════════════════════════════════════════════════════════════╗
║       FACE RECOGNITION ATTENDANCE SYSTEM                     ║
║       File: attendance_system.py  (Main Entry Point)         ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python attendance_system.py

Requirements:
    pip install opencv-python face_recognition numpy pandas
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import time
import pickle
from datetime import datetime
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CONFIG = {
    "camera_index": 0,            # 0 = default webcam
    "frame_width": 640,
    "frame_height": 480,
    "process_every_n_frames": 3,  # process face detection every N frames (speed)
    "recognition_threshold": 0.5, # lower = stricter match (0.0–1.0)
    "cooldown_seconds": 60,       # min seconds between duplicate attendance entries
    "encodings_file": "data/encodings.pkl",
    "attendance_folder": "attendance",
    "show_fps": True,
    "detection_model": "hog",     # "hog" (CPU-fast) or "cnn" (GPU-accurate)
    "scale_factor": 0.5,          # resize frame before detection for speed
}

# ─── COLOURS (BGR for OpenCV) ─────────────────────────────────────────────────

GREEN  = (0, 230, 160)
RED    = (60, 80, 255)
YELLOW = (0, 210, 255)
WHITE  = (240, 240, 240)
BLACK  = (10, 10, 10)
GRAY   = (120, 120, 120)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD KNOWN FACES
# ══════════════════════════════════════════════════════════════════════════════

def load_known_faces(encodings_file: str):
    """
    Load pre-computed face encodings from disk.
    Run register_faces.py first to generate this file.
    """
    path = Path(encodings_file)
    if not path.exists():
        print(f"[ERROR] Encodings file not found: {encodings_file}")
        print("  → Run  python register_faces.py  first to register students.")
        return [], []

    with open(path, "rb") as f:
        data = pickle.load(f)

    names     = data["names"]
    encodings = data["encodings"]
    print(f"[OK] Loaded {len(names)} known face(s): {list(set(names))}")
    return encodings, names


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ATTENDANCE CSV MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_attendance_filepath() -> Path:
    """Return today's attendance CSV path, creating folder if needed."""
    folder = Path(CONFIG["attendance_folder"])
    folder.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return folder / f"attendance_{date_str}.csv"


def load_today_attendance() -> pd.DataFrame:
    """Load today's attendance DataFrame (or create empty one)."""
    filepath = get_attendance_filepath()
    if filepath.exists():
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame(columns=["Name", "Date", "Time", "Confidence", "Status"])


def mark_attendance(name: str, confidence: float, last_marked: dict) -> bool:
    """
    Append an attendance row to today's CSV.
    Returns True if newly marked, False if within cooldown.

    Args:
        name:         Student name
        confidence:   Match confidence 0–100
        last_marked:  Dict tracking last mark timestamp per student
    """
    now = time.time()
    cooldown = CONFIG["cooldown_seconds"]

    # Cooldown check — avoid duplicate rows
    if name in last_marked and (now - last_marked[name]) < cooldown:
        remaining = int(cooldown - (now - last_marked[name]))
        print(f"[SKIP] {name} already marked. Next mark in {remaining}s")
        return False

    last_marked[name] = now

    filepath = get_attendance_filepath()
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    row = pd.DataFrame([{
        "Name":       name,
        "Date":       date_str,
        "Time":       time_str,
        "Confidence": f"{confidence:.1f}%",
        "Status":     "Present",
    }])

    # Append (write header only if file is new)
    row.to_csv(filepath, mode="a", header=not filepath.exists(), index=False)
    print(f"[MARKED] ✓  {name}  |  {time_str}  |  {confidence:.1f}%")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FACE RECOGNITION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def recognize_faces(rgb_frame, known_encodings, known_names,
                    threshold: float):
    """
    Detect and recognize all faces in an RGB frame.

    Returns:
        list of dicts: [{ "name", "confidence", "box" (top,right,bottom,left) }]
    """
    results = []

    # Detect face locations
    face_locations = face_recognition.face_locations(
        rgb_frame, model=CONFIG["detection_model"]
    )
    if not face_locations:
        return results

    # Compute encodings for detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        name       = "Unknown"
        confidence = 0.0

        if known_encodings:
            # Euclidean distances to all known faces
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx  = np.argmin(distances)
            best_dist = distances[best_idx]

            if best_dist <= threshold:
                name       = known_names[best_idx]
                confidence = (1 - best_dist) * 100   # convert distance → %

        results.append({
            "name":       name,
            "confidence": confidence,
            "box":        location,   # (top, right, bottom, left)
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_face_box(frame, result: dict):
    """Draw bounding box, name label, and confidence bar on frame."""
    top, right, bottom, left = result["box"]
    name       = result["name"]
    confidence = result["confidence"]
    known      = name != "Unknown"

    box_color   = GREEN if known else RED
    label_color = GREEN if known else RED

    # ── Bounding box ──
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

    # ── Corner accents (cyber look) ──
    size = 14
    thickness = 2
    for cx, cy, dx, dy in [
        (left, top,     1,  1),
        (right, top,   -1,  1),
        (left, bottom,  1, -1),
        (right, bottom,-1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*size, cy),            box_color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy*size),            box_color, thickness)

    # ── Label background ──
    label      = f"{name}  {confidence:.0f}%" if known else "Unknown"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    cv2.rectangle(frame,
                  (left, top - th - 12),
                  (left + tw + 10, top),
                  box_color, -1)

    # ── Label text ──
    cv2.putText(frame, label,
                (left + 5, top - 5),
                font, font_scale, BLACK, 1, cv2.LINE_AA)

    # ── Confidence bar below box ──
    if known:
        bar_w    = right - left
        filled_w = int(bar_w * confidence / 100)
        cv2.rectangle(frame, (left, bottom + 4), (right, bottom + 8),       GRAY, -1)
        cv2.rectangle(frame, (left, bottom + 4), (left + filled_w, bottom + 8), GREEN, -1)


def draw_hud(frame, fps: float, present_count: int, total_faces: int,
             attendance_file: str):
    """Overlay HUD info on the top-left of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 95), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.45
    med   = 0.55

    cv2.putText(frame, "FACE ATTENDANCE SYSTEM",   (10, 18),  font, small, GREEN,  1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:      {fps:.1f}",      (10, 38),  font, small, WHITE,  1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces:    {total_faces}",  (10, 54),  font, small, WHITE,  1, cv2.LINE_AA)
    cv2.putText(frame, f"Present:  {present_count}",(10, 70),  font, small, GREEN,  1, cv2.LINE_AA)

    time_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, time_str,                    (10, 88),  font, small, YELLOW, 1, cv2.LINE_AA)

    # Bottom bar — CSV file name
    cv2.rectangle(frame, (0, h - 24), (w, h), BLACK, -1)
    cv2.putText(frame, f"CSV: {attendance_file}",
                (8, h - 8), font, 0.4, GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit",
                (w - 130, h - 8), font, 0.4, GRAY, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 60)

    # Load known faces
    known_encodings, known_names = load_known_faces(CONFIG["encodings_file"])

    # Open webcam
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check camera_index in CONFIG.")
        return

    print(f"[OK] Camera opened ({CONFIG['frame_width']}×{CONFIG['frame_height']})")
    print(f"[OK] Recognition threshold: {CONFIG['recognition_threshold']}")
    print(f"[OK] Detection model: {CONFIG['detection_model']}")
    print("[INFO] Press Q to quit, S to save screenshot\n")

    # State
    last_marked    = {}           # name → timestamp
    present_names  = set()
    frame_count    = 0
    last_results   = []           # cache results between processed frames

    # FPS tracking
    fps         = 0.0
    fps_counter = 0
    fps_start   = time.time()

    attendance_file = str(get_attendance_filepath())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame_count += 1

        # ── Process every N frames for performance ──
        if frame_count % CONFIG["process_every_n_frames"] == 0:
            # Resize for faster detection
            scale = CONFIG["scale_factor"]
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            raw_results = recognize_faces(
                rgb, known_encodings, known_names,
                CONFIG["recognition_threshold"]
            )

            # Scale bounding boxes back to original frame size
            last_results = []
            inv = 1.0 / scale
            for r in raw_results:
                top, right, bottom, left = r["box"]
                last_results.append({
                    "name":       r["name"],
                    "confidence": r["confidence"],
                    "box": (
                        int(top    * inv),
                        int(right  * inv),
                        int(bottom * inv),
                        int(left   * inv),
                    ),
                })

            # Mark attendance for recognized people
            for r in last_results:
                if r["name"] != "Unknown":
                    newly = mark_attendance(r["name"], r["confidence"], last_marked)
                    if newly:
                        present_names.add(r["name"])

        # ── Draw detections ──
        for r in last_results:
            draw_face_box(frame, r)

        # ── FPS ──
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps       = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # ── HUD ──
        draw_hud(frame, fps, len(present_names), len(last_results), attendance_file)

        cv2.imshow("Face Attendance System  [Q = quit]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"screenshot_{ts}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[SAVED] Screenshot → {fn}")

    cap.release()
    cv2.destroyAllWindows()

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Present ({len(present_names)}): {', '.join(sorted(present_names)) or 'None'}")
    print(f"  Attendance saved → {attendance_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()