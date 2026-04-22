import cv2
import face_recognition
import numpy as np
import pickle
import os
import argparse
from pathlib import Path
from datetime import datetime
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
 
KNOWN_FACES_DIR = Path("known_faces")
ENCODINGS_FILE  = Path("data/encodings.pkl")
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAPTURE_COUNT   = 10   # photos to capture per person in --capture mode
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ENCODE FROM FOLDER
# ══════════════════════════════════════════════════════════════════════════════
 
def encode_known_faces():
    """
    Walk known_faces/<Name>/*.jpg and compute 128-D embeddings.
    Saves everything to data/encodings.pkl.
    """
    if not KNOWN_FACES_DIR.exists():
        print(f"[ERROR] Folder not found: {KNOWN_FACES_DIR}")
        print("  Create it and add subfolders named after each student.")
        return
 
    all_encodings = []
    all_names     = []
    total_images  = 0
    failed_images = 0
 
    person_dirs = [d for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()]
    if not person_dirs:
        print(f"[ERROR] No subfolders found inside {KNOWN_FACES_DIR}/")
        return
 
    print(f"\n[SCAN] Found {len(person_dirs)} person(s) to encode...\n")
 
    for person_dir in sorted(person_dirs):
        name   = person_dir.name.replace("_", " ")
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
 
        if not images:
            print(f"  [SKIP] {name} — no valid images found")
            continue
 
        person_encodings = []
 
        for img_path in images:
            total_images += 1
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [WARN] Cannot read: {img_path.name}")
                failed_images += 1
                continue
 
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
            # Detect face locations in this image
            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                print(f"  [WARN] No face detected in {img_path.name}")
                failed_images += 1
                continue
 
            # Encode the first (largest) face found
            enc = face_recognition.face_encodings(rgb, boxes)
            if enc:
                person_encodings.append(enc[0])
 
        if not person_encodings:
            print(f"  [FAIL] {name} — no usable encodings extracted")
            continue
 
        # Average encoding across all photos → more robust representation
        avg_encoding = np.mean(person_encodings, axis=0)
        all_encodings.append(avg_encoding)
        all_names.append(name)
 
        print(f"  [OK]   {name:<25} {len(person_encodings)}/{len(images)} photos encoded")
 
    if not all_encodings:
        print("\n[ERROR] No encodings produced. Check your image files.")
        return
 
    # ── Save to disk ──
    ENCODINGS_FILE.parent.mkdir(exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": all_encodings, "names": all_names}, f)
 
    print(f"\n{'='*50}")
    print(f"  ENCODING COMPLETE")
    print(f"{'='*50}")
    print(f"  People:  {len(all_names)}")
    print(f"  Images:  {total_images}  (failed: {failed_images})")
    print(f"  Saved →  {ENCODINGS_FILE}")
    print(f"{'='*50}\n")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LIVE WEBCAM CAPTURE MODE
# ══════════════════════════════════════════════════════════════════════════════
 
def capture_new_person():
    """
    Interactive webcam mode to register a new person.
    Captures CAPTURE_COUNT photos automatically, then encodes.
    """
    name = input("\nEnter student name (e.g. Priya Nair): ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        return
 
    folder_name = name.replace(" ", "_")
    save_dir    = KNOWN_FACES_DIR / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"\n[INFO] Will capture {CAPTURE_COUNT} photos for: {name}")
    print("[INFO] Face the camera. Press SPACE to capture, Q to quit early.\n")
 
    cap         = cv2.VideoCapture(0)
    captured    = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
 
    while captured < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
 
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces   = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
 
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 230, 160), 2)
 
        status = f"Captured: {captured}/{CAPTURE_COUNT}  |  Press SPACE to capture"
        cv2.putText(display, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 160), 2)
        cv2.putText(display, f"Registering: {name}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
 
        cv2.imshow("Register New Face  [SPACE = capture, Q = done]", display)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and len(faces) > 0:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = save_dir / f"{folder_name}_{ts}.jpg"
            cv2.imwrite(str(filename), frame)
            captured += 1
            print(f"  [SNAP] {captured}/{CAPTURE_COUNT} saved")
        elif key == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
    print(f"\n[INFO] {captured} photos saved to {save_dir}")
    print("[INFO] Re-running encoding...\n")
    encode_known_faces()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Registration Tool")
    parser.add_argument("--capture", action="store_true",
                        help="Live webcam capture mode for new student")
    args = parser.parse_args()
 
    if args.capture:
        capture_new_person()
    else:
        encode_known_faces()