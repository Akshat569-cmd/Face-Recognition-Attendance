import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime

DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"
MATCH_THRESHOLD = 0.6
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Load & encode dataset ─────────────────────────────────────────────────────

def load_known_faces(path: str):
    """Load images and return parallel lists of encodings and names.
    Skips files with no detectable face instead of misaligning indices."""
    encodings, names = [], []

    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VALID_EXTENSIONS:         # ✅ skip non-image files
            continue

        # Open, convert to RGB, and resize large images for dlib compatibility
        from PIL import Image as PILImage
        pil_img = PILImage.open(os.path.join(path, filename)).convert('RGB')
        # Resize if image is too large (max 1000px on longest side)
        max_size = 1000
        if max(pil_img.size) > max_size:
            ratio = max_size / max(pil_img.size)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, PILImage.LANCZOS)
        img = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
        face_locs = face_recognition.face_locations(img)
        face_encs = face_recognition.face_encodings(img, face_locs)

        if not face_encs:
            print(f"[WARN] No face found in '{filename}', skipping.")
            continue

        encodings.append(face_encs[0])
        names.append(os.path.splitext(filename)[0])

    return encodings, names


known_encodings, class_names = load_known_faces(DATASET_PATH)
print(f"[INFO] Loaded {len(class_names)} face(s): {class_names}")

# ── Attendance (in-memory cache to avoid per-frame CSV reads) ─────────────────

# Pre-load or create the CSV once at startup
if os.path.exists(ATTENDANCE_FILE):
    attendance_df = pd.read_csv(ATTENDANCE_FILE)
else:
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

marked_today: set[str] = set(attendance_df["Name"].values)  # ✅ fast O(1) lookup


def mark_attendance(name: str):
    """Mark attendance only once per session; write CSV only on new entries."""
    global attendance_df

    if name in marked_today:        # ✅ in-memory guard — no CSV read per frame
        return

    now = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame([{"Name": name, "Time": now}])
    attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)
    attendance_df.to_csv(ATTENDANCE_FILE, index=False)
    marked_today.add(name)
    print(f"[INFO] Marked attendance: {name} at {now}")


# ── Camera loop ───────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

if not cap.isOpened():              # ✅ check before entering loop
    raise RuntimeError("Could not open camera. Check device index or permissions.")

print("[INFO] Starting camera. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to read frame.")
        break

    # Downscale for faster detection, convert to RGB for face_recognition
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # ✅ correct place for conversion

    face_locs = face_recognition.face_locations(small_rgb, model="hog")
    face_encs = face_recognition.face_encodings(small_rgb, face_locs)

    for enc, loc in zip(face_encs, face_locs):
        name = "UNKNOWN"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))

            if distances[best_idx] < MATCH_THRESHOLD:
                name = class_names[best_idx].upper()
                mark_attendance(name)

        # Scale bounding box back to original frame size
        top, right, bottom, left = [v * 4 for v in loc]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done. Attendance saved to", ATTENDANCE_FILE)
