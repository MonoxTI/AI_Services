# server_brain.py
from fastapi import FastAPI, UploadFile, File
import face_recognition, numpy as np, os, cv2
from datetime import datetime

app = FastAPI()

# ===============================
# LOAD KNOWLEDGE (Server does this)
# ===============================
known_encodings = []
known_names = []

# Server loads the faces once at startup
for file in os.listdir("known_faces"):
    if file.endswith((".jpg",".png")):
        img = face_recognition.load_image_file(f"known_faces/{file}")
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(os.path.splitext(file)[0])

print(" Server Brain Ready. Knowledge Loaded.")

# ===============================
# ANALYSIS ENDPOINT
# ===============================
@app.post("/check_face")
async def check_face(file: UploadFile = File(...)):
    # 1. Read image sent by Pi
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Run Face Recognition (Heavy lifting)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    
    is_intruder = False
    intruder_name = "Unknown"

    for face_enc in encodings:
        distances = face_recognition.face_distance(known_encodings, face_enc)
        if len(distances) > 0:
            best_match = np.argmin(distances)
            if distances[best_match] > 0.45:  # Threshold
                is_intruder = True
    
    # 3. Save Log if Intruder (Server storage, not Pi)
    if is_intruder:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/intruder_{ts}.jpg"
        cv2.imwrite(filename, frame)
        print(f" Saved intruder log: {filename}")

    # 4. Tell Pi what to do
    return {"is_intruder": is_intruder}