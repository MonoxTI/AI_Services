import cv2, os, numpy as np, pygame, time, requests
from datetime import datetime

# ===============================
# CONFIG (No heavy AI settings here)
# ===============================
SERVER_URL = "http://192.168.1.50:8001/check_face"  # Address of your Server
NIGHT_THRESHOLD = 60
CHECK_INTERVAL = 2  # Check for faces every 2 seconds

# ===============================
# SOUND INIT (Pi still plays the sound)
# ===============================
pygame.mixer.init()
sound_files = ["sounds/bark1.wav","sounds/bark2.wav"]

def play_sound():
    pygame.mixer.music.load(random.choice(sound_files))
    pygame.mixer.music.play()

# ===============================
# CAMERA INIT
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_check_time = 0

print("📷 Camera Lite Started. Sending to Server...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 1. LIGHT PROCESS (Pi still does this)
    brightness = np.mean(frame)
    if brightness < NIGHT_THRESHOLD:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

    # 2. SEND TO SERVER FOR ANALYSIS (Every 2 seconds)
    current_time = time.time()
    if current_time - last_check_time > CHECK_INTERVAL:
        try:
            # Encode image to send over network
            _, buffer = cv2.imencode(".jpg", frame)
            files = {'file': buffer.tobytes()}
            
            # Send to Server and wait for answer
            response = requests.post(SERVER_URL, files=files, timeout=5)
            result = response.json()
            
            # 3. ACT ON SERVER'S DECISION
            if result.get("is_intruder"):
                print("⚠️ Intruder Detected by Server!")
                play_sound()
                # Note: We do NOT save the log here. Server does it.
            
        except Exception as e:
            print(f"Connection error: {e}")
            
        last_check_time = current_time

    # 4. SHOW LOCAL PREVIEW (Optional)
    cv2.imshow('Pi Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()