import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from collections import deque
import time
import winsound as ws
import twilio as tw

n_time_steps = 40
lm_list = deque(maxlen=n_time_steps)

label = "Normal"
prediction_confidence = 0.0
is_predicting = False
frame_counter = 0

last_beep_time = 0
beep = 1.0  

url = "rtsp://admin:L2EB9D1A@192.168.1.108:37777/cam/realmonitor?channel=1&subtype=1"

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,        
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.6
)


model = tf.keras.models.load_model("model8.h5")

#def send_sms():
    

def exponential_smoothing(old, new, alpha=0.3):
    """Filter làm mượt dự đoán."""
    return alpha * new + (1 - alpha) * old


def play_alert_sound():
    global last_beep_time

    now = time.time()
    if now - last_beep_time >= beep:
        ws.Beep(4000, 500)
        last_beep_time = now


def extract_landmarks(results):
    """Lấy pose+hands, nguyên bản, không thay đổi."""
    row = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        row.extend([0.0] * 132)

    # Tay trái
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 63)

    # Tay phải
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 63)

    return row


def draw_all_landmarks(mp_drawing, results, img):
    """Giữ nguyên vẽ như bạn muốn."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    return img


def detect(model, lm_seq):
    """Predict trong thread riêng."""
    global prediction_confidence, label, is_predicting

    try:
        arr = np.array(lm_seq)[np.newaxis, ...]
        res = float(model.predict(arr, verbose=0)[0][0])

        # smoothing để cực mượt
        prediction_confidence = exponential_smoothing(prediction_confidence, res)

        if prediction_confidence > 0.82:
            label = "Heart pain"
        else:
            label = "Normal"

        print(f"[PREDICT] {label} - {prediction_confidence:.3f}")

    except Exception as e:
        print("[ERROR]:", e)

    finally:
        is_predicting = False


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("System started...")


while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize ảnh giúp Mediapipe chạy nhanh hơn
    small = cv2.resize(frame, (480, 320))

    frameRGB = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = holistic.process(frameRGB)

    # Vẽ lên frame gốc
    frame = draw_all_landmarks(mp_draw, results, frame)
    frame = cv2.flip(frame, 1)

    # Lấy landmarks
    lm = extract_landmarks(results)
    lm_list.append(lm)

    frame_counter += 1

    
    if len(lm_list) == n_time_steps and frame_counter % 3 == 0:
        if not is_predicting:
            is_predicting = True
            t = threading.Thread(target=detect, args=(model, list(lm_list)))
            t.daemon = True
            t.start()

    color = (0, 0, 255) if label == "Heart pain" else (0, 255, 0)

    cv2.putText(
        frame,
        f"{label} {prediction_confidence:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color, 3
    )

    # Alert
    if label == "Heart pain" and prediction_confidence > 0.82:
        play_alert_sound()

    cv2.imshow("Heart Pain Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
holistic.close()
