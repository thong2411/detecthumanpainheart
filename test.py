import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from collections import deque
import time
import winsound as ws
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# THÊM: Biến kiểm soát gửi email
last_email_sent_time = 0
EMAIL_SEND_INTERVAL = 30
REQUIRED_DETECTIONS = 20 
confidence_history = []  

n_time_steps = 40
lm_list = deque(maxlen=n_time_steps)

label = "Normal"
prediction_confidence = 0.0
is_predicting = False
frame_counter = 0

last_beep_time = 0
beep = 1.0  

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,        
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.6
)

model = tf.keras.models.load_model("modelnew.h5")

# CẢI TIẾN HÀM GỬI EMAIL - THÊM TRUNG BÌNH ĐỘ TIN CẬY
def send_email(to_email, subject, message, avg_confidence, detection_count):
    """
    Gửi email thông báo với xử lý lỗi và template HTML
    """
    sender_email = "thongdv2411@gmail.com"
    password = "kmif bvio ysuw oycn"
    
 
    msg = MIMEMultipart('alternative')
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
        
    text_content = message
    html_content = f"""
        <html>
          <head>
            <style>
              body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 0;
                background-color: #f4f4f4;
              }}
              .container {{ 
                max-width: 600px; 
                margin: 20px auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
              }}
              .header {{ 
                background-color: #ff4444; 
                color: white; 
                padding: 30px; 
                text-align: center;
                border-radius: 10px 10px 0 0;
              }}
              .content {{ 
                padding: 30px;
              }}
              .alert-box {{
                background-color: #fff3cd;
                border-left: 4px solid #ff9800;
                padding: 15px;
                margin: 20px 0;
              }}
              .info {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
              }}
              .stats {{
                background-color: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 15px 0;
              }}
              .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 12px;
                border-top: 1px solid #eee;
              }}
              .confidence {{
                font-size: 24px;
                font-weight: bold;
                color: #ff4444;
              }}
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
               <h1>CẢNH BÁO KHẨN CẤP</h1>
              </div>
              <div class="content">
                <div class="alert-box">
                  <h2>Phát hiện dấu hiệu đau tim</h2>
                  <p><strong>Thời gian:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>

                <div class="info">
                  <p><strong>Nguồn:</strong> Camera giám sát AI</p>
                  <p><strong>Trạng thái:</strong> {label}</p>
                </div>
                
                <p style="color: #d32f2f; font-weight: bold; font-size: 16px;">Vui lòng kiểm tra ngay lập tức!</p>
                
                <p>Đề nghị kiểm tra tình trạng người trong khu vực camera ngay lập tức.</p>
              </div>
              <div class="footer">
                <p>Email này được gửi tự động từ hệ thống giám sát AI</p>
                <p>2025 Heart Pain Detection System</p>
              </div>
            </div>
          </body>
        </html>
        """
        
        # Attach cả 2 phiên bản
    part1 = MIMEText(text_content, 'plain', 'utf-8')
    part2 = MIMEText(html_content, 'html', 'utf-8')
        
    msg.attach(part1)
    msg.attach(part2)
        
        # Kết nối và gửi
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, to_email, msg.as_string())
    server.quit()

    return True



def send_email_async(to_email, subject, message, avg_confidence, detection_count):
    """
    Gửi email trong thread riêng để không block camera
    """
    global last_email_sent_time
    
    now = time.time()
    # Kiểm tra interval để tránh spam
    if now - last_email_sent_time < EMAIL_SEND_INTERVAL:
        remaining = EMAIL_SEND_INTERVAL - (now - last_email_sent_time)
        print(f"Chờ {remaining:.0f}s để gửi email tiếp")
        return
    last_email_sent_time = now
    def send():
        if send_email(to_email, subject, message, avg_confidence, detection_count):
            global last_email_sent_time
            last_email_sent_time = time.time()
    
    thread = threading.Thread(target=send, daemon=True)
    thread.start()


# GỬI EMAIL CHO NHIỀU NGƯỜI - CẬP NHẬT VỚI TRUNG BÌNH
def send_alert_emails(avg_confidence, detection_count):
    
    recipients = [
        "quanganh13042005@gmail.com",
        "pubgss9oii@gmail.com"
    ]
    
    subject = f"CẢNH BÁO KHẨN CẤP - Phát hiện đau tim ({detection_count} lần)"
    message = f"""
    CẢNH BÁO KHẨN CẤP!
    
    Hệ thống AI đã phát hiện dấu hiệu đau tim trong khu vực camera giám sát.
    Thống kê:
    - Số lần phát hiện liên tiếp: {detection_count} lần
    - Độ tin cậy trung bình: {avg_confidence:.2%}
    - Độ tin cậy hiện tại: {prediction_confidence:.2%}
    - Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
    - Trạng thái: {label}
    
    Vui lòng kiểm tra ngay lập tức!
    """
    
    for recipient in recipients:
        send_email_async(recipient, subject, message, avg_confidence, detection_count)


def exponential_smoothing(old, new, alpha=0.3):
    return alpha * new + (1 - alpha) * old


def play_alert_sound():
    global last_beep_time
    now = time.time()
    if now - last_beep_time >= beep:
        ws.Beep(4000, 500)
        last_beep_time = now


def extract_landmarks(results):
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
    global prediction_confidence, label, is_predicting

    try:
        arr = np.array(lm_seq)[np.newaxis, ...]
        res = float(model.predict(arr, verbose=0)[0][0])

        # smoothing
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


# MAIN LOOP
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)



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

    # Prediction
    if len(lm_list) == n_time_steps and frame_counter % 3 == 0:
        if not is_predicting:
            is_predicting = True
            t = threading.Thread(target=detect, args=(model, list(lm_list)))
            t.daemon = True
            t.start()

    color = (0, 0, 255) if label == "Heart pain" else (0, 255, 0)

    # LOGIC ĐẾM SỐ LẦN PHÁT HIỆN
    if label == "Heart pain" and prediction_confidence > 0.85:
        heart_pain_count += 1
        confidence_history.append(prediction_confidence)
        
        # Giới hạn lịch sử chỉ lưu 10 giá trị gần nhất
        if len(confidence_history) > REQUIRED_DETECTIONS:
            confidence_history.pop(0)
        
        
        # Phát âm thanh mỗi lần phát hiện
        play_alert_sound()
        
        
        if heart_pain_count >= REQUIRED_DETECTIONS:
            avg_confidence = sum(confidence_history) / len(confidence_history)
            
            send_alert_emails(avg_confidence, heart_pain_count)
            
            # Reset sau khi gửi
            heart_pain_count = 0
            confidence_history = []
           
    else:
        heart_pain_count = 0
        confidence_history = []

    # Hiển thị text trên màn hình
    cv2.putText(
        frame,
        f"{label} {prediction_confidence:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color, 3
    )
    
    
    if heart_pain_count > 0:
        cv2.putText(
            frame,
            f"Detections: {heart_pain_count}/{REQUIRED_DETECTIONS}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0), 2
        )

    cv2.imshow("Heart Pain Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
