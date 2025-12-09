import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def make_landmark_pose(results):
    """Trả về list [x, y, z, visibility] cho 33 pose landmarks"""
    if not results.pose_landmarks:
        return None
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
        
    return c_lm
def make_landmark_hand_left(result):
    """Trả về list [x, y, z] cho 21 left hand landmarks"""
    if not result.left_hand_landmarks:
        return None
    c_lm_hl = []
    for lm_hl in result.left_hand_landmarks.landmark:              
        c_lm_hl.extend([lm_hl.x, lm_hl.y, lm_hl.z])
        
    return c_lm_hl

def make_landmark_hand_right(results):
    """Trả về list [x, y, z] cho 21 right hand landmarks"""
    if not results.right_hand_landmarks:
        return None
    c_lm_hr = []
    for lm_hr in results.right_hand_landmarks.landmark:
        c_lm_hr.extend([lm_hr.x, lm_hr.y, lm_hr.z])
        
    return c_lm_hr
def draw_all_landmarks(mp_drawing, results, img):
    """Vẽ tất cả landmarks lên frame"""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
    
    return img
def extract_landmarks(video_path, output_csv,show_video=True,save_video=False):
    out = os.makedirs("painvideo",exist_ok=True)
    

#nap video
    cap = cv2.VideoCapture(video_path)
    
    lm_list = []
    
    

    with mp_holistic.Holistic(
        static_image_mode = False,
        model_complexity = 2,
        enable_segmentation = False,
        refine_face_landmarks = True,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            #chuyen BGR sang RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            
            
            
            successful_left_hand = 0
            successful_right_hand = 0
            
            
            #Lấy landmarks
            pose_lm = make_landmark_pose(results)
            left_hand_lm = make_landmark_hand_left(results)
            right_hand_lm = make_landmark_hand_right(results)

            if pose_lm:
                row = []
                row.extend(pose_lm)
                row.extend(left_hand_lm if left_hand_lm else [0.0] * 63)
                row.extend(right_hand_lm if right_hand_lm else [0.0] * 63)
                lm_list.append(row)
                if left_hand_lm:
                    successful_left_hand += 1
                if right_hand_lm:
                    successful_right_hand += 1
            if show_video or save_video:
                    frame = draw_all_landmarks(mp_drawing, results, frame)
                
                # Hiển thị thông tin
                    cv2.putText(frame, f"Left Hand: {successful_left_hand}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right Hand: {successful_right_hand}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if show_video:
                cv2.imshow("Holistic Detection", frame)
                cv2.waitKey(1) & 0xFF == ord('q')
                   
    
    cap.release()
    df = pd.DataFrame(lm_list)
    if show_video:
        cv2.destroyAllWindows()
    df.to_csv(output_csv , index=False)
    print(f"Landmarks extracted and saved to {output_csv}")
extract_landmarks("dautim_video/video20_pain1.mp4","painout/dautim25.csv",show_video=True,save_video=False)