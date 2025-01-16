import cv2
import mediapipe as mp
import time
import math
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
left_wrist_history = deque(maxlen=10)
right_wrist_history = deque(maxlen=10)
def compute_velocity(history_deque):
    """
    Given a deque of (x, y, timestamp) for the last frames,
    compute approximate 2D velocity (pixels/sec).
    """
    if len(history_deque) < 2:
        return 0.0, 0.0
    x1, y1, t1 = history_deque[0]
    x2, y2, t2 = history_deque[-1]
    dt = t2 - t1
    if dt == 0:
        return 0.0, 0.0
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    return vx, vy

def angle_2d(ax, ay, bx, by, cx, cy):
    """
    Returns the angle (in degrees) formed by points A, B, C
    with B as the vertex.
    """
    BA = (ax - bx, ay - by)
    BC = (cx - bx, cy - by)
    dot_prod = BA[0]*BC[0] + BA[1]*BC[1]
    mag_ba = math.sqrt(BA[0]**2 + BA[1]**2)
    mag_bc = math.sqrt(BC[0]**2 + BC[1]**2)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = dot_prod / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def compute_angle_vertical(x1, y1, x2, y2):
    """
    Computes angle between a line and vertical axis
    Returns angle in degrees
    """
    vertical = (x1, y1 - 100) 
    return angle_2d(vertical[0], vertical[1], x1, y1, x2, y2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam.")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        prev_time = time.time()

        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

            data_dict = {}
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if len(lm) > 32:
                    nose = lm[0]
                    neck = lm[12] 
                    neck_angle = compute_angle_vertical(
                        neck.x * w, neck.y * h,
                        nose.x * w, nose.y * h
                    )
                    data_dict["NeckTilt(deg)"] = round(neck_angle, 2)
                    l_shoulder = lm[11]
                    r_shoulder = lm[12]
                    shoulder_height_diff = abs(l_shoulder.y - r_shoulder.y) * h
                    data_dict["ShoulderAsymmetry(px)"] = round(shoulder_height_diff, 2)
                    l_shoulder = lm[11]
                    l_elbow = lm[13]
                    l_wrist = lm[15]
                    left_elbow_angle = angle_2d(
                        l_shoulder.x * w, l_shoulder.y * h,
                        l_elbow.x * w, l_elbow.y * h,
                        l_wrist.x * w, l_wrist.y * h
                    )
                    data_dict["LeftElbowAngle(deg)"] = round(left_elbow_angle, 2)
                    r_shoulder = lm[12]
                    r_elbow = lm[14]
                    r_wrist = lm[16]
                    right_elbow_angle = angle_2d(
                        r_shoulder.x * w, r_shoulder.y * h,
                        r_elbow.x * w, r_elbow.y * h,
                        r_wrist.x * w, r_wrist.y * h
                    )
                    data_dict["RightElbowAngle(deg)"] = round(right_elbow_angle, 2)
                    l_hip = lm[23]
                    r_hip = lm[24]
                    hip_angle = compute_angle_vertical(
                        l_hip.x * w, l_hip.y * h,
                        r_hip.x * w, r_hip.y * h
                    )
                    data_dict["HipTilt(deg)"] = round(hip_angle, 2)
                    neck_mid = ((l_shoulder.x + r_shoulder.x)/2, (l_shoulder.y + r_shoulder.y)/2)
                    hip_mid = ((l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2)
                    spine_length = math.sqrt(
                        (neck_mid[0] - hip_mid[0])**2 + 
                        (neck_mid[1] - hip_mid[1])**2
                    ) * h
                    data_dict["SpineLength(px)"] = round(spine_length, 2)
            y_pos = 30
            for key, value in data_dict.items():
                if isinstance(value, tuple):
                    text = f"{key}: ({value[0]}, {value[1]})"
                else:
                    text = f"{key}: {value}"
                cv2.putText(frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25

            print("---- Frame Data ----")
            for k, v in data_dict.items():
                print(f"{k}: {v}")
            print("--------------------\n")
            cv2.imshow("Holistic Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
