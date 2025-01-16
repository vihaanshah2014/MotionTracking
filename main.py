import cv2
import mediapipe as mp
import time
import math
from collections import deque

# -------------------------- 1. Setup Holistic & Drawing --------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# For optional drawing styles (e.g., face keypoints in different colors)
mp_drawing_styles = mp.solutions.drawing_styles

# We'll track the wrist positions over time to compute velocity
# Deques store the last few frames of (x, y, t) for each wrist
left_wrist_history = deque(maxlen=10)
right_wrist_history = deque(maxlen=10)

# Utility function to compute velocity from position history
def compute_velocity(history_deque):
    """
    Given a deque of (x, y, timestamp) for the last frames,
    compute approximate 2D velocity (pixels/sec).
    """
    if len(history_deque) < 2:
        return 0.0, 0.0
    
    # Compare the latest point with an older point
    x1, y1, t1 = history_deque[0]
    x2, y2, t2 = history_deque[-1]
    dt = t2 - t1
    if dt == 0:
        return 0.0, 0.0
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    return vx, vy

# A quick helper to compute a 2D angle between three points (A-B-C) at B
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
    # clamp cos_angle to [-1,1] to avoid floating precision errors
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

# -------------------------- 2. Main Loop --------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam.")
        return

    # Create a Holistic object. Setting refine_face_landmarks=True for better face tracking
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

            # Flip horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Holistic
            results = holistic.process(frame_rgb)

            # ---------------------- 2A. Drawing Landmarks ----------------------
            # Draw pose, face and hands landmarks on the BGR frame (in-place)
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

            # ---------------------- 2B. Compute Metrics (Head Turn, Back Alignment, etc.) ----------------------
            # We'll gather some data in a dictionary to print each frame
            data_dict = {}

            # 1) Head Turn Angle:
            #    We can approximate by using face landmarks, e.g. left ear & right ear & nose,
            #    or we can use pose landmarks for left/right ear (index 7,8).
            #    If using face_landmarks, we might pick certain points. Let's do pose for simplicity: 
            #       7: left_ear, 8: right_ear, 0: nose
            if results.pose_landmarks:
                poseLms = results.pose_landmarks.landmark
                if len(poseLms) > 8:
                    # We'll do an angle at the nose: (left_ear -> nose -> right_ear)
                    left_ear = poseLms[7]
                    right_ear = poseLms[8]
                    nose = poseLms[0]
                    # Convert to pixel coords
                    lx, ly = left_ear.x * w, left_ear.y * h
                    rx, ry = right_ear.x * w, right_ear.y * h
                    nx, ny = nose.x * w, nose.y * h
                    head_angle = angle_2d(lx, ly, nx, ny, rx, ry)
                    data_dict["HeadTurn(deg)"] = round(head_angle, 2)

            # 2) Back Alignment:
            #    Let's approximate with shoulders vs. hips alignment (pose indices: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip).
            #    We'll compute angle at shoulders or overall tilt vs. vertical.
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if len(lm) > 24:
                    # We'll do an angle between the line (left_shoulder -> right_shoulder) and (left_hip -> right_hip)
                    ls = lm[11]  # left_shoulder
                    rs = lm[12]  # right_shoulder
                    lh = lm[23]  # left_hip
                    rh = lm[24]  # right_hip

                    # midpoint of shoulders
                    msx = (ls.x + rs.x) / 2 * w
                    msy = (ls.y + rs.y) / 2 * h
                    # midpoint of hips
                    mhx = (lh.x + rh.x) / 2 * w
                    mhy = (lh.y + rh.y) / 2 * h

                    # We'll measure the angle between the vertical line and the line from mid-hip to mid-shoulders
                    # Let's define a "vertical" point above mid-hip
                    topx, topy = mhx, mhy - 100  # 100 pixels above hips
                    back_angle = angle_2d(topx, topy, mhx, mhy, msx, msy)
                    data_dict["BackAlignment(deg)"] = round(back_angle, 2)

            # 3) Wrist Movement:
            #    We'll track left wrist (pose idx 15) and right wrist (16) positions in a history to compute velocity.
            #    If you want finger-based wrist, you can also get them from the hand solution (landmark 0 in that).
            curr_time = time.time()
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # left wrist idx=15, right wrist idx=16
                if len(lm) > 16:
                    lw = lm[15]
                    rw = lm[16]
                    lwx, lwy = lw.x * w, lw.y * h
                    rwx, rwy = rw.x * w, rw.y * h

                    # Append to history
                    left_wrist_history.append((lwx, lwy, curr_time))
                    right_wrist_history.append((rwx, rwy, curr_time))

                    # Compute velocity
                    lvx, lvy = compute_velocity(left_wrist_history)
                    rvx, rvy = compute_velocity(right_wrist_history)
                    data_dict["LeftWristVel(px/s)"] = (round(lvx,1), round(lvy,1))
                    data_dict["RightWristVel(px/s)"] = (round(rvx,1), round(rvy,1))

            # 4) Finger Placement:
            #    We can track each hand in detail from results.left_hand_landmarks, results.right_hand_landmarks
            #    Each has 21 landmarks. We'll just print their coordinates or do some logic if you want.
            #    Landmark indexing for hands: 0=wrist, 4=thumb tip, 8=index tip, etc.
            #    We'll store them in data_dict
            if results.left_hand_landmarks:
                left_hand_points = []
                for i, mark in enumerate(results.left_hand_landmarks.landmark):
                    xh, yh = mark.x * w, mark.y * h
                    left_hand_points.append((i, round(xh,1), round(yh,1)))
                data_dict["LeftHandFingers"] = left_hand_points

            if results.right_hand_landmarks:
                right_hand_points = []
                for i, mark in enumerate(results.right_hand_landmarks.landmark):
                    xh, yh = mark.x * w, mark.y * h
                    right_hand_points.append((i, round(xh,1), round(yh,1)))
                data_dict["RightHandFingers"] = right_hand_points

            # ---------------------- 2C. Display & Print Data ----------------------
            # Show the data on console each frame
            print("---- Frame Data ----")
            for k, v in data_dict.items():
                print(f"{k}: {v}")
            print("--------------------\n")

            # Optionally overlay some text on the frame
            cv2.putText(frame, f"HeadAngle: {data_dict.get('HeadTurn(deg)',0):.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"BackAngle: {data_dict.get('BackAlignment(deg)',0):.2f}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Show the final frame
            cv2.imshow("Holistic Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
