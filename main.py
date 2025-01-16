import cv2
import mediapipe as mp
import math
import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) formed by the points A, B, C
    with B as the vertex (e.g., shoulder -> elbow -> wrist).
    
    a, b, c are (x, y) coordinates in 2D.
    """
    # Convert them to vectors: BA and BC
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    # Dot product and magnitude
    dot_prod = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    # Avoid division by zero
    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    # Angle in radians
    cos_angle = dot_prod / (mag_ba * mag_bc)
    # Numerical safety for floating precision
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    angle = math.degrees(math.acos(cos_angle))
    return angle

def main():
    cap = cv2.VideoCapture(0)
    
    # For counting reps or partial reps
    stage = None  # "up" or "down"
    rep_count = 0

    # For FPS
    prev_time = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Collect data points
            data_points = {}  # e.g., {"shoulder": (x,y), "elbow": (x,y), ...}

            if results.pose_landmarks:
                # Get image dims
                h, w, _ = frame.shape

                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark

                # MediaPipe Pose indices (Right arm for example)
                #   12 - right shoulder
                #   14 - right elbow
                #   16 - right wrist
                #   24 - right hip
                #   11 - left shoulder (for torso angle reference)
                #   23 - left hip (for torso angle reference)

                # We only focus on right side for the bicep:
                right_shoulder = landmarks[12]
                right_elbow = landmarks[14]
                right_wrist = landmarks[16]
                right_hip = landmarks[24]

                # Left side for checking torso angle (optional):
                left_shoulder = landmarks[11]
                left_hip = landmarks[23]

                # Convert normalized coordinates [0,1] to pixel values
                r_shoulder_xy = (right_shoulder.x * w, right_shoulder.y * h)
                r_elbow_xy = (right_elbow.x * w, right_elbow.y * h)
                r_wrist_xy = (right_wrist.x * w, right_wrist.y * h)
                r_hip_xy = (right_hip.x * w, right_hip.y * h)
                
                l_shoulder_xy = (left_shoulder.x * w, left_shoulder.y * h)
                l_hip_xy = (left_hip.x * w, left_hip.y * h)

                data_points["right_shoulder"] = r_shoulder_xy
                data_points["right_elbow"] = r_elbow_xy
                data_points["right_wrist"] = r_wrist_xy
                data_points["right_hip"] = r_hip_xy
                data_points["left_shoulder"] = l_shoulder_xy
                data_points["left_hip"] = l_hip_xy

                # ---------------------------
                # 1. Calculate Elbow Angle
                # ---------------------------
                # Angle at the elbow: (shoulder -> elbow -> wrist)
                elbow_angle = calculate_angle(r_shoulder_xy, r_elbow_xy, r_wrist_xy)

                # ---------------------------
                # 2. Calculate Torso Angle
                # ---------------------------
                # For simplicity, let's look at the angle of the spine (shoulder -> hip)
                # We'll do right side or we can do average of left & right
                # Right side: (shoulder -> hip -> horizontal reference)
                # But simpler approach is to compare left and right shoulders vs hips to see if torso is vertical
                # Let's do angle between left_shoulder and right_shoulder, and left_hip and right_hip:
                # Actually, let's do a simpler approach: angle between shoulders and hips for "lean".
                torso_angle = calculate_angle(l_shoulder_xy, r_shoulder_xy, r_hip_xy)  # This is a rough approach
                # (Alternatively, you might measure the angle between the vector from left_hip to right_hip 
                # and left_shoulder to right_shoulder, or compare it to a horizontal line.)

                # ---------------------------
                # 3. Infer Bicep Curl Progress
                # ---------------------------
                # Typical full extension is ~160-180 degrees at the elbow, 
                # full flexion is ~30-60 degrees. We'll do a rough approach:
                full_extension_angle = 160.0
                full_flexion_angle   = 60.0
                # Clip the angle to our range
                clipped_angle = max(min(elbow_angle, full_extension_angle), full_flexion_angle)
                # 0% = fully extended, 100% = fully flexed
                curl_percent = (
                    (full_extension_angle - clipped_angle) / 
                    (full_extension_angle - full_flexion_angle) * 100
                )
                curl_percent = round(curl_percent, 1)

                # ---------------------------
                # 4. Rep Counting Logic
                # ---------------------------
                # If elbow_angle > ~150 => arm is "down"
                # If elbow_angle < ~70 => arm is "up"
                if elbow_angle > 150:
                    stage = "down"
                if elbow_angle < 70 and stage == "down":
                    stage = "up"
                    rep_count += 1

                # ---------------------------
                # 5. Display Data on Screen
                # ---------------------------
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Display elbow angle
                cv2.putText(frame, f'Elbow Angle: {int(elbow_angle)} deg',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Display curl progress
                cv2.putText(frame, f'Curl: {curl_percent}%',
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Display torso angle (roughly)
                cv2.putText(frame, f'Torso Angle (approx): {int(torso_angle)} deg',
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Display rep count
                cv2.putText(frame, f'Reps: {rep_count}',
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # (Optional) Log the data points
                # For demonstration, print them out (you could also write to CSV or store in memory)
                print("-- Data Points --")
                for k, v in data_points.items():
                    print(f"{k}: {v}")
                print(f"Elbow Angle: {elbow_angle}, Torso Angle: {torso_angle}, Curl%: {curl_percent}, Reps: {rep_count}")
                print("-----------------\n")

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {int(fps)}',
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Biceps Curl Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
