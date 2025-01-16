import cv2
import mediapipe as mp
import math
import time
import threading
import collections

from vpython import canvas, cylinder, sphere, vector, color, rate

# Import the correct class
from mediapipe.framework.formats import landmark_pb2

# -------------- 1. MediaPipe Pose + OpenCV Setup ------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# We'll store the smoothed landmarks in a global variable (also store a buffer of recent frames)
latest_landmarks_smoothed = None
landmarks_buffer = collections.deque(maxlen=5)  # Keep the last 5 frames
landmarks_lock = threading.Lock()
stop_flag = False  # to signal threads to stop

def mediapipe_thread():
    """Thread function: captures webcam, runs MediaPipe, updates global landmarks."""
    global latest_landmarks_smoothed, stop_flag

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        stop_flag = True
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while not stop_flag:
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            # Flip horizontally (mirror view)
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw landmarks on the original BGR frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                with landmarks_lock:
                    # Store the raw 33 landmarks in a buffer for smoothing
                    landmarks_buffer.append(list(results.pose_landmarks.landmark))
                    # Compute a smoothed version
                    latest_landmarks_smoothed = smooth_landmarks(list(landmarks_buffer))

            # Show camera image
            cv2.imshow("Camera + Pose", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                stop_flag = True
                break

    cap.release()
    cv2.destroyAllWindows()
    print("MediaPipe thread stopped.")


def smooth_landmarks(buffered_landmarks):
    """
    Takes a list of 'frames' of landmarks (each frame is a list of 33 landmarks),
    and returns an averaged (smoothed) list of 33 landmarks.
    We do a simple average across all frames in the buffer for each landmark coordinate.

    If a landmark has very low visibility in multiple frames, we keep the last known good.
    """
    num_frames = len(buffered_landmarks)
    if num_frames == 0:
        return None

    num_landmarks = len(buffered_landmarks[0])
    sums = [{'x':0.0, 'y':0.0, 'z':0.0, 'visibility':0.0} for _ in range(num_landmarks)]

    # Accumulate
    for frame_lm in buffered_landmarks:
        for i, lm in enumerate(frame_lm):
            sums[i]['x'] += lm.x
            sums[i]['y'] += lm.y
            sums[i]['z'] += lm.z
            sums[i]['visibility'] += lm.visibility

    # Now average
    averaged = []
    for i in range(num_landmarks):
        out_lm = landmark_pb2.NormalizedLandmark()
        out_lm.x = sums[i]['x'] / num_frames
        out_lm.y = sums[i]['y'] / num_frames
        out_lm.z = sums[i]['z'] / num_frames
        out_lm.visibility = sums[i]['visibility'] / num_frames
        averaged.append(out_lm)

    return averaged

# -------------- 2. Simple Angle Calculations -----------------------
def calc_2d_angle(ax, ay, bx, by, cx, cy):
    """
    Given three 2D points A,B,C, return the angle ABC (in degrees)
    with B as the vertex.
    """
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    dot_prod = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    cos_angle = dot_prod / (mag_ba * mag_bc)
    # Clamp to avoid floating precision errors
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg

def get_arm_angles(lm, width, height, side="right", vis_threshold=0.4):
    """
    Returns (shoulder_angle, elbow_angle) in degrees for a given side.
    If landmarks are not visible enough, we keep them from the last known angles
    or return None to indicate we skip updating.
    """
    if lm is None:
        return None

    if side == "right":
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16
    else:
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15

    # Visibility check
    if (lm[shoulder_idx].visibility < vis_threshold or
        lm[elbow_idx].visibility < vis_threshold or
        lm[wrist_idx].visibility < vis_threshold):
        return None

    sx, sy = lm[shoulder_idx].x * width, lm[shoulder_idx].y * height
    ex, ey = lm[elbow_idx].x * width, lm[elbow_idx].y * height
    wx, wy = lm[wrist_idx].x * width, lm[wrist_idx].y * height

    # Shoulder angle => (shoulder->elbow->wrist)
    shoulder_angle_deg = calc_2d_angle(wx, wy, sx, sy, ex, ey)
    # Elbow angle => (shoulder->elbow->wrist)
    elbow_angle_deg = calc_2d_angle(sx, sy, ex, ey, wx, wy)

    # Adjust angles to allow full range of motion
    if shoulder_angle_deg > 180:
        shoulder_angle_deg -= 360
    if elbow_angle_deg > 180:
        elbow_angle_deg -= 360

    return (shoulder_angle_deg, elbow_angle_deg)

# -------------- 3. VPython 3D Setup --------------------------------
def create_stick_figure():
    scene = canvas(title="Pose Tracking", width=800, height=600)
    scene.background = color.black
    scene.camera.pos = vector(0, 0, 5)
    scene.camera.axis = vector(0, 0, -1)
    
    scale = 2.0
    line_radius = 0.03 * scale
    joint_radius = 0.06 * scale
    primary_color = color.cyan
    accent_color = color.blue
    
    # Torso
    torso = cylinder(
        pos=vector(0, 0, 0),
        axis=vector(0, 1.5*scale, 0),
        radius=line_radius,
        color=primary_color,
        opacity=0.9
    )

    # Head
    head = sphere(
        pos=vector(0, 1.5*scale + 0.15*scale, 0),
        radius=0.15*scale,
        color=primary_color,
        opacity=0.8
    )

    # Right Arm
    shoulder_r = sphere(pos=vector(0.15*scale, 1.4*scale, 0), radius=joint_radius, color=accent_color, opacity=0.9)
    elbow_r = sphere(pos=vector(0.45*scale, 1.2*scale, 0), radius=joint_radius, color=accent_color, opacity=0.9)
    r_upper_arm = cylinder(pos=shoulder_r.pos, axis=vector(0.3*scale, -0.2*scale, 0), radius=line_radius, color=accent_color, opacity=0.8)
    r_forearm = cylinder(pos=elbow_r.pos, axis=vector(0.25*scale, -0.1*scale, 0), radius=line_radius, color=accent_color, opacity=0.8)

    # Left Arm
    shoulder_l = sphere(pos=vector(-0.15*scale, 1.4*scale, 0), radius=joint_radius, color=accent_color, opacity=0.9)
    elbow_l = sphere(pos=vector(-0.45*scale, 1.2*scale, 0), radius=joint_radius, color=accent_color, opacity=0.9)
    l_upper_arm = cylinder(pos=shoulder_l.pos, axis=vector(-0.3*scale, -0.2*scale, 0), radius=line_radius, color=accent_color, opacity=0.8)
    l_forearm = cylinder(pos=elbow_l.pos, axis=vector(-0.25*scale, -0.1*scale, 0), radius=line_radius, color=accent_color, opacity=0.8)

    figure_parts = {
        "torso": torso,
        "head": head,
        "shoulder_r": shoulder_r,
        "elbow_r": elbow_r,
        "r_upper_arm": r_upper_arm,
        "r_forearm": r_forearm,
        "shoulder_l": shoulder_l,
        "elbow_l": elbow_l,
        "l_upper_arm": l_upper_arm,
        "l_forearm": l_forearm,
    }

    # You can add legs, hips, etc. in a similar manner or keep it simple
    return scene, figure_parts

# -------------- 4. Updating the 3D Figure in Real-Time -------------
# We'll store the last known angles to avoid snapping if the new frame is invalid
last_right_angles = (0.0, 0.0)
last_left_angles = (0.0, 0.0)

def update_arms(figure_parts, angles_right, angles_left):
    """Update arms based on calculated angles."""
    # Right arm updates
    shoulder_r = figure_parts["shoulder_r"]
    elbow_r = figure_parts["elbow_r"]
    r_upper_arm = figure_parts["r_upper_arm"]
    r_forearm = figure_parts["r_forearm"]
    
    shoulder_r_deg, elbow_r_deg = angles_right
    shoulder_r_rad = math.radians(shoulder_r_deg)
    elbow_r_rad = math.radians(elbow_r_deg)

    # Update upper arm
    upper_arm_length = 0.4  # Standardized length
    new_ax_x = upper_arm_length * math.cos(shoulder_r_rad)
    new_ax_y = upper_arm_length * math.sin(shoulder_r_rad)
    r_upper_arm.axis = vector(new_ax_x, new_ax_y, 0)

    # Update elbow position
    elbow_r.pos = shoulder_r.pos + r_upper_arm.axis
    r_forearm.pos = elbow_r.pos

    # Update forearm
    forearm_length = 0.3  # Slightly shorter than upper arm
    new_fax_x = forearm_length * math.cos(elbow_r_rad + shoulder_r_rad)
    new_fax_y = forearm_length * math.sin(elbow_r_rad + shoulder_r_rad)
    r_forearm.axis = vector(new_fax_x, new_fax_y, 0)

    # Left arm updates
    shoulder_l = figure_parts["shoulder_l"]
    elbow_l = figure_parts["elbow_l"]
    l_upper_arm = figure_parts["l_upper_arm"]
    l_forearm = figure_parts["l_forearm"]
    
    shoulder_l_deg, elbow_l_deg = angles_left
    shoulder_l_rad = math.radians(shoulder_l_deg)
    elbow_l_rad = math.radians(elbow_l_deg)

    # Update upper arm
    new_ax_x = upper_arm_length * math.cos(shoulder_l_rad)
    new_ax_y = upper_arm_length * math.sin(shoulder_l_rad)
    l_upper_arm.axis = vector(new_ax_x, new_ax_y, 0)

    # Update elbow position
    elbow_l.pos = shoulder_l.pos + l_upper_arm.axis
    l_forearm.pos = elbow_l.pos

    # Update forearm
    new_fax_x = forearm_length * math.cos(elbow_l_rad + shoulder_l_rad)
    new_fax_y = forearm_length * math.sin(elbow_l_rad + shoulder_l_rad)
    l_forearm.axis = vector(new_fax_x, new_fax_y, 0)

def update_body(figure_parts, lm, width, height):
    """Update the body parts we care about (arms, etc.) with smoothed/valid landmarks."""
    if lm is None:
        return

    angles_right = get_arm_angles(lm, width, height, side="right")
    angles_left = get_arm_angles(lm, width, height, side="left")

    # Update arms
    update_arms(figure_parts, angles_right, angles_left)
    # If you had legs or torso logic, you'd update them similarly
    # e.g. update_legs(figure_parts, lm, width, height), etc.

# -------------- 5. Main Function / Program -------------------------
def main():
    # 1. Start the MediaPipe thread
    thread_mp = threading.Thread(target=mediapipe_thread, daemon=True)
    thread_mp.start()

    # 2. Create our VPython 3D scene
    scene, figure_parts = create_stick_figure()

    # 3. Real-time update loop (in main thread)
    while not stop_flag:
        rate(30)  # ~30 fps

        with landmarks_lock:
            lm_smoothed = latest_landmarks_smoothed

        if lm_smoothed is not None:
            # We pick a reference 2D resolution
            w, h = 640, 480
            update_body(figure_parts, lm_smoothed, w, h)

    print("Main loop stopping...")
    thread_mp.join()
    print("All done.")

if __name__ == "__main__":
    main()
