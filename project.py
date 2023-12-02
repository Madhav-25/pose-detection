
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def is_squat(hip, knee, ankle):
    angle_knee = calculate_angle(hip, knee, ankle)
    return angle_knee <= 90  # Adjust the threshold as needed

def is_push_up(shoulder, elbow, hip, knee):
    angle_elbow = calculate_angle(shoulder, elbow, hip)
    angle_knee = calculate_angle(hip, knee, hip)
    
    return angle_elbow <= 90 and angle_knee <= 90  # Adjust the thresholds as needed

def is_leg_raise(hip, knee, ankle,shoulder):
    angle_knee = calculate_angle(hip, knee, ankle)
    angle_hip = calculate_angle(shoulder, hip, knee)
    return 150 < angle_knee <= 210 and 60 < angle_hip <= 120   # Adjust the range as needed

def is_sit_up(shoulder, hip, knee):
    angle_hip = calculate_angle(shoulder, hip, knee)
    return 60 < angle_hip <= 120  # Adjust the range as needed

def is_tadasana(shoulder, hip, knee, ankle,wrist):
    angle_hip = calculate_angle(shoulder, hip, knee)
    angle_ankle = calculate_angle(hip, knee, ankle)
    angle_shoulder = calculate_angle(wrist, shoulder, hip)
    return 150 < angle_shoulder <= 210 and 150 < angle_hip <= 210 and 150 < angle_ankle <= 210  # Adjust the thresholds as needed

def is_bridge(shoulder, hip, knee, ankle):
    angle_hip = calculate_angle(shoulder, hip, knee)
    angle_ankle = calculate_angle(hip, knee, ankle)
    return 150 <=angle_hip <= 230 and 50 <=angle_ankle <= 120  # Adjust the thresholds as needed

def is_kneepush_up(shoulder, elbow, hip, knee,ankle):
    angle_elbow = calculate_angle(shoulder, elbow, hip)
    angle_knee = calculate_angle(hip, knee, ankle)
    
    return angle_elbow <= 90 and angle_knee <= 90  # Adjust the thresholds as needed

# ...

def process_webcam():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        squat_stage = None
        push_up_stage = None
        leg_raise_stage = None
        sit_up_stage = None
        tadasana_stage = None

        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Draw landmarks on the image
                for landmark in mp_pose.PoseLandmark:
                    cv2.circle(image, (int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0])), 5, (0, 255, 0), -1)

                # Draw lines connecting landmarks
                connections = [(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                               (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                               (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                               (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)]

                for connection in connections:
                    start_point = (int(landmarks[connection[0].value].x * frame.shape[1]), int(landmarks[connection[0].value].y * frame.shape[0]))
                    end_point = (int(landmarks[connection[1].value].x * frame.shape[1]), int(landmarks[connection[1].value].y * frame.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

                if is_squat(hip, knee, ankle):
                    squat_stage = "Squatting"
                else:
                    squat_stage = "Idle"

                if is_push_up(shoulder, elbow, hip, knee):
                    push_up_stage = "Push-up"
                else:
                    push_up_stage = "Idle"

                if is_leg_raise(hip, knee, ankle,shoulder):
                    leg_raise_stage = "Leg Raise"
                else:
                    leg_raise_stage = "Idle"

                if is_sit_up(shoulder, hip, knee):
                    sit_up_stage = "Sit-up"
                else:
                    sit_up_stage = "Idle"

                if is_tadasana(shoulder, hip, knee, ankle,wrist):
                    tadasana_stage = "Tadasana"
                else:
                    tadasana_stage = "Idle"

                if is_bridge(shoulder, hip, knee, ankle):
                    bridge_stage = "Glute Bridge"
                else:
                    bridge_stage = "Idle"

                if is_kneepush_up(shoulder, elbow, hip, knee,ankle):
                    kneepush_up_stage = "Knee Push-up"
                else:
                    kneepush_up_stage = "Idle"


            except:
                pass

            cv2.putText(image, f"Squat Status: {squat_stage}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Push-up Status: {push_up_stage}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Leg Raise Status: {leg_raise_stage}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Sit-up Status: {sit_up_stage}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Tadasana Status: {tadasana_stage}", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, f"Glute Bridge Status: {bridge_stage}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Knee Push-up Status: {kneepush_up_stage}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Pose Detection', image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):             
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_webcam()
