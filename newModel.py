import cv2
import mediapipe as mp
import vlc
import time

# --- RULE-BASED GESTURE DETECTION FUNCTION ---
def detect_gesture(landmarks):
    fingers_up = []
    # Thumb: Tip (4) vs IP joint (3)
    if landmarks[4].x < landmarks[3].x:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    # Index to pinky: Tip vs PIP joints
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers_up.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    # Define gestures
    if fingers_up == [1, 1, 1, 1, 1]:      # Open hand
        return "play"
    elif fingers_up == [0, 0, 0, 0, 0]:   # Fist
        return "pause"
    elif fingers_up == [0, 1, 1, 0, 0]:   # V-shape
        return "fast_forward"
    elif fingers_up == [1, 1, 0, 0, 0]:   # Ring & pinky up
        return "rewind"
    return "none"

# VLC player setup
video_path = "SampleVideo.mp4"  # Change this as needed
player = vlc.MediaPlayer(video_path)
player.play()
time.sleep(1)  # Let VLC initialize

# MediaPipe and OpenCV webcam setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

cooldown = 2  # seconds
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    now = time.time()
    gesture = "none"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            # Rule-based gesture recognition
            gesture = detect_gesture(hand.landmark)

            if now - last_time > cooldown:
                if gesture == "play":
                    player.play()
                    print("▶️ Play")
                elif gesture == "pause":
                    player.pause()
                    print("⏸️ Pause")
                elif gesture == "fast_forward":
                    # Skip ahead 10 seconds (10,000 ms)
                    player.set_time(player.get_time() + 10_000)
                    print("⏩ Forward")
                elif gesture == "rewind":
                    # Skip back 10 seconds (10,000 ms)
                    player.set_time(max(0, player.get_time() - 10_000))
                    print("⏪ Rewind")
                last_time = now

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

    # Display detected gesture on webcam window
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Control (Rule-Based)", frame)

    if cv2.waitKey(10) & 0xFF == 27:  # Esc to exit
        break

cap.release()
cv2.destroyAllWindows()