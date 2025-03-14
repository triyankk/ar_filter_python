import cv2
import mediapipe as mp
import os
import imageio
from settings import Settings

# Initialize Mediapipe face and hand detection
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=Settings.MIN_DETECTION_CONFIDENCE)
hands = mp_hands.Hands(min_detection_confidence=Settings.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=Settings.MIN_TRACKING_CONFIDENCE)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=Settings.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=Settings.MIN_TRACKING_CONFIDENCE)

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Load overlay GIF
overlay_gif_path = os.path.join(Settings.OVERLAY_PATH, 'cat.gif')
overlay_gif = imageio.mimread(overlay_gif_path)
overlay_gif = [cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA) for img in overlay_gif]
gif_index = 0

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay img_overlay on top of img at (x, y) and blend using alpha_mask."""
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if no part of the overlay is in the frame
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    img_crop[:] = alpha * img_overlay_crop[:, :, :3] + (1.0 - alpha) * img_crop

def is_hand_open(hand_landmarks):
    """Determine if the hand is open and palm is facing upward."""
    # Check if the hand is open by comparing the y-coordinates of the landmarks
    # and if the palm is facing upward by comparing the z-coordinates
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    palm_base = hand_landmarks.landmark[0]

    is_open = (thumb_tip.y < palm_base.y and
               index_tip.y < palm_base.y and
               middle_tip.y < palm_base.y and
               ring_tip.y < palm_base.y and
               pinky_tip.y < palm_base.y)

    is_facing_upward = (thumb_tip.z < palm_base.z and
                        index_tip.z < palm_base.z and
                        middle_tip.z < palm_base.z and
                        ring_tip.z < palm_base.z and
                        pinky_tip.z < palm_base.z)

    return is_open and is_facing_upward

def calculate_hand_area(hand_landmarks, image_shape):
    """Calculate the surface area of the hand."""
    h, w, _ = image_shape
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
    hand_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
    return hand_area

def generate_frames():
    global gif_index
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame for face detection
            face_results = face_detection.process(rgb_frame)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    cv2.rectangle(frame, bbox, (255, 0, 0), 2)
            
            # Process the frame for hand detection
            hand_results = hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if is_hand_open(hand_landmarks):
                        # Get the coordinates of the wrist (landmark 0)
                        wrist = hand_landmarks.landmark[0]
                        h, w, c = frame.shape
                        cx, cy = int(wrist.x * w), int(wrist.y * h)
                        
                        # Calculate the hand area and adjust the overlay size
                        hand_area = calculate_hand_area(hand_landmarks, frame.shape)
                        overlay_size = int((hand_area ** 0.5) * Settings.OVERLAY_SIZE_FACTOR)
                        resized_overlay = cv2.resize(overlay_gif[gif_index], (overlay_size, overlay_size))
                        
                        # Adjust coordinates to center the overlay image on the wrist
                        cx -= resized_overlay.shape[1] // 2  # Adjust horizontal alignment
                        cy -= resized_overlay.shape[0] // 2  # Adjust vertical alignment
                        
                        # Apply additional offsets
                        cx += Settings.OVERLAY_HORIZONTAL_OFFSET
                        cy -= Settings.OVERLAY_VERTICAL_OFFSET
                        
                        # Overlay the GIF frame on the wrist
                        overlay_image_alpha(frame, resized_overlay, cx, cy, resized_overlay[:, :, 3])
                        
                        # Update GIF frame index
                        gif_index = (gif_index + 1) % len(overlay_gif)
            
            # Process the frame for face mesh
            face_mesh_results = face_mesh.process(rgb_frame)
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')