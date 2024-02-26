import cv2
import mediapipe as mp
from helper import Functions
from constants import Parameters, Text

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=Parameters.MIN_DETECTION_CONFIDENCE)

# Debug mod
DEBUG = False

# Start capturing video from the first webcam
cap = cv2.VideoCapture(0)

last_position = None
blur = True

# The main while loop; runs til the escape button is hit.
while cap.isOpened():
    success, image = cap.read()
    if not success:
        if DEBUG:
            print(Text.Empty_CAM_FRAME)
        continue
    
    # Flipping image to create a mirror effect.
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False

    results = face_detection.process(image)

    # Converting colors back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #The blur effect applys in case there is no face detected.
    if blur:
        # Apply blur effect
        image = cv2.blur(image, (Parameters.BLUR_INTENSITY, Parameters.BLUR_INTENSITY))
    if results.detections:
        for detection in results.detections:
            # Draw detection frame in debug mod.
            if DEBUG:
                mp_drawing.draw_detection(image, detection)
            
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            current_position = (x + w // 2, y + h // 2)
            distance = Functions.calculate_distance(w,iw,Parameters.FACE_WIDTH,Parameters.FIELD_OF_VIEW,DEBUG=DEBUG)

            # Check if face is too close or too far or moving too much.
            if distance < Parameters.DISTANCE_THRESHOLD[0] or distance > Parameters.DISTANCE_THRESHOLD[1]:
                blur = True
            elif Functions.detect_movement(current_position, last_position, Parameters.MOVEMENT_THRESHOLD, DEBUG=DEBUG):
                blur = True
            else:
                blur = False

            last_position = current_position


    # Display the resulting image
    cv2.imshow(Text.TITLE, image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
