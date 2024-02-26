import numpy as np
from constants import Text

# Helper function to calculate distance (adjust the face width and field of view as needed)
def calculate_distance(bbox_width, image_width, face_width, fov, DEBUG=False):
    """
    Calculate the distance to the face based on the width of the bounding box.
    
    :param bbox_width: The width of the bounding box in pixels.
    :param image_width: The width of the image in pixels.
    :param face_width: The real-world width of the face (in meters). Default is 0.15m.
    :param fov: The horizontal field of view of the camera in degrees. Default is 60 degrees.
    :return: The estimated distance in meters.
    """
    # Calculate the focal length based on the FOV and image width
    focal_length = (image_width / 2) / np.tan(np.radians(fov / 2))
    
    # Estimate distance using the known width of the face and the perceived width in pixels
    distance = (face_width * focal_length) / bbox_width

    if DEBUG:
        print(Text.DISTANCE_TEXT + str(distance))
    
    return distance

# Helper function to detect significant movement
def detect_movement(current_position, last_position, threshold, DEBUG=False):
    """
    detects the movement of the face exceeding the movement threshold.
    
    :param current_position: current face position.
    :param last_position: last position of the face.
    :param the threshold to consider a movement.
    :return: bool value if there is a movment exceeding the threshold.
    """
    if last_position is None:
        if DEBUG :
            print(Text.LAST_POSITION_IS_NONE)
        return False
    # Calculating the deference between the current and last position.
    movement = np.linalg.norm(np.array(current_position) - np.array(last_position))
    return movement > threshold
