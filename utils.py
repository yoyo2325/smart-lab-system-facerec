import face_recognition
import cv2
import numpy as np

def load_image(image_path: str, resize_ratio: float = 1) -> np.ndarray:
    """
    Load an image from a file path.
    """

    image = cv2.imread(image_path)

    if resize_ratio != 1:
        image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)

    assert image is not None, f"Failed to load image from path: {image_path}"

    return image

def get_frame_rgb(video_capture: cv2.VideoCapture, continious: bool = False, resize_ratio: float = 1) -> np.ndarray:
    """
    Get a single frame from the video capture and return the frame in RGB format.
    """

    ret, frame = video_capture.read()

    if resize_ratio != 1:
        frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)

    if not continious:
        video_capture.release()

    assert ret, "Failed to read frame from video capture."

    return frame[:, :, ::-1]

def get_faces_from_frame(frame: np.ndarray) -> np.ndarray:
    """
    Get all faces from a frame.
    """

    face_locations = face_recognition.face_locations(frame)
    faces = []
    for (top, right, bottom, left) in face_locations:
        face_frame = frame[top:bottom, left:right]
        faces.append(face_frame)

    return np.array(faces)

def get_face_encodings_from_faces(faces: np.ndarray) -> np.ndarray:
    """
    Get the face encodings from a list of faces.
    """
    face_encodings = []
    for face in faces:
        face_encoding = face_recognition.face_encodings(face)
        face_encodings.append(face_encoding) 

    return np.array(face_encodings)