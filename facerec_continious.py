#%%
import cv2
import numpy as np
import face_recognition
import pickle
from utils import get_frame_rgb, get_faces_from_frame, get_face_encodings_from_faces
import time

#%%
# Load the known_faces from the pickle file
file_path = "./known_faces/known_faces.pickle"
with open(file_path, 'rb') as f:
    known_faces = pickle.load(f)

# Print known persons
print("Known persons:")
for person in known_faces['names']:
    print(person)
print()

#%%
# Continiously recognize the faces from the video capture

# Open the video capture
video_capture = cv2.VideoCapture(0)

# Setup tolerance for face recognition. Lower is more strict.
tolerance = 0.5

while True:
    # Start the timer for calculating the frames per second
    start = time.time()

    # Get the frame from the video capture
    frame = get_frame_rgb(video_capture, continious=True)

    # Recognize the faces from the frame
    name = "Unknown"
    try:
        faces = get_faces_from_frame(frame)
        face_encoding = get_face_encodings_from_faces(faces)[0] # Only handle the first face in current implementation
        face_distances = face_recognition.face_distance(known_faces['encodings'], face_encoding)
        best_match_index = np.argmin(face_distances) # Get the index of the best match

        # If the best match is within the tolerance, set the name
        if face_distances[best_match_index] < tolerance:
            name = known_faces['names'][best_match_index]
    except:
        pass
    
    # Calculate the recognized frames per second
    interval = time.time() - start
    fps = 1 / interval

    print(f"FPS: {fps:.2f} - Found {name}        ", end="\r")

#%%