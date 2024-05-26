#%%
import os
import numpy as np
import pickle

from utils import load_image, get_faces_from_frame, get_face_encodings_from_faces

#%%
# For each directory under known_faces, get all the images and get the face encodings and calculate the average face encoding.
known_faces_dir = "./known_faces"

known_faces = {'names': [], 'encodings': []}

for person in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Processing {person}")
    person_images = os.listdir(person_dir)

    person_face_encodings = []
    for image in person_images:
        image_path = os.path.join(person_dir, image)
        image = load_image(image_path)
        faces = get_faces_from_frame(image)
        face_encodings = get_face_encodings_from_faces(faces)
        person_face_encodings.extend(face_encodings[0])
    person_face_encodings = np.array(person_face_encodings)

    person_average_face_encoding = np.mean(person_face_encodings, axis=0)
    
    known_faces['names'].append(person)
    known_faces['encodings'].append(person_average_face_encoding)

# Save the known_faces dict to a file
file_path = os.path.join(known_faces_dir, "known_faces.pickle")
with open(file_path, 'wb') as f:
    pickle.dump(known_faces, f)
print("Saved known_faces to a file")

# %%
