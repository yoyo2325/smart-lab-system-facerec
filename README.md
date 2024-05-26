# smart-lab-system-facerec

## Installation
Please make sure you have already installed conda. If not, please download it from [here](https://www.anaconda.com/products/distribution).
Follow the steps below to install the required packages.
1. `conda create -n facerec python`
2. `conda activate facerec`
3. `conda install -c conda-forge dlib`
4. `pip install face_recognition opencv-pytho`

## Usage
### Make Known Faces Embeddings
1. Put the images of known faces in the `known_faces` folder. Please refer to the following structure.
    ```
    known_faces
    ├── [NAME_OF_PERSON_1]
    │   ├── person1_1.jpg
    │   ├── person1_2.jpg
    │   └── ...
    ├── [NAME_OF_PERSON_1]
    │   ├── person2_1.jpg
    │   ├── person2_2.jpg
    │   └── ...
    └── ...
    ```
    Note that the name of files under each folder don't matter, you can name them whatever you want (`*.jpg`).
2. Run the following command to generate the pre-calculated embeddings pickle of known faces.
    ```
    python make_known_faces_embeddings.py
    ```
    The embeddings will be saved in the `known_faces` folder as `known_faces_embeddings.pickle`.
3. Finally, you can run the face recognition script by running the following command.
    ```
    python face_recognition.py
    ```
