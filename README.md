# search-face

**search-face** is a tool designed for those who need to find specific or matching faces from a large collection of photos using facial recognition.

## Get started and Demo

1. Clone the repository

   ```bash
   git clone https://github.com/tanakatsu/search-face.git
   cd search-face
    ````

1. Create a virtual environment (optional but recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

1. Get sample photo

   ```bash
   wget https://raw.githubusercontent.com/deepinsight/insightface/refs/heads/master/python-package/insightface/data/images/t1.jpg
   ```

1. Detect faces

   ```bash
   python detect_faces.py t1.jpg
   ```
   This will create a `detected` directory with the detected face features (`face_*.npy`) and an annotated image (`result.jpg`).

    ```
    ├── detected/    # Note: You change directory name by --output_dir option
    │   ├── face_1.npy
    │   ├── face_2.npy
    │   ├── face_3.npy
    │   ├── face_4.npy
    │   ├── face_5.npy
    │   ├── face_6.npy
    │   └── result.jpg
    ```

1. Register faces

    Let's say you want to register the detected faces. You can move the `.npy` files to a subdirectory of `registered_faces` named for the user:

    ```
    ├── registered_faces/
    │   └── user_A/    # Note: Multiple faces can be registered
    │   │   └── face_1.npy
    │   └── user_B/
    │       └── face_2.npy
    ```

1. Create an album directory

    ```bash
    mkdir sample_album
    mv t1.jpg sample_album/
    ```

    ```
    ├── sample_album/  # Of course, Album can contain multiple photos
    │   └── t1.jpg
    ```

1. Identify faces
    ```bash
    python identify_faces.py sample_album  # Note: You can change matching level by --threshold option
    ```

    This will create an `output` directory with the identified faces photos and a `_confirm` directory for confirmation (annotated images).
    ```
    ├── output/
    │   └── user_A/
    │   │   └── t1.jpg
    │   └── user_B/
    │   │   └── t1.jpg
    │   └── _confirm/
    │       └── user_A/
    │       │   └── t1.jpg
    │       └── user_B/
    │           └── t1.jpg
    ```

1. Check faces similarites with registered faces

    ```bash
    python check_similarity.py -u user_A sample_album/t1.jpg
    ```

    This will output the similarity scores of the faces in `sample_album/t1.jpg` against the registered faces of `user_A`.

    ```
    Found 6 faces.
    face_1: 1.000
    face_2: 0.064
    face_3: 0.036
    face_4: -0.010
    face_5: -0.029
    face_6: -0.083
    ```

    The scores are between -1 and 1, where 1 means the same face and -1 means completely different faces. The closer to 1, the more similar the faces are.

    Also, `output.jpg` will be created with the detected faces and names (`face_1`, `face_2`, etc.)
